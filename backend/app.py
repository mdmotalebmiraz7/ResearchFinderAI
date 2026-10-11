"""Research Finder API (fixed).

Run locally:
    pip install -r requirements.txt
    uvicorn research_finder:app --reload --host 0.0.0.0 --port 8000

Run directly (terminal or Colab `%run research_finder.py`):
    python research_finder.py

Endpoints:
    GET  /health
    GET  /logs
    POST /research         {"idea": "...", "api_key": "..."}
    POST /paper            {"rank": 1}
    GET  /paper/pdf?rank=1
    POST /paper/process    {"rank": 1}
    POST /ask              {"question": "...", "api_key": "..."}
    GET  /docs
"""
import asyncio
import json
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, TypedDict

import fitz
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util


# -------------------- Configuration --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
YEAR_FROM = 2020
PAPERS_PER_TOPIC = 100
TITLE_TOP_K = 20
FINAL_TOP_K = 10
MAX_SEARCH_PAGES = 5
NO_CONTEXT_MESSAGE = "The provided context does not contain relevant information to answer this question."

# In-memory prototype storage. Results are lost when the server restarts.
# NOTE: global state means one active session at a time (not multi-user safe).
result = None
selected_paper = None
chunks: list[Document] = []
vector_store = None
retriever = None
logs: list[dict] = []
log_lock = threading.Lock()


def add_log(message: str, level: str = "INFO") -> None:
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
    }
    with log_lock:
        logs.append(entry)
        if len(logs) > 500:
            del logs[:-500]
    print(f"[{level}] {message}")


semantic_model = None
embeddings = None
models_ready = threading.Event()
model_error: str | None = None


def load_models() -> None:
    """Load heavy models in the background so the server starts answering /health at once."""
    global semantic_model, embeddings, model_error
    try:
        add_log("Loading semantic ranking model...")
        semantic_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        add_log("Semantic ranking model loaded. Loading embedding model...")
        # This model may download weights the first time the API starts.
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        models_ready.set()
        add_log("All models loaded. Research Finder API is ready.")
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"
        add_log(f"Model loading failed: {model_error}", "ERROR")


def require_models() -> None:
    if model_error:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {model_error}")
    if not models_ready.is_set():
        raise HTTPException(status_code=503, detail="Models are still loading. Check /health and try again shortly.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    add_log("Server started. Loading models in the background...")
    threading.Thread(target=load_models, daemon=True).start()
    yield


app = FastAPI(title="Research Finder API", version="1.0.2", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend origin in production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- Request and graph schemas --------------------

class ResearchRequest(BaseModel):
    idea: str = Field(min_length=3, max_length=2000)
    api_key: str = Field(min_length=10)


class RankRequest(BaseModel):
    rank: int = Field(ge=1, le=10)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    api_key: str = Field(min_length=10)


class State(TypedDict):
    idea: str
    summary: str
    titles: list[str]
    search_results: Annotated[list[list[dict]], list.__add__]
    papers: list[dict]
    final_results: list[dict]
    api_key: str


class WorkerState(TypedDict):
    title: str
    summary: str
    api_key: str


class ResearchPlan(BaseModel):
    summary: str
    titles: list[str]


# -------------------- Paper search and ranking --------------------

def reconstruct_abstract(data) -> str:
    if not data:
        return ""
    words = []
    for word, positions in data.items():
        for position in positions:
            words.append((position, word))
    words.sort()
    return " ".join(word for _, word in words)


def search_papers(
    query: str,
    search_limit: int = 100,
    year_from: int = 2020,
) -> list[dict]:
    if not query.strip():
        return []

    add_log(f"Searching OpenAlex for: {query}")

    papers: list[dict] = []
    cursor = "*"
    pages = 0

    while len(papers) < search_limit and pages < MAX_SEARCH_PAGES:
        pages += 1
        params = {
            "search": query,
            "filter": f"from_publication_date:{year_from}-01-01",
            "per-page": 100,
            "cursor": cursor,
        }

        try:
            response = requests.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            add_log(f"OpenAlex search failed for '{query}': {exc}", "ERROR")
            raise

        for item in data.get("results", []):
            title = item.get("title") or ""
            if not title:
                continue

            best_oa = item.get("best_oa_location") or {}
            primary = item.get("primary_location") or {}

            pdf_url = best_oa.get("pdf_url") or primary.get("pdf_url")
            if not pdf_url:  # skip papers without a PDF link
                continue

            doi = item.get("doi") or ""
            link = (
                doi
                or best_oa.get("landing_page_url")
                or primary.get("landing_page_url")
                or item.get("id", "")
            )

            papers.append(
                {
                    "title": title,
                    "abstract": reconstruct_abstract(item.get("abstract_inverted_index")),
                    "year": item.get("publication_year"),
                    "citations": item.get("cited_by_count", 0),
                    "doi": doi,
                    "url": link,
                    "link": link,
                    "openalex_url": item.get("id", ""),
                    "pdf_url": pdf_url,
                }
            )

            if len(papers) >= search_limit:
                break

        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            break

    add_log(f"Found {len(papers)} papers for topic: {query}")
    return papers


async def semantic_title_rank(
    papers: list[dict], query: str, top_k: int = TITLE_TOP_K
) -> list[dict]:
    if not papers:
        return []

    def rank() -> list[dict]:
        query_vector = semantic_model.encode(query, convert_to_tensor=True)
        titles = [paper.get("title", "") for paper in papers]
        title_vectors = semantic_model.encode(titles, convert_to_tensor=True)
        scores = util.cos_sim(query_vector, title_vectors)[0]
        indices = scores.argsort(descending=True)[:top_k]
        ranked = []
        for idx in indices:
            i = int(idx)
            item = papers[i].copy()
            item["Semantic"] = float(scores[i])
            ranked.append(item)
        return ranked

    return await asyncio.to_thread(rank)


async def semantic_abstract_rank(
    papers: list[dict], summary: str, top_k: int = FINAL_TOP_K
) -> list[dict]:
    if not papers:
        return []

    def rank() -> list[dict]:
        query_vector = semantic_model.encode(summary, convert_to_tensor=True)
        abstracts = [paper.get("abstract") or paper.get("title", "") for paper in papers]
        abstract_vectors = semantic_model.encode(abstracts, convert_to_tensor=True)
        scores = util.cos_sim(query_vector, abstract_vectors)[0]
        indices = scores.argsort(descending=True)[:top_k]

        ranked = []
        for rank_num, idx in enumerate(indices, start=1):
            i = int(idx)
            paper = papers[i]
            url = paper.get("url") or paper.get("link") or ""
            ranked.append(
                {
                    "Rank": rank_num,
                    "title": paper.get("title", ""),
                    "link": url,
                    "url": url,
                    "pdf_url": paper.get("pdf_url") or url.replace("/abs/", "/pdf/"),
                    "abstract": paper.get("abstract", ""),
                    "year": paper.get("year"),
                    "doi": paper.get("doi", ""),
                    "openalex_url": paper.get("openalex_url", ""),
                    "Semantic": float(scores[i]),
                }
            )
        return ranked

    return await asyncio.to_thread(rank)


# -------------------- LangGraph research workflow --------------------

PLAN_SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Create a concise research summary and exactly 10 distinct academic search topics. "
    'Return ONLY a JSON object of the form {"summary": "...", "titles": ["...", ...]} '
    "with exactly 10 strings in titles."
)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "titles": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "titles"],
    "additionalProperties": False,
}


def _parse_plan(content: str) -> ResearchPlan:
    data = json.loads(content)
    summary = str(data.get("summary", "")).strip()
    seen, titles = set(), []
    for t in data.get("titles", []):
        if isinstance(t, str) and t.strip() and t.strip().lower() not in seen:
            seen.add(t.strip().lower())
            titles.append(t.strip())
    if not summary:
        raise ValueError("The model returned an empty summary.")
    if len(titles) < 10:
        raise ValueError(f"Expected 10 research topics, got {len(titles)}.")
    return ResearchPlan(summary=summary, titles=titles[:10])


def generate_research_plan(idea: str, api_key: str) -> ResearchPlan:
    client = Groq(api_key=api_key)
    # Try strict schema first, then fall back to plain JSON mode.
    modes = ["json_schema", "json_schema", "json_object"]
    last_error: Exception | None = None

    for attempt, mode in enumerate(modes, start=1):
        if mode == "json_schema":
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "research_plan", "strict": True, "schema": PLAN_SCHEMA},
            }
        else:
            response_format = {"type": "json_object"}

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Research idea:\n{idea}"},
                ],
                response_format=response_format,
                temperature=0.2,
                max_completion_tokens=2500,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("The model returned an empty response.")
            return _parse_plan(content)
        except Exception as exc:
            last_error = exc
            add_log(f"Research plan attempt {attempt} ({mode}) failed: {exc}", "WARNING")

    raise ValueError(f"Could not generate a research plan: {last_error}")


def generate(state: State) -> dict:
    plan = generate_research_plan(state["idea"], state["api_key"])
    return {"summary": plan.summary, "titles": plan.titles}


def dispatch_search(state: State):
    return [
        Send(
            "process_title",
            {"title": title, "summary": state["summary"], "api_key": state["api_key"]},
        )
        for title in state["titles"]
    ]


async def process_title(state: WorkerState) -> dict:
    title = state["title"]
    try:
        papers = await asyncio.to_thread(search_papers, title, PAPERS_PER_TOPIC, YEAR_FROM)
    except Exception as exc:
        # One failing topic must not kill the whole research run.
        add_log(f"Skipping topic '{title}': {exc}", "WARNING")
        return {"search_results": [[]]}
    ranked = await semantic_title_rank(papers, title, TITLE_TOP_K)
    return {"search_results": [ranked]}


def collect_unique_papers(state: State) -> dict:
    unique: dict[str, dict] = {}
    for topic_results in state["search_results"]:
        for paper in topic_results:
            url = paper.get("url") or paper.get("link") or paper.get("openalex_url") or ""
            if url and url not in unique:
                unique[url] = paper.copy()
    papers = list(unique.values())
    add_log(f"Collected {len(papers)} unique papers from topic searches.")
    return {"papers": papers}


async def process_abstract(state: State) -> dict:
    final_results = await semantic_abstract_rank(
        state["papers"], state["summary"], FINAL_TOP_K
    )
    add_log(f"Final ranking complete: {len(final_results)} papers.")
    return {"final_results": final_results}


builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("process_title", process_title)
builder.add_node("collect_unique_papers", collect_unique_papers)
builder.add_node("process_abstract", process_abstract)
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", dispatch_search, ["process_title"])
builder.add_edge("process_title", "collect_unique_papers")
builder.add_edge("collect_unique_papers", "process_abstract")
builder.add_edge("process_abstract", END)
graph = builder.compile()


# -------------------- Robust PDF fetching (proxy + cache) --------------------

MAX_PDF_BYTES = 50 * 1024 * 1024
PDF_CACHE_LIMIT = 8
pdf_cache: dict[str, bytes] = {}
pdf_lock = threading.Lock()  # protects only the cache dict, never held during downloads
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
# Set UNPAYWALL_EMAIL in your environment. Unpaywall is skipped if it is empty.
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "")


def _clean_pdf_url(url: str) -> str:
    match = re.match(r"^\[[^\]]+\]\((https?://[^)]+)\)$", url or "")
    return match.group(1) if match else (url or "")


def _download_pdf(url: str) -> bytes:
    """Download a PDF with browser-like headers, retries and validation."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": BROWSER_UA,
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    last_error = "unknown error"
    for attempt in range(3):
        try:
            response = session.get(url, timeout=(10, 60), stream=True, allow_redirects=True)
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code}"
                response.close()
                time.sleep(2 * (attempt + 1))
                continue
            if response.status_code in (401, 403):
                response.close()
                raise ValueError(
                    f"The publisher blocked automated access (HTTP {response.status_code}). "
                    "Use 'Open PDF' to view it in your browser."
                )
            response.raise_for_status()

            data = bytearray()
            for part in response.iter_content(chunk_size=65536):
                data.extend(part)
                if len(data) > MAX_PDF_BYTES:
                    response.close()
                    raise ValueError("PDF is larger than the 50 MB limit.")
            final_url = response.url
            response.close()

            if b"%PDF-" in data[:1024]:
                return bytes(data)

            # Got an HTML landing page: look for the real PDF link in its meta tags.
            html = bytes(data[:300000]).decode("utf-8", errors="ignore")
            meta = re.search(
                r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)',
                html,
                re.I,
            )
            if meta:
                new_url = requests.compat.urljoin(final_url, meta.group(1))
                if new_url != url:
                    url = new_url
                    last_error = "landing page, following citation_pdf_url"
                    continue
            raise ValueError("The link did not return a PDF (it returned a web page).")
        except requests.RequestException as exc:
            last_error = str(exc)
            time.sleep(1.5 * (attempt + 1))
    raise ValueError(f"Could not download the PDF: {last_error}")


def _norm_doi(doi: str) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", (doi or "").strip(), flags=re.I)


def _api_json(url: str, **params):
    try:
        r = requests.get(url, params=params or None, timeout=20, headers={"User-Agent": BROWSER_UA})
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


def fallback_pdf_urls(paper: dict) -> list[str]:
    """Other legal open-access copies of the same paper (arXiv, Unpaywall, Semantic Scholar, OpenAlex)."""
    urls: list[str] = []
    doi = _norm_doi(paper.get("doi", ""))
    blob = " ".join([paper.get("pdf_url", ""), paper.get("url", ""), paper.get("link", ""), doi])
    arx = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(?:\.pdf|v\d+)?(?:$|\s)", blob, re.I) or re.search(
        r"10\.48550/arxiv\.([\w.\-]+)", blob, re.I
    )
    if arx:
        urls.append(f"https://arxiv.org/pdf/{arx.group(1)}")
    if doi:
        if UNPAYWALL_EMAIL:
            data = _api_json(f"https://api.unpaywall.org/v2/{doi}", email=UNPAYWALL_EMAIL)
            if data:
                locs = [data.get("best_oa_location")] + (data.get("oa_locations") or [])
                urls += [loc.get("url_for_pdf") for loc in locs if loc and loc.get("url_for_pdf")]
        data = _api_json(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}", fields="openAccessPdf")
        if data and (data.get("openAccessPdf") or {}).get("url"):
            urls.append(data["openAccessPdf"]["url"])
        data = _api_json(f"https://api.openalex.org/works/https://doi.org/{doi}")
        if data:
            for loc in data.get("locations") or []:
                if loc.get("pdf_url"):
                    urls.append(loc["pdf_url"])
    seen, unique = set(), []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def get_pdf_bytes(paper: dict) -> bytes:
    """Return cached PDF bytes. Tries the main link, then alternative open-access copies."""
    primary = _clean_pdf_url(paper.get("pdf_url", ""))
    cache_key = primary or paper.get("openalex_url") or paper.get("title", "")

    with pdf_lock:
        cached = pdf_cache.get(cache_key)
    if cached is not None:
        return cached

    errors: list[str] = []
    tried: set[str] = set()

    def attempt(u: str) -> bytes | None:
        if not u or u in tried:
            return None
        tried.add(u)
        try:
            return _download_pdf(u)
        except Exception as exc:
            host = re.sub(r"^https?://([^/]+).*", r"\1", u)
            errors.append(f"{host}: {exc}")
            add_log(f"PDF source failed ({host}): {exc}", "WARNING")
            return None

    data = attempt(primary)
    if data is None:
        for alt in fallback_pdf_urls(paper):
            data = attempt(alt)
            if data is not None:
                add_log(f"PDF fetched from alternative source: {alt}")
                break
    if data is None:
        raise ValueError("All PDF sources failed. " + " | ".join(errors[:4]))

    with pdf_lock:
        if len(pdf_cache) >= PDF_CACHE_LIMIT:
            pdf_cache.pop(next(iter(pdf_cache)))
        pdf_cache[cache_key] = data
    return data


def find_paper(rank: int) -> dict:
    if result is None:
        raise HTTPException(status_code=400, detail="Run a research search first.")
    paper = next((p for p in result["final_results"] if p.get("Rank") == rank), None)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper rank not found.")
    return paper


# -------------------- PDF processing and paper Q&A --------------------

def extract_pdf_chunks(pdf_bytes: bytes) -> tuple[int, list[Document]]:
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_docs = []
        for i, page in enumerate(pdf):
            text = page.get_text()
            if text.strip():
                page_docs.append(Document(page_content=text, metadata={"page": i + 1}))
    finally:
        pdf.close()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return len(page_docs), splitter.split_documents(page_docs)


def retrieve_text(question: str) -> str:
    if retriever is None:
        return ""
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are ResearchFinderAI, an academic question-answering assistant
specializing in scientific papers.

Your main goal is to provide a useful answer whenever the provided
paper context has even a small amount of relevant information.

STRICT RULES:

1. Use ONLY the provided paper context as the source of factual
   information. Never add facts from your own knowledge.

2. Be highly inclusive when answering. If even a small part of the
   context is related to the question, use it to answer.

3. Do not require a direct, complete, or exact answer in the context.
   Related explanations, partial evidence, examples, definitions,
   methods, findings, and indirect statements are all useful.

4. If the context only partially relates to the question, provide
   whatever relevant information is available. Do not reject the
   question merely because the answer is incomplete.

5. Carefully organize, summarize, combine, and rephrase the relevant
   information from the context to make the answer clear and useful.

6. You may connect related statements from different passages, but
   do not create new facts, explanations, examples, conclusions,
   methods, or claims that are not supported by the context.

7. Do not fill information gaps using assumptions. If a detail is
   missing, simply state that the provided context does not specify it.

8. If the context provides only weak or indirect relevance, still
   answer using that information and briefly clarify the connection.
   Do not exaggerate its relevance or certainty.

9. Refuse to answer from the context ONLY when it contains absolutely
   no meaningful information relevant to the question. In that case,
   say:
   "The provided context does not contain relevant information
   to answer this question."

10. Never confuse "the answer is not explicitly stated" with
    "the context contains no relevant information."

11. Answer in the same language as the user's question.

12. Start with the relevant answer. Keep it clear, direct, and
    appropriately detailed.

13. Do not claim that the paper proves something when it merely
    discusses, suggests, or partially supports it.

14. Your job is to ORGANIZE the available information, not to
    CREATE missing information.

Remember:
A partially relevant answer is better than an unnecessary refusal.
Use every relevant piece of evidence available, but never invent
anything to make the answer appear complete.
            """,
        ),
        (
            "human",
            """
Paper context:
{context}

Question:
{question}

Instructions:
First identify the relevant evidence in the context.
Then answer the question using that evidence.
If multiple passages are relevant, synthesize them into one coherent answer.
Do not confuse information missing from the retrieved context with
information missing from the full paper.

Answer:
            """,
        ),
    ]
)


def answer_question(question: str, question_llm) -> str:
    context = retrieve_text(question)
    if not context.strip():
        return NO_CONTEXT_MESSAGE
    chain = qa_prompt | question_llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# -------------------- API endpoints --------------------

@app.get("/health")
async def health_check():
    return {
        "status": "error" if model_error else ("ok" if models_ready.is_set() else "loading"),
        "models_ready": models_ready.is_set(),
        "model_error": model_error,
        "message": "Research Finder API is running.",
    }


@app.get("/logs")
async def get_logs():
    with log_lock:
        return {"logs": list(logs)}


@app.post("/research")
async def run_research(request: ResearchRequest):
    global result, selected_paper, chunks, vector_store, retriever
    require_models()
    add_log(f"Research started: {request.idea}")
    try:
        new_result = await graph.ainvoke(
            {
                "idea": request.idea,
                "api_key": request.api_key,
                "summary": "",
                "titles": [],
                "search_results": [],
                "papers": [],
                "final_results": [],
            }
        )
        result = new_result
        selected_paper = None
        chunks = []
        vector_store = None
        retriever = None
        add_log("Research completed successfully.")
        return {
            "summary": result["summary"],
            "titles": result["titles"],
            "papers": result["final_results"],
        }
    except Exception as exc:
        add_log(f"Research failed: {exc}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Research failed: {exc}") from exc


@app.post("/paper")
async def get_paper(request: RankRequest):
    paper = find_paper(request.rank)
    return {
        "Rank": paper["Rank"],
        "title": paper.get("title", ""),
        "Paper_URL": paper.get("link", ""),
        "PDF_URL": paper.get("pdf_url", ""),
    }


@app.get("/paper/pdf")
async def paper_pdf(rank: int):
    """Serve the paper PDF from our own origin so the iframe is never blocked."""
    paper = find_paper(rank)
    if not paper.get("pdf_url", ""):
        raise HTTPException(status_code=400, detail="PDF URL is not available.")
    try:
        pdf_bytes = await asyncio.to_thread(get_pdf_bytes, paper)
    except Exception as exc:
        add_log(f"PDF proxy failed for rank {rank}: {exc}", "ERROR")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'inline; filename="paper.pdf"', "Cache-Control": "private, max-age=3600"},
    )


@app.post("/paper/process")
async def process_selected_paper(request: RankRequest):
    global selected_paper, chunks, vector_store, retriever
    require_models()
    paper = find_paper(request.rank)

    if not paper.get("pdf_url", ""):
        raise HTTPException(status_code=400, detail="PDF URL is not available.")

    add_log(f"Downloading PDF for rank {request.rank}: {paper.get('title', '')}")
    try:
        pdf_bytes = await asyncio.to_thread(get_pdf_bytes, paper)
        total_pages, new_chunks = await asyncio.to_thread(extract_pdf_chunks, pdf_bytes)
        if not new_chunks:
            raise ValueError("No selectable text found in the PDF. It may be a scanned document.")

        new_vector_store = await asyncio.to_thread(FAISS.from_documents, new_chunks, embeddings)
        chunks = new_chunks
        vector_store = new_vector_store
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 24},
        )
        selected_paper = paper
        add_log(f"PDF processed: {total_pages} pages, {len(chunks)} chunks.")
        return {
            "title": paper.get("title", ""),
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "message": "Paper processed successfully.",
        }
    except Exception as exc:
        add_log(f"PDF processing failed: {exc}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Failed to download or process the PDF: {exc}") from exc


@app.post("/ask")
async def ask_question_endpoint(request: QuestionRequest):
    if not chunks or retriever is None:
        raise HTTPException(status_code=400, detail="Process a paper PDF before asking questions.")
    try:
        question_llm = ChatGroq(model=MODEL_NAME, api_key=request.api_key, temperature=0)
        answer = await asyncio.to_thread(answer_question, request.question, question_llm)
        add_log(f"Answered question: {request.question}")
        return {"question": request.question, "answer": answer}
    except Exception as exc:
        add_log(f"Question answering failed: {exc}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Failed to answer the question: {exc}") from exc


# -------------------- Entry point --------------------

if __name__ == "__main__":
    try:
        asyncio.get_running_loop()
        in_notebook = True  # Colab / Jupyter already run an event loop.
    except RuntimeError:
        in_notebook = False

    if in_notebook:
        server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info"))
        threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True).start()
        print("FastAPI server is starting on port 8000...")
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
