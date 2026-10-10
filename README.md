# ResearchFinderAI 🔎

**Discover Research Papers. Explore Ideas. Find Your Next Research Direction.**

ResearchFinderAI is an AI-assisted research discovery tool designed to help students, researchers, and academics explore research topics and discover relevant academic papers through a simple web interface.

Instead of manually searching through hundreds of papers, ResearchFinderAI aims to make the research discovery process more organized and accessible.

## ✨ What Can You Do With ResearchFinderAI?

- **Explore Research Topics:** Start exploring papers related to your research interests.
- **Discover Relevant Papers:** Find potentially useful research papers for your chosen topic.
- **AI-Assisted Analysis:** Use AI-powered analysis features available in the application to help review research information.
- **Simple Web Interface:** Interact with the application through its frontend.
- **Easy Local Setup:** Run the project on your own computer using the provided startup script.

*Available features depend on the current implementation and configuration of the project.*

## 👥 Who Is It For?

ResearchFinderAI is designed for:

- University students looking for research ideas.
- Undergraduate and postgraduate researchers exploring a new field.
- Academics who want to investigate relevant literature.
- Anyone beginning a research project and looking for related papers.

## 🚀 Getting Started

### 1. Download the Project

First, install Git if you do not already have it.

Download Git: https://git-scm.com/downloads

Open a terminal and run:

```bash
git clone https://github.com/mdmotalebmiraz7/ResearchFinderAI.git
cd ResearchFinderAI
```

Alternatively, open the repository on GitHub and select **Code → Download ZIP**.

### 2. Install the Required Software

Install the following tools:

| Software | Purpose | Download |
|---|---|---|
| Python | Runs the backend | https://www.python.org/downloads/ |
| Node.js (LTS) | Runs the frontend tools | https://nodejs.org/en/download |
| Git | Downloads and updates the project | https://git-scm.com/downloads |

**Windows users:** During Python installation, enable **Add Python to PATH**.

### 3. Configure the AI Service

If your version of ResearchFinderAI uses Groq, you may need a Groq API key.

1. Create an account or sign in at https://console.groq.com/
2. Visit https://console.groq.com/keys
3. Create an API key.
4. Configure it according to the application's instructions.

Keep your API key private. Never share it publicly or upload it to GitHub.

### 4. Start ResearchFinderAI

Open the project folder and double-click:

`run.bat`

The startup script is intended to prepare the required dependencies, launch the backend and frontend, and open the application in your default browser.

If the browser does not open automatically, try:

http://localhost:5173

Keep the backend and frontend terminal windows open while using the application.

## 💻 System Requirements

You will need:

- A computer running Windows, or a compatible environment with the required tools installed.
- An internet connection for initial setup, downloading dependencies, and any online research or AI services.
- A valid API key if your selected AI features require one.

Actual performance depends on your computer, network connection, and the services used by the application.

## ❓ Troubleshooting

**The application does not start**

Make sure Python and Node.js are installed correctly, then try running `run.bat` again. Check the terminal windows for error messages.

**The frontend does not open**

Try opening http://localhost:5173 manually and confirm that the frontend terminal reports a successful startup.

**AI features are not working**

Check your API key configuration, internet connection, and the availability of the configured AI model.

**Installation fails**

Check the error message and confirm that the project's dependencies have been installed successfully.

## 🔒 Privacy and API Keys

Keep all secret credentials private. Do not commit API keys, passwords, or private environment files to a public repository.

When using external AI services, review the relevant provider's privacy policy and terms before submitting sensitive research material.

## 🛠️ Project Status

ResearchFinderAI is an evolving project. Features, supported workflows, and setup instructions may change as development continues.

Contributions, suggestions, and bug reports are welcome.

## 📜 License

No license has been specified yet. Reuse and redistribution permissions should be clarified by adding an appropriate license.

---

**ResearchFinderAI — Making research discovery more accessible.**
