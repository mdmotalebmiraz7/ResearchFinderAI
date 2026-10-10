# ResearchFinderAI

An AI-assisted research paper discovery and analysis project with a web-based frontend and Python backend.

## Features

- Research paper discovery and analysis
- React + Vite frontend
- Python backend
- Groq API integration, if configured
- Windows startup using `run.bat`

## 1. Requirements

Install the following software:

| Software | Download |
|---|---|
| Python | https://www.python.org/downloads/ |
| Node.js (LTS) | https://nodejs.org/en/download |
| Git | https://git-scm.com/downloads |
| Groq API key | https://console.groq.com/keys |

During Python installation on Windows, enable **Add Python to PATH**.

Verify the installations in PowerShell:

```powershell
python --version
node --version
npm --version
git --version
```

If `python` is not recognized, try `py --version`.

## 2. Configure the Groq API Key

If your research workflow uses Groq:

1. Visit https://console.groq.com/
2. Sign in or create an account.
3. Open https://console.groq.com/keys
4. Create an API key and store it securely.
5. Configure the key using the method expected by the backend code.

**Security:** Never commit API keys, passwords, or secret `.env` files to GitHub. Do not put secret API keys in frontend code.

## 3. Download the Repository

Open PowerShell in the directory where you want to store the project:

```powershell
git clone https://github.com/mdmotalebmiraz7/ResearchFinderAI.git
cd ResearchFinderAI
```

## 4. Install Backend Dependencies

From the project root directory, run:

```powershell
python -m venv backend\venv
backend\venv\Scripts\python.exe -m pip install --upgrade pip
backend\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

These commands assume the repository contains `backend\requirements.txt`.

## 5. Install Frontend Dependencies

From the project root directory:

```powershell
cd frontend
npm install
cd ..
```

This installs the packages defined in `frontend\package.json`.

## 6. Run the Application

From the project root directory, run:

```powershell
.\run.bat
```

The startup script is designed to check dependencies, install missing packages, start the backend and frontend in separate terminal windows, and open the frontend in your browser.

Typical local addresses:

- Frontend: http://localhost:5173
- Backend: http://localhost:8000

The actual addresses depend on the project configuration. Keep both service windows open while using the application.

## 7. Run the Services Manually

If `run.bat` does not work, start each service separately.

### Backend

Open a PowerShell window in the project root:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -m uvicorn research_finder:app --host 127.0.0.1 --port 8000
```

This assumes `backend\research_finder.py` contains a FastAPI application named `app`. If the module or application object has a different name, update the command accordingly.

### Frontend

Open a second PowerShell window in the project root:

```powershell
cd frontend
npm run dev
```

Open the URL printed by Vite in your browser.

## 8. Update the Project

To download the latest committed changes:

```powershell
git pull origin main
```

To upload your changes:

```powershell
git status
git add README.md run.bat
git commit -m "Update project documentation and startup script"
git push origin main
```

If you changed other project files, add those files separately as needed.

Before pushing, check that API keys, passwords, `.env` files, virtual environments, and `node_modules` are not being uploaded.

## Troubleshooting

### Python is not recognized

Install Python from https://www.python.org/downloads/, enable **Add Python to PATH**, and reopen PowerShell.

### Node.js or npm is not recognized

Install Node.js LTS from https://nodejs.org/en/download and reopen PowerShell.

### Vite is not recognized

Run these commands from the frontend directory:

```powershell
npm install
npm run dev
```

### Backend dependency installation fails

Verify that `backend\requirements.txt` exists and inspect the full error message.

### Backend fails to start

Check the backend terminal. Confirm that the FastAPI module name and application object match the `uvicorn` command.

### Groq API errors

Verify that the API key is configured correctly and that the selected model is available to your account.

## License

No license has been specified for this repository. Add a license file if you want to grant others explicit permission to reuse or modify the project.
