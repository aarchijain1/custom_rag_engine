@echo off
REM Local RAG Agent - Quick Setup Script for Windows

echo ======================================================================
echo LOCAL RAG AGENT WITH LANGGRAPH - SETUP (Windows)
echo ======================================================================
echo.

REM Check Python
echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)
echo OK: Python is installed
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo WARNING: Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    echo OK: Virtual environment created
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo OK: Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
echo OK: pip upgraded
echo.

REM Install dependencies
echo Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo OK: Dependencies installed
echo.

REM Create .env file
if not exist .env (
    echo Creating .env file...
    copy .env.example .env
    echo OK: .env file created
    echo.
    echo WARNING: Please edit .env and add your Gemini API key!
    echo Get your API key from: https://makersuite.google.com/app/apikey
    echo.
) else (
    echo OK: .env file already exists
    echo.
)

REM Create directories
echo Creating necessary directories...
if not exist documents mkdir documents
if not exist chroma_db mkdir chroma_db
echo OK: Directories created
echo.

REM Summary
echo ======================================================================
echo SETUP COMPLETE!
echo ======================================================================
echo.
echo Next steps:
echo 1. Edit .env and add your GOOGLE_API_KEY
echo 2. Run: python main.py
echo 3. Use option 6 to create sample documents
echo 4. Use option 2 to index the documents
echo 5. Use option 1 to start chatting!
echo.
echo For help, see README.md
echo.
echo To activate the virtual environment in the future:
echo   venv\Scripts\activate.bat
echo.
echo ======================================================================
pause