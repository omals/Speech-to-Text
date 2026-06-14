@echo off
:: Run ambient_notes.py inside the virtual environment
cd /d "%~dp0"
call canary_env\Scripts\activate.bat
python ambient_notes.py
pause
