@echo off
cd /d "E:\claude code\hexapod-grid-sim"
venv\Scripts\python.exe -m pytest tests/ -v
pause
