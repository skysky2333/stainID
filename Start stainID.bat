@echo off
rem Double-click to start stainID. It opens in your web browser; keep this window open while you work.
cd /d "%~dp0"
.venv\Scripts\stainid serve
