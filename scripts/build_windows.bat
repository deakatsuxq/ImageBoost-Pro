@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File "%~dp0build_windows.ps1" %*
exit /b %ERRORLEVEL%
