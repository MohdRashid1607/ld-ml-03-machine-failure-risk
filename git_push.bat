@echo off
REM ============================================================
REM  LD ML 03 — Push project to GitHub
REM  Run this from the project root:
REM    H:
REM    cd "internship projects\ld-ml-03-machine-failure-risk"
REM    git_push.bat
REM ============================================================

echo.
echo === LD ML 03 GitHub Push Script ===
echo.

REM -- Check if remote is set --
git remote -v
echo.

REM -- If no remote shown above, set it (uncomment and edit the line below):
REM git remote add origin https://github.com/YOUR_USERNAME/ld-ml-03-machine-failure-risk.git

REM -- Stage all files EXCEPT what .gitignore excludes --
git add .

REM -- Show what will be committed --
echo Files staged for commit:
git status --short
echo.

REM -- Create the commit --
git commit -m "v0.1.0 - Complete LD ML 03 submission: Isolation Forest anomaly detector, 6-tab Gradio Space, full docs"

REM -- Push to GitHub main branch --
git push -u origin main

echo.
echo === Push complete! ===
echo GitHub repo: Check your github.com profile
echo HF Space   : https://huggingface.co/spaces/khalidml65/lottery-machine-failure-risk
echo HF Dataset : https://huggingface.co/datasets/khalidml65/lottery-draw-machine-telemetry
echo HF Model   : https://huggingface.co/khalidml65/lottery-isolation-forest-detector
echo.
pause
