@echo off
title Vivarium Launcher

echo ============================================================
echo  Starting llama.cpp server...
echo ============================================================

start "llama-server" "C:\AI\llama.cpp\llama-server.exe" ^
  -m "C:\Users\ethan\.lmstudio\models\lmstudio-community\Qwen3-30B-A3B-Instruct-2507-GGUF\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf" ^
  --ctx-size 32768 ^
  --n-gpu-layers 99 ^
  --port 1234 ^
  --host 0.0.0.0

echo Waiting for server to load model...
timeout /t 30 /nobreak

echo ============================================================
echo  Starting Vivarium...
echo ============================================================

cd /d "F:\Books\scriptorium-vivarium"
call .\.venv\Scripts\activate.bat
python .\tools\rome_experience_cli_final.py ^
  --db "F:\Books\as_project\db\vivarium_rome_core.sqlite" ^
  --base-url "http://localhost:1234/v1" ^
  --model "Qwen3-30B-A3B-Instruct-2507-Q4_K_M" ^
  --max-tokens 16000 ^
  --debug

pause
