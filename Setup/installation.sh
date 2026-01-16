#!/bin/bash

set -e

read -p "Enter your API key: " API_KEY

echo "[*] Creating virtual environment: apex_env"
python3 -m venv apex_env

echo "[*] Activating virtual environment..."
source apex_env/bin/activate

echo "[*] Downloading wheel file..."
curl -o apex_arena-0.1.0-py3-none-any.whl -X GET \
  "https://proxy-319533213591.us-west2.run.app/download_apex_wheel" \
  -H "Authorization: Bearer $API_KEY"

echo "[*] Installing package..."
pip install ./apex_arena-0.1.0-py3-none-any.whl

echo "[✓] Installation complete."
echo "[✓] To activate your environment again later, run: source apex_env/bin/activate"