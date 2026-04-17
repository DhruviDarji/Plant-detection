#!/bin/bash
# ─────────────────────────────────────────────────
#  Verdana Backend — Setup & Run
#  Usage:  bash run.sh
# ─────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

echo "🌿 Verdana Backend Setup"
echo "────────────────────────"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "→ Creating virtual environment..."
  python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "→ Installing dependencies..."
pip install -r requirements.txt -q

# Copy HTML files next to app.py so Flask can serve them
# Adjust these paths if your HTML files are elsewhere
if [ -f "../purchase.html" ]; then
  cp ../purchase.html ./purchase.html
fi
if [ -f "../prototype.html" ]; then
  cp ../prototype.html ./prototype.html
fi

echo "→ Starting server on http://localhost:5000"
echo ""
python app.py
