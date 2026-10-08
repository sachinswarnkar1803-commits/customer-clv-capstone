"""Launch the Streamlit dashboard from any working directory."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
app = ROOT / "dashboard" / "app.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], cwd=ROOT, check=True)
