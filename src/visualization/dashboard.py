import subprocess
import sys
from pathlib import Path

def main():
    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])

if __name__ == "__main__":
    main()
