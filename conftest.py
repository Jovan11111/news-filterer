import sys
from pathlib import Path

# Ensure the project root is importable so tests can use import src....
sys.path.insert(0, str(Path(__file__).resolve().parent))
