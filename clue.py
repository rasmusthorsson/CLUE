"""
    CLUE Main launcher script
"""

from pathlib import Path
import sys

src = Path(__file__).parent / "CLUE-CORE" / "src"
sys.path.insert(0, str(src))

from clueGUI import ClueGui

def main():
    app = ClueGui()
    app.run()

if __name__ == "__main__":
    main()