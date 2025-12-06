# File: concept/generate_concepts.py
import sys
from pathlib import Path

# Ensure this folder is importable
sys.path.append(str(Path(__file__).parent))

from concepts_creation import CREATE_ALL_CONCEPT_DIRS

if __name__ == "__main__":
    out_dir = Path("positive concepts dataset")
    out_dir.mkdir(exist_ok=True)

    print("Generating concepts into:", out_dir.resolve())
    CREATE_ALL_CONCEPT_DIRS()
    print("Done! All concepts saved.")
