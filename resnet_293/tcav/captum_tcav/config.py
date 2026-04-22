from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from captum_tcav.asvspoof5.config import *  # noqa: F403
else:
    from .asvspoof5.config import *  # noqa: F403
