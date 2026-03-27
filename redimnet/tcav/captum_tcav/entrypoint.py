from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from captum_tcav.vox2.entrypoint import *  # noqa: F403
else:
    from .vox2.entrypoint import *  # noqa: F403
