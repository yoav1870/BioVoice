from __future__ import annotations

from tcav_core.config import CONFIG
from tcav_core.runner import TCAVRunner


def main() -> None:
    runner = TCAVRunner(CONFIG)
    runner.run()


if __name__ == "__main__":
    main()
