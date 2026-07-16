#!/usr/bin/env python3
"""Runner for the self-improving-harness demos.

Usage:
    python3 demo/run.py           # interactive menu
    python3 demo/run.py 1         # run demo 1
    python3 demo/run.py all       # run every demo back to back
    python3 demo/run.py all --fast  # no dramatic pauses (CI / smoke test)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

MENU = """
  Self-Improving Agent Harnesses — live demos
  (concepts from Ben Dickson, "A Primer on Self-Improving Agent Harnesses", TechTalks, Jul 2026)

    1. Composable harness (HarnessX)   hot-swap one lifecycle hook, prove the rest is untouched
    2. AEGIS evolution engine          Digester -> Planner -> Evolver -> Critic; a static scan catches a reward hack
    3. Harness-model co-evolution      two policies, one shared replay buffer, GRPO-style group-relative updates

    a. run all      q. quit
"""


def run(number: str) -> None:
    import demo1_composable_harness
    import demo2_aegis_pipeline
    import demo3_coevolution_grpo

    demos = {
        "1": demo1_composable_harness.main,
        "2": demo2_aegis_pipeline.main,
        "3": demo3_coevolution_grpo.main,
    }
    if number in ("a", "all"):
        for main in demos.values():
            main()
    elif number in demos:
        demos[number]()
    else:
        print(f"unknown demo: {number}")
        sys.exit(2)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--fast"]
    if "--fast" in sys.argv:
        os.environ["DEMO_FAST"] = "1"

    if args:
        run(args[0])
        return

    while True:
        print(MENU)
        choice = input("  choose> ").strip().lower()
        if choice in ("q", "quit", "exit", ""):
            return
        run(choice)


if __name__ == "__main__":
    main()
