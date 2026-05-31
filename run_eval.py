"""Compatibility wrapper for the canonical evaluation script.

The maintained implementation lives in scripts/run_eval.py. This file remains
so older commands such as `python run_eval.py ...` continue to work.
"""

from scripts.run_eval import main


if __name__ == "__main__":
    main()
