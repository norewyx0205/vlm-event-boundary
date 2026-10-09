"""Run the fixed Phase 3C primary pilot after baseline and technical preflight."""

try:
    from .run_phase3c_preflight import main
except ImportError:
    from run_phase3c_preflight import main


if __name__ == "__main__":
    main(stages=("patch", "routing", "knockout"), description=__doc__)
