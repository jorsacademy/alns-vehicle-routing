from pprint import pprint

from alns_vrp.experiment import run_demo


if __name__ == "__main__":
    pprint(run_demo(iterations=3000, seed=2026, customers=36))
