import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "pricm",
    2,
    folder="N5R2_N3R2_TC",
    exact_cost=True,
    hp_opt="overlap",
    generate_new_historical=True,
    N_historical=10
)