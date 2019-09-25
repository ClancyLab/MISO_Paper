import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "pricm",
    1,
    folder="N3R2_N1R2_TC",
    exact_cost=True,
    hp_opt="overlap",
    generate_new_historical=True,
    N_historical=10
)