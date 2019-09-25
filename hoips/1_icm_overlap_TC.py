import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "icm",
    1,
    folder="N3R2_N1R2_TC",
    exact_cost=True,
    hp_opt="overlap",
    generate_new_historical=False,
    N_historical=10
)