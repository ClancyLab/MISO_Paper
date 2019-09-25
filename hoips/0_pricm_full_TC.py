import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "pricm",
    0,
    folder="N1R3_N1R2_TC",
    exact_cost=True,
    hp_opt="full",
    generate_new_historical=False,
    N_historical=10
)