import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "icm",
    folder="CO",
    hp_opt="overlap",
    sample_domain=500
)