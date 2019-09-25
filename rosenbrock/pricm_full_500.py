import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "pricm",
    folder="rosenbrock_500",
    hp_opt="full",
    sample_domain=500
)