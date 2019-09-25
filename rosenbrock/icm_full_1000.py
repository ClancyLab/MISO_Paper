import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "icm",
    folder="rosenbrock_1000",
    hp_opt="full",
    sample_domain=1000
)