import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "miso",
    folder="rosenbrock_1000",
    hp_opt="overlap",
    sample_domain=1000
)