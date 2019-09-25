import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "miso",
    folder="rosenbrock_500",
    hp_opt="is0",
    sample_domain=500
)