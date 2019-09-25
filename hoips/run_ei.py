from pal.opt import Optimizer
from pal.constants.solvents import solvents
from pal.kernels.matern import maternKernel52 as mk52
from pal.stats.MLE import MLE
from pal.stats.MAP import MAP

import os
import copy
import time
import numpy as np
import cPickle as pickle


def run(run_index, folder="data_dumps", infosources=0, exact_cost=True):

    # Store data for debugging
    IS_N5R2 = pickle.load(open("enthalpy_N5_R2_wo_GBL_Ukcal-mol", 'r'))
    IS_N3R2 = pickle.load(open("enthalpy_N3_R2_Ukcal-mol", 'r'))
    IS_N1R2 = pickle.load(open("enthalpy_N1_R2_Ukcal-mol", 'r'))
    IS_N1R3 = pickle.load(open("enthalpy_N1_R3_Ukcal-mol", 'r'))

    if infosources == 0:
        IS0 = IS_N1R3
        if exact_cost:
            costs = [6.0]
        else:
            costs = [10.0]
    elif infosources == 1:
        IS0 = IS_N3R2
        if exact_cost:
            costs = [14.0]
        else:
            costs = [10.0]
    elif infosources == 2:
        IS0 = IS_N5R2
        if exact_cost:
            costs = [27.0]
        else:
            costs = [100.0]
    elif infosources == 3:
        IS0 = IS_N5R2
        if exact_cost:
            costs = [27.0]
        else:
            costs = [100.0]
    else:
        raise Exception("HOW?")

    # Generate the main object
    sim = Optimizer()

    # Assign simulation properties
    sim.hyperparameter_objective = MLE
    ###################################################################################################
    # File names
    sim.fname_out = "enthalpy_ei.dat"
    sim.fname_historical = "%s/%d_reduced.history" % (folder, run_index)

    print "Waiting on %s to be written..." % sim.fname_historical,
    while not all([os.path.exists(sim.fname_historical), os.path.exists("%s/%d.combos" % (folder, run_index))]):
        time.sleep(30)
    print " DONE"

    # Information sources, in order from expensive to cheap
    sim.IS = [
        lambda h, c, s: -1.0 * IS0[' '.join([''.join(h), c, s])],
    ]
    sim.costs = costs
    sim.save_extra_files = False

    sim.logger_fname = "%s/%d_ei.log" % (folder, run_index)
    if os.path.exists(sim.logger_fname):
        os.system("rm %s" % sim.logger_fname)
    os.system("touch %s" % sim.logger_fname)

    sim.historical_nsample = len(pickle.load(open("%s/%d_reduced.history" % (folder, run_index), 'r')))
    sim.combinations = [c for c in pickle.load(open("%s/%d.combos" % (folder, run_index), 'r')) if c.endswith("0")]
    ########################################

    sim.n_start = 10  # The number of starting MLE samples
    # sim.reopt = 20
    sim.reopt = float("inf")  # Don't reopt hyperparams
    sim.ramp_opt = None
    sim.parallel = False

    # Possible compositions by default
    sim.A = ["Cs", "MA", "FA"]
    sim.B = ["Pb"]
    sim.X = ["Cl", "Br", "I"]
    sim.solvents = copy.deepcopy(solvents)
    sim.S = list(set([v["name"] for k, v in sim.solvents.items()]))
    sim.mixed_halides = True
    sim.mixed_solvents = False

    # Parameters for debugging and overwritting
    sim.debug = False
    sim.verbose = True
    sim.overwrite = True  # If True, warning, else Error

    # Functional forms of our mean and covariance
    # MEAN: 4 * mu_alpha + mu_zeta
    # COV: sig_alpha * |X><X| + sig_beta * I_N + sig_zeta + MaternKernel(S, weights, sig_m)

    SCALE = [2.0, 4.0][int(sim.mixed_halides)]
    # _1, _2, _3 used as dummy entries
    sim.mean = lambda _1, Y, theta: np.array([SCALE * theta.mu_alpha + theta.mu_zeta for _ in Y])

    def cov(X, Y, theta):
        A = theta.sig_alpha * np.dot(np.array(X)[:, 1:-3], np.array(X)[:, 1:-3].T)
        B = theta.sig_beta * np.diag(np.ones(len(X)))
        C = theta.sig_zeta
        D = mk52(np.array(X)[:, -3:-1], [theta.l1, theta.l2], theta.sig_m)

        return A + B + C + D

    sim.cov = cov

    sim.theta.bounds = {}
    sim.theta.mu_alpha, sim.theta.bounds['mu_alpha'] = None, (1E-3, lambda _, Y: max(Y))
    sim.theta.sig_alpha, sim.theta.bounds['sig_alpha'] = None, (1E-2, lambda _, Y: 10.0 * np.var(Y))
    sim.theta.sig_beta, sim.theta.bounds['sig_beta'] = None, (1E-2, lambda _, Y: 10.0 * np.var(Y))
    sim.theta.mu_zeta, sim.theta.bounds['mu_zeta'] = None, (1E-3, lambda _, Y: max(Y))
    sim.theta.sig_zeta, sim.theta.bounds['sig_zeta'] = None, (1E-2, lambda _, Y: 10.0 * np.var(Y))
    sim.theta.sig_m, sim.theta.bounds['sig_m'] = None, (1E-2, lambda _, Y: np.var(Y))
    sim.theta.l1, sim.theta.bounds['l1'] = None, (1E-1, 1)
    sim.theta.l2, sim.theta.bounds['l2'] = None, (1E-1, 1)

    # NOTE! This is a reserved keyword in misoKG.  We will generate a list of the same length
    # of the information sources, and use this for scaling our IS.
    # sim.theta.rho, sim.theta.bounds['rho'] = {"[0, 0]": 1}, (1E-1, 5.0)
    # NOTE! This is a reserved keyword in misoKG.  We will generate a list of the same length
    # of the information sources, and use this for scaling our IS.
    sim.theta.rho = {"[0, 0]": 1}
    sim.theta.bounds['rho [0, 0]'] = (1, 1)

    sim.theta.set_hp_names()

    h, c, s = min([(IS0[k], k) for k in IS0.keys()])[1].split()
    sim.recommendation_kill_switch = "%sPb%s_%s_0" % (c, h, s)
    sim.primary_rho_opt = False
    sim.update_hp_only_with_IS0 = False

    ###################################################################################################

    # Start simulation
    sim.run()


if __name__ == "__main__":
    run(6660, folder="data_dumps", infosources=0, exact_cost=True)

