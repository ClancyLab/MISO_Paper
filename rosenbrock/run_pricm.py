from pal.opt import Optimizer
import pal.utils.strings as pal_strings
from pal.constants.solvents import solvents
from pal.kernels.matern import maternKernel52 as mk52
from pal.kernels.squared import squared
from pal.stats.likelihood import gaussian as g_loglike
from pal.acquisition.misokg import getNextSample_misokg
from pal.stats.MLE import MLE
from pal.stats.MAP import MAP

import os
import copy
import time
import numpy as np
import scipy.stats
import cPickle as pickle


def isFloat(v):
    try:
        float(v)
    except (ValueError, TypeError):
        return False
    return True


def run(run_index, model,
        folder="data_dumps", hp_opt="IS0", sample_domain=1000):
    '''
    This function will run CO optimization using one of several coregionalization methods.

        1. Pearson R Intrinsic Coregionalization Method (PRICM).  This approach
           will dynamically calculate the Pearson R value for the off-diagonals
           in the ICM.  Diagonals are kept as 1.0.
        2. Intrinsic Coregionalization Method (ICM).  This approach will use a
           lower triangular matrix (L) of hyperparameters to generate the
           coregionalization matrix B = LL^T.

    Further, we can parameterize the hyperparameters in many ways:

        1. IS0 - Only parameterize hyperparameters using values sampled at IS0.
        2. Full - Parameterize hyperparameters using all sampled data.
        3. Overlap - Parameterize hyperparameters using data that overlaps all IS.

    **Parameters**

        run_index: *int*
            This is simply used for a naming convention.
        model: *str*
            The model to be used (PRICM or ICM).
        folder: *str, optional*
            What to name the folder where the data will go.
        hp_opt: *str, optional*
            With what data should the hyperparameters be parameterized.
            Options: IS0, full, overlap
        sample_domain: *int, optional*
            How many data points to sample from the domain.
    '''

    hp_opt = hp_opt.lower()
    allowed_hp_opt = ["is0", "full", "overlap"]
    assert hp_opt in allowed_hp_opt, "Error, hp_opt (%s) not in %s" % (hp_opt, ", ".join(allowed_hp_opt))

    # Generate the main object
    sim = Optimizer()

    # Assign simulation properties
    sim.hyperparameter_objective = MLE
    ###################################################################################################
    # File names
    sim.fname_out = None
    sim.fname_historical = None

    sim.logger_fname = "%s/%d_%s_%s.log" % (folder, run_index, model, hp_opt)
    if os.path.exists(sim.logger_fname):
        os.system("rm %s" % sim.logger_fname)
    os.system("touch %s" % sim.logger_fname)
    sim.obj_vs_cost_fname = None
    sim.mu_fname = None
    sim.sig_fname = None
    sim.combos_fname = None
    sim.hp_fname = None
    sim.acquisition_fname = None
    sim.save_extra_files = True

    # Information sources, in order from expensive to cheap
    rosenbrock = lambda x1, x2: (1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2 - 456.3
    sim.IS = [
        lambda x1, x2: -1.0 * rosenbrock(x1, x2),
        lambda x1, x2: -1.0 * (rosenbrock(x1, x2) + 0.1 * np.sin(10.0 * x1 + 5.0 * x2))
    ]
    sim.costs = [
        1000.0,
        1.0
    ]

    sim.save_extra_files = False
    ########################################
    sim.numerical = True
    sim.historical_nsample = 5
    sim.domain = [
        (-2.0, 2.0),
        (-2.0, 2.0)
    ]
    sim.sample_n_from_domain = sample_domain
    ########################################

    sim.n_start = 10  # The number of starting MLE samples
    # sim.reopt = 20
    sim.reopt = float('inf')  # Never re-opt hyperparams
    sim.ramp_opt = None
    sim.parallel = False

    # Parameters for debugging and overwritting
    sim.debug = False
    sim.verbose = True
    sim.overwrite = True  # If True, warning, else Error

    sim.acquisition = getNextSample_misokg

    # Functional forms of our mean and covariance
    sim.mean = lambda X, Y, theta: np.array([-456.3 for _ in Y])

    def cov_miso(X0, Y, theta, split=False):
        Kx = squared(np.array(X0), [theta.l1], theta.sig_1)
        Kx_l = squared(np.array(X0), [theta.l2], theta.sig_2)
        return np.block([[Kx, Kx], [Kx, Kx + Kx_l]])

    def cov_pricm(X0, Y, theta, split=False):
        Kx = squared(np.array(X0), [theta.l1], theta.sig_1)
        Kx = Kx + 1E-6 * np.eye(Kx.shape[0])

        if model.lower() == "pricm":
            Ks = np.array([
                np.array([theta.rho[str(sorted([i, j]))] for j in range(theta.n_IS)])
                for i in range(theta.n_IS)
            ])
        elif model.lower() == "icm":
            L = np.array([
                np.array([theta.rho[str(sorted([i, j]))] if i >= j else 0.0 for j in range(theta.n_IS)])
                for i in range(theta.n_IS)
            ])
            # Force it to be positive semi-definite
            Ks = L.dot(L.T)

        if split:
            return Ks, Kx
        else:
            return np.kron(Ks, Kx)

    sim.theta.bounds = {}
    sim.theta.sig_1, sim.theta.bounds['sig_1'] = None, (1E-2, lambda _, Y: np.var(Y))
    sim.theta.l1, sim.theta.bounds['l1'] = None, (1E-1, 1)

    if model == "miso":
        sim.cov = cov_miso
        sim.theta.sig_2, sim.theta.bounds['sig_2'] = None, (1E-2, lambda _, Y: np.var(Y))
        sim.theta.l2, sim.theta.bounds['l2'] = None, (1E-1, 1)
        sim.theta.rho = {str(sorted([i, j])): 1.0 for i in range(len(sim.IS)) for j in range(i, len(sim.IS))}
    else:
        sim.cov = cov_pricm
        sim.theta.rho = {"[0, 0]": None, "[0, 1]": None, "[1, 1]": None}
        if model.lower() == "icm":
            sim.theta.rho = {str(sorted([i, j])): None for i in range(len(sim.IS)) for j in range(i, len(sim.IS))}
        elif model.lower() == "pricm":
            sim.theta.rho = {str(sorted([i, j])): 1.0 for i in range(len(sim.IS)) for j in range(i, len(sim.IS))}
            sim.dynamic_pc = True
        else:
            raise Exception("Invalid model.  Use MISO, ICM, or PRICM")

    for k in sim.theta.rho.keys():
        sim.theta.bounds['rho %s' % k] = (0.1, 1.0)
        a, b = eval(k)
        if a != b:
            sim.theta.bounds['rho %s' % k] = (0.01, 1.0 - 1E-6)

    sim.theta.set_hp_names()

    # Define how we update hyperparameters
    hp_opt = hp_opt.lower()
    if hp_opt == "is0":
        sim.update_hp_only_with_IS0 = True
        sim.update_hp_only_with_overlapped = False
    elif hp_opt == "overlap":
        sim.update_hp_only_with_IS0 = False
        sim.update_hp_only_with_overlapped = True
    elif hp_opt == "full":
        sim.update_hp_only_with_IS0 = False
        sim.update_hp_only_with_overlapped = False
    else:
        raise Exception("Unknown hp_opt (%s)." % hp_opt)
        
    # These should be False by default, but ensure they are
    sim.theta.normalize_L = False
    sim.theta.normalize_Ks = False
    sim.preconditioned = False

    # Assign our likelihood function.
    sim.loglike = g_loglike
    ###################################################################################################

    # Start simulation
    sim.iteration_kill_switch = None
    sim.cost_kill_switch = 100000
    sim.run()


if __name__ == "__main__":
    if not os.path.exists("data_dumps"):
        os.mkdir("data_dumps")
    run(6660, "pricm",
        folder="data_dumps",
        hp_opt="IS0", sample_domain=100)
