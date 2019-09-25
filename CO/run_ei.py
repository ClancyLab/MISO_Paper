from pal.opt import Optimizer
import pal.utils.strings as pal_strings
from pal.constants.solvents import solvents
from pal.kernels.matern import maternKernel52 as mk52
from pal.kernels.squared import squared
from pal.stats.likelihood import gaussian as g_loglike
from pal.acquisition.misokg import getNextSample_misokg
from pal.acquisition.EI import getNextSample_EI
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


def run(run_index,
        folder="data_dumps", sample_domain=1000):
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
        folder: *str, optional*
            What to name the folder where the data will go.
        sample_domain: *int, optional*
            How many data points to sample from the domain.
    '''

    # Generate the main object
    sim = Optimizer()

    # Assign simulation properties
    sim.hyperparameter_objective = MLE
    sim.acquisition = getNextSample_EI
    ###################################################################################################
    # File names
    sim.fname_out = None
    sim.fname_historical = None

    sim.logger_fname = "%s/%d_%s.log" % (folder, run_index, "ei")
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
    IS0 = pickle.load(open("IS0.pickle", 'r'))

    sim.IS = [
        lambda x1: -1.0 * IS0[int((x1 - 0.5) * 1000.0)][0],
    ]
    sim.costs = [
        np.mean([IS[1] for IS in IS0]),
    ]

    sim.save_extra_files = False
    ########################################
    sim.numerical = True
    sim.historical_nsample = 5
    sim.domain = [
        (0.5, 2.5)
    ]
    sim.sample_n_from_domain = sample_domain
    ########################################

    sim.n_start = 10  # The number of starting MLE samples
    # sim.reopt = 20
    sim.reopt = float("inf")
    sim.ramp_opt = None
    sim.parallel = False

    # Parameters for debugging and overwritting
    sim.debug = False
    sim.verbose = True
    sim.overwrite = True  # If True, warning, else Error

    # Functional forms of our mean and covariance
    sim.mean = lambda X, Y, theta: np.array([0.0 for _ in Y])
    def cov(X0, Y, theta):
        return squared(np.array(X0), [theta.l1], theta.sig_1)

    sim.cov = cov

    sim.theta.bounds = {}
    sim.theta.sig_1, sim.theta.bounds['sig_1'] = None, (1E-2, lambda _, Y: np.var(Y))
    sim.theta.l1, sim.theta.bounds['l1'] = None, (1E-1, 1)

    sim.theta.rho = {str(sorted([i, j])): 1.0 for i in range(len(sim.IS)) for j in range(i, len(sim.IS))}
    for k in sim.theta.rho.keys():
        sim.theta.bounds['rho %s' % k] = (0.1, 1.0)
        a, b = eval(k)
        if a != b:
            sim.theta.bounds['rho %s' % k] = (0.01, 1.0 - 1E-6)

    sim.theta.set_hp_names()

    # Define how we update hyperparameters
    sim.update_hp_only_with_IS0 = False
    sim.update_hp_only_with_overlapped = False
 
    # These should be False by default, but ensure they are
    sim.theta.normalize_L = False
    sim.theta.normalize_Ks = False
    sim.preconditioned = False

    # Assign our likelihood function.
    sim.loglike = g_loglike
    ###################################################################################################

    # Start simulation
    sim.iteration_kill_switch = None
    sim.cost_kill_switch = 3000
    sim.run()


if __name__ == "__main__":
    if not os.path.exists("data_dumps"):
        os.mkdir("data_dumps")
    run(6660, folder="data_dumps", sample_domain=100)

