from pal.opt import Optimizer
import pal.utils.strings as pal_strings
from pal.constants.solvents import solvents
from pal.kernels.matern import maternKernel52 as mk52
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


def run(run_index, model, infosources,
        folder="data_dumps", exact_cost=False, hp_opt="IS0",
        generate_new_historical=False, N_historical=10):
    '''
    This function will run HOIP optimization using one of several coregionalization methods.

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
        infosources: *int*
            The information source to use:
                0 - IS_N1R3 vs IS_N1R2
                1 - IS_N3R2 vs IS_N1R2
                2 - IS_N5R2 vs IS_N3R2
                3 - IS_N5R2 vs IS_N3R2 vs IS_N1R2
        folder: *str, optional*
            What to name the folder where the data will go.
        exact_cost: *bool, optional*
            Whether to use the exact cost (approximated by time to simulate in
            DFT), or an approximate cost (done as order magnitude differences).
        hp_opt: *str, optional*
            With what data should the hyperparameters be parameterized.
            Options: IS0, full, overlap
        generate_new_historical: *bool, optional*
            Whether to generate new historical data, or read in previous
            historical data.
        N_historical: *int, optional*
            The number of historical data to generate when generate_new_historical
            is set to True.
    '''

    hp_opt = hp_opt.lower()
    allowed_hp_opt = ["is0", "full", "overlap"]
    assert hp_opt in allowed_hp_opt, "Error, hp_opt (%s) not in %s" % (hp_opt, ", ".join(allowed_hp_opt))

    # Stored data
    IS_N5R2 = pickle.load(open("enthalpy_N5_R2_wo_GBL_Ukcal-mol", 'r'))
    IS_N3R2 = pickle.load(open("enthalpy_N3_R2_Ukcal-mol", 'r'))
    IS_N1R2 = pickle.load(open("enthalpy_N1_R2_Ukcal-mol", 'r'))
    IS_N1R3 = pickle.load(open("enthalpy_N1_R3_Ukcal-mol", 'r'))

    # Grab the IS and costs
    IS2 = None
    if infosources == 0:
        IS0 = IS_N1R3
        IS1 = IS_N1R2
        if exact_cost:
            costs = [6.0, 1.0]
        else:
            costs = [10.0, 1.0]
    elif infosources == 1:
        IS0 = IS_N3R2
        IS1 = IS_N1R2
        if exact_cost:
            costs = [14.0, 1.0]
        else:
            costs = [10.0, 1.0]
    elif infosources == 2:
        IS0 = IS_N5R2
        IS1 = IS_N3R2
        if exact_cost:
            costs = [27.0, 14.0]
        else:
            costs = [100.0, 10.0]
    elif infosources == 3:
        IS0 = IS_N5R2
        IS1 = IS_N3R2
        IS2 = IS_N1R2
        if exact_cost:
            costs = [27.0, 14.0, 1.0]
        else:
            costs = [100.0, 10.0, 1.0]
    else:
        raise Exception("An invalid infosources was specified.")

    # Trim so we only have data in all IS
    IS1 = {k: IS1[k] for k in IS0.keys()}
    if IS2 is not None:
        IS2 = {k: IS2[k] for k in IS0.keys()}

    # Generate the main object
    sim = Optimizer()

    # Assign simulation properties
    sim.hyperparameter_objective = MLE
    ###################################################################################################
    # File names
    sim.fname_out = "enthalpy_%s.dat" % model
    sim.fname_historical = None

    # Information sources, in order from expensive to cheap
    sim.IS = [
        lambda h, c, s: -1.0 * IS0[' '.join([''.join(h), c, s])],
        lambda h, c, s: -1.0 * IS1[' '.join([''.join(h), c, s])],
    ]
    if IS2 is not None:
        sim.IS.append(lambda h, c, s: -1.0 * IS2[' '.join([''.join(h), c, s])])
    sim.costs = costs

    # Logging of output
    sim.logger_fname = "%s/%d_%s_%s.log" % (folder, run_index, model, hp_opt)
    if os.path.exists(sim.logger_fname):
        os.system("rm %s" % sim.logger_fname)
    os.system("touch %s" % sim.logger_fname)

    sim.save_extra_files = False
    ########################################

    if generate_new_historical:
        sim.historical_nsample = N_historical
        print("Generating a new historical data set of %d samples." % sim.historical_nsample)
        # Override the possible combinations with the reduced list of IS0
        # Because we do this, we should also generate our own historical sample
        combos_no_IS = [k[1] + "Pb" + k[0] + "_" + k[2] for k in [key.split() for key in IS0.keys()]]
        choices = np.random.choice(combos_no_IS, sim.historical_nsample, replace=False)
        tmp_data = pal_strings.alphaToNum(
            choices,
            solvents,
            mixed_halides=True,
            name_has_IS=False)
    
        data = []
        for IS in range(len(sim.IS)):
            for i, d in enumerate(tmp_data):
                h, c, _, s, _ = pal_strings.parseName(pal_strings.parseNum(d, solvents, mixed_halides=True, num_has_IS=False), name_has_IS=False)
                c = c[0]
                data.append([IS] + d + [sim.IS[IS](h, c, s)])
    
        sim.fname_historical = "%s/%d.history" % (folder, run_index)
        pickle.dump(data, open(sim.fname_historical, 'w'))
        simple_data = [d for d in data if d[0] == 0]
        pickle.dump(simple_data, open("%s/%d_reduced.history" % (folder, run_index), 'w'))

        # Get a list of all combinations
        all_combos = [c + "_" + str(i) for i in range(len(sim.IS)) for c in combos_no_IS]
        pickle.dump(all_combos, open("%s/%d.combos" % (folder, run_index), 'w'))
        sim.combinations = all_combos
    else:
        sim.fname_historical = "%s/%d.history" % (folder, run_index)
        print "Waiting on %s to be written..." % sim.fname_historical,
        while not all([os.path.exists(sim.fname_historical), os.path.exists("%s/%d.combos" % (folder, run_index))]):
            time.sleep(30)
        print " DONE"
        sim.historical_nsample = len(pickle.load(open("%s/%d_reduced.history" % (folder, run_index), 'r')))
        sim.combinations = pickle.load(open("%s/%d.combos" % (folder, run_index), 'r'))

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

    sim.acquisition = getNextSample_misokg

    # Functional forms of our mean and covariance
    # MEAN: 4 * mu_alpha + mu_zeta
    # COV: sig_alpha * |X><X| + sig_beta * I_N + sig_zeta + MaternKernel(S, weights, sig_m)

    SCALE = [2.0, 4.0][int(sim.mixed_halides)]
    # _1, _2, _3 used as dummy entries
    def mean(X, Y, theta):
        mu = np.array([SCALE * theta.mu_alpha + theta.mu_zeta for _ in X])
        return mu
    sim.mean = mean

    def cov(X0, Y, theta, split=False):
        '''
        This is the covariance Kx from the previous paper.  However, in this case
        we also calculate a Ks as the coregionalization matrix based on either the
        PRICM or ICM method.
        '''
        A = theta.sig_alpha * np.dot(np.array(X0)[:, :-3], np.array(X0)[:, :-3].T)
        B = theta.sig_beta * np.diag(np.ones(len(X0)))
        C = theta.sig_zeta
        D = mk52(np.array(X0)[:, -3:-1], [theta.l1, theta.l2], theta.sig_m)
        Kx = A + B + C + D

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

    # # NOTE! theta.rho is a reserved keyword in misoKG.  We will generate a list of the same length
    # # of the information sources, and use this for scaling our IS.
    if model.lower() == "icm":
        sim.theta.rho = {str(sorted([i, j])): None for i in range(len(sim.IS)) for j in range(i, len(sim.IS))}
    elif model.lower() == "pricm":
        sim.theta.rho = {str(sorted([i, j])): 1.0 for i in range(len(sim.IS)) for j in range(i, len(sim.IS))}
        sim.dynamic_pc = True
    else:
        raise Exception("Invalid model.  Use ICM or PRICM")
    for k in sim.theta.rho.keys():
        sim.theta.bounds['rho %s' % k] = (0.1, 1.0)
        a, b = eval(k)
        if a != b:
            sim.theta.bounds['rho %s' % k] = (0.01, 1.0 - 1E-6)
    sim.theta.set_hp_names()

    # Set a kill switch so that when the best is found we stop searching.
    h, c, s = min([(IS0[k], k) for k in IS0.keys()])[1].split()
    sim.recommendation_kill_switch = "%sPb%s_%s_0" % (c, h, s)
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
    sim.run()


if __name__ == "__main__":
    if not os.path.exists("data_dumps"):
        os.mkdir("data_dumps")
    run(6660, "pricm", 0,
        folder="data_dumps", exact_cost=False, hp_opt="IS0",
        generate_new_historical=True, N_historical=10)
