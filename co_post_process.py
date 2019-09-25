'''
'''
import os
import numpy as np
from squid import units
import cPickle as pickle
from matplotlib import pyplot as plt

linestyles = [
    (0, ()),
    (0, (1, 1)),

    (0, (5, 10)),
    (0, (5, 5)),

    (0, (3, 1, 1, 1)),
    (0, (3, 10, 1, 10)),
    (0, (3, 5, 1, 5)),
    (0, (5, 1)),

    (0, (3, 10, 1, 10, 1, 10)),
    (0, (3, 5, 1, 5, 1, 5)),
    (0, (3, 1, 1, 1, 1, 1)),
    (0, (1, 10)),
    (0, (1, 5)),
]

color_aliases = {
    "EI": 'dodgerblue',
    "PCM": 'green',
    "ICM": 'orange',
    "SGP": 'navy',
}

colors = [
    'dodgerblue',  # EI
    'green',  # PCM
    'orange',  # ICM
    'navy',  # SGP
    'pink',
    'purple',
    'black',
    'mediumblue',
    'goldenrod',
    'lightgreen',
    'seagreen',
    'saddlebrown',
    'red',
]


def rosenbrock(x1, x2):
    '''
    This is the rosenbrock function
    '''
    return 1.0 * ((1.0 - x1)**2 + 100.0 * (x2 - x1**2)**2 - 456.3)


def convert_perov_to_key(perov):
    '''
    Given a perovskite name (ex. "MAPbBrClCl_THTO_0"), convert it to a
    key (ex. "BrClCl MA THTO").

    **Parameters**

        perov: *str*
            The perovskite name in "ABX3_S_i" format.

    **Returns**

        key: *str*
            The key in "X3 A S" format.
    '''
    ABX, S, _ = perov.split("_")
    A, X3 = ABX.split("Pb")
    return " ".join([X3, A, S])


def get_index(lc, c, inv=False):
    '''
    We want to find the index of the value that first exceeds a cost.

    **Parameters**

        lc: *list, float*
            Some list of values.
        c: *float*
            A value that we want to find we first exceed.
        inv: *bool, optional*
            Whether to look at the inverse problem.

    **Returns**

        i: *int*
            Index that first exceeds some cost

    **References**

        - https://stackoverflow.com/a/2236935
    '''
    try:
        if inv:
            return next(i for i, x in enumerate(lc) if x < c)
        else:
            return next(i for i, x in enumerate(lc) if x > c)
    except StopIteration:
        return -1


def read_data(folder, sffx, i, IS0, verbose=True):
    '''
    For a given folder, iteration, and sffx, read in the cost and associated
    energy.

    **Parameters**

        folder: *str*
            The folder to analyze.
        sffx: *str*
            The sffx you want to read.
        IS0: *dict*
            The dictionary of perovskites ("H1H2H3 C S") to corresponding
            binding energy.
        verbose: *bool, optional*
            Whether to print out verbose statements or not.

    **Returns**

        cost: *list, float*
            A list of total cost at each iteration.
        energy: *list, float*
            The corresponding energy of the best prediction.
    '''
    if not folder.endswith("/"):
        folder += "/"
    fname = "%s%d_%s.log" % (folder, i, sffx)

    if not os.path.exists(fname):
        if verbose:
            print("%s does not exist." % fname)
        return None, None

    # This is what we will split by.  Right before this should be
    # the recommended perovskite, and right after is the current cost.
    identifier = ", Current Cost ="
    ENERGY = [0.0 for _ in range(1000)]
    COST = [0.0 for _ in range(1000)]
    counter = 0

    for row in open(fname, 'r').read().strip().split("\n"):
        if identifier not in row:
            continue
        a, b = row.split(identifier)
        x1 = float(a.split("Recommendation = [0.0, ")[1].split(",")[0][:-1])
        ENERGY[counter] = IS0(x1)
        COST[counter] = float(b.split(",")[0])
        counter += 1
        if counter == len(ENERGY):
            ENERGY = ENERGY + [0.0 for _ in range(1000)]
            COST = COST + [0.0 for _ in range(1000)]

    return COST[:counter], ENERGY[:counter]


def parse_co(models, training_IS, best_so_far=False):

    IS_CO0_f = pickle.load(open("CO/IS0.pickle", 'r'))
    IS_CO0 = lambda x1: 1.0 * IS_CO0_f[int((x1 - 0.5) * 1000.0)][0]

    # Read in all data for each folder and sffx
    FOLDERS_IS0 = [
        ("CO/CO/", IS_CO0)
    ]

    # Find how many we are dealing with
    n_replications = None
    for folder, _ in FOLDERS_IS0:
        fptrs = [int(f.split("_")[0]) for f in os.listdir(folder)
                 if f.endswith(".log")]
        if n_replications is None:
            n_replications = max(fptrs)
        else:
            n_replications = min(n_replications, max(fptrs))

    folder_data = {}
    for folder, IS0 in FOLDERS_IS0:
        cost = np.arange(0, 3000, 1)

        folder_data[folder] = {}
        for sffx in models:
            all_energies = []
            for i in range(n_replications):
                l_cost, l_energy = read_data(
                    folder,
                    sffx + "_" + training_IS if sffx != "ei" else sffx,
                    i, IS0
                )
                if l_cost is None:
                    continue

                local_energies = np.array(
                        [l_energy[get_index(l_cost, c)]
                         for c in cost]
                    )
                all_energies.append(
                    local_energies
                )

                if best_so_far:
                    all_energies[-1] = np.array([
                        local_energies[0] if j == 0 else min(local_energies[:j + 1])
                        for j in range(len(local_energies))
                    ])
            if len(all_energies) == 0:
                continue
            all_energies = np.array(all_energies)
            folder_data[folder][sffx] = {
                'cost': cost,
                'energies_first_above_cost': all_energies,
                'mean_first_above_cost': np.mean(all_energies, axis=0),
                'se_first_above_cost': np.std(all_energies, axis=0) / float(len(all_energies)**0.5),
            }

    append = ""
    if best_so_far:
        append += "_sorted"
    pickle.dump(folder_data, open("CO%s.pickle" % append, 'w'))


def parse_co_1000(models, training_IS, best_so_far=False):

    IS_CO0_f = pickle.load(open("CO/IS0.pickle", 'r'))
    IS_CO0 = lambda x1: 1.0 * IS_CO0_f[int((x1 - 0.5) * 1000.0)][0]

    # Read in all data for each folder and sffx
    FOLDERS_IS0 = [
        ("CO/CO_1000/", IS_CO0)
    ]

    # Find how many we are dealing with
    n_replications = None
    for folder, _ in FOLDERS_IS0:
        fptrs = [int(f.split("_")[0]) for f in os.listdir(folder)
                 if f.endswith(".log")]
        if n_replications is None:
            n_replications = max(fptrs)
        else:
            n_replications = min(n_replications, max(fptrs))

    legacy_sffx = {
        "pcm": "bdpc",
        "pricm": "bdpc",
        "miso": "misokg",
        "icm": "bvl",
        "ei": "ei"
    }

    folder_data = {}
    for folder, IS0 in FOLDERS_IS0:
        cost = np.arange(0, 3000, 1)

        folder_data[folder] = {}
        for sffx in models:
            all_energies = []
            for i in range(n_replications):
                l_cost, l_energy = read_data(
                    folder, legacy_sffx[sffx], i, IS0
                )
                if l_cost is None:
                    continue

                local_energies = np.array(
                        [l_energy[get_index(l_cost, c)]
                         for c in cost]
                    )
                all_energies.append(
                    local_energies
                )

                if best_so_far:
                    all_energies[-1] = np.array([
                        local_energies[0] if j == 0 else min(local_energies[:j + 1])
                        for j in range(len(local_energies))
                    ])
            if len(all_energies) == 0:
                continue
            all_energies = np.array(all_energies)
            folder_data[folder][sffx] = {
                'cost': cost,
                'energies_first_above_cost': all_energies,
                'mean_first_above_cost': np.mean(all_energies, axis=0),
                'se_first_above_cost': np.std(all_energies, axis=0) / float(len(all_energies)**0.5),
            }

    append = ""
    if best_so_far:
        append += "_sorted"
    pickle.dump(folder_data, open("CO_1000%s.pickle" % append, 'w'))


if __name__ == "__main__":
    models = ["pricm", "icm", "miso", "ei"]

    # Generate the pickle files for plotting
    parse_co(
        models, "overlap", best_so_far=False)
