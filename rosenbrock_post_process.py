'''
'''
import os
import tarfile
import numpy as np
import cPickle as pickle
# import pickle
import matplotlib
matplotlib.use("Agg")
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
    "EGO": 'dodgerblue',
    "PCM": 'green',
    "PEARSONKG": 'green',
    "ICM": 'orange',
    "MULTITASKKG": 'orange',
    "SGP": 'navy',
    "MGP": 'navy',
    "MISOKG": 'navy'
}

color_aliases_IS = {
    "is0": 'dodgerblue',
    "full": 'green',
    "overlap": 'orange',
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

    rosenbrock_upper_cut = 95000
    max_cost = float('-inf')

    for row in open(fname, 'r').read().strip().split("\n"):
        if identifier not in row:
            continue
        a, b = row.split(identifier)
        x1, x2 = map(float, a.replace("]", " ").split("Recommendation = [0.0, ")[1].split(","))
        ENERGY[counter] = IS0(x1, x2)

        COST[counter] = float(b.split(",")[0])
        if COST[counter] > max_cost:
            max_cost = COST[counter]
        counter += 1
        if counter == len(ENERGY):
            ENERGY = ENERGY + [0.0 for _ in range(1000)]
            COST = COST + [0.0 for _ in range(1000)]

    # Only read in data that has finished.
    if max_cost < rosenbrock_upper_cut:
        return None, None

    return COST[:counter], ENERGY[:counter]


def parse_rosenbrock(models, training_IS, best_so_far=False):
    '''
    This file parses out all the rosenbrock results into a pickle file for
    plotting later.

    **Parameters**

        models: *list, str*
            A list of the model names used.
        training_IS: *str*
            What was used to train hyperparameters.
        best_so_far: *bool, optional*
            Whether to store best found so far, or the instantaneous
            recommendation value.

    **Returns**

        None
    '''

    if not isinstance(training_IS, list):
        training_IS = [training_IS]
    if not isinstance(models, list):
        models = [models]

    # Read in all data for each folder and sffx
    NAMES_FOLDERS_IS0 = [
        ("Rosenbrock_250", "rosenbrock/rosenbrock_250/",
            rosenbrock, False),
        # ("Noisy_Rosenbrock_250", "rosenbrock_noise/rosenbrock_noise_250/",
        #     rosenbrock, False),
        ("Rosenbrock_500", "rosenbrock/rosenbrock_500/",
            rosenbrock, False),
        ("Rosenbrock_1000", "rosenbrock/rosenbrock_1000/",
            rosenbrock, False),
        # ("Noisy_Rosenbrock_500", "rosenbrock_noise/rosenbrock_noise_500/",
        #     rosenbrock, False),
        # ("Rosenbrock_1000", "rosenbrock/rosenbrock_1000/",
        #     rosenbrock, True),
        ("Noisy_Rosenbrock_1000", "rosenbrock_noise/rosenbrock_1000/",
            rosenbrock, False)
    ]
    ei_folder = "rosenbrock/rosenbrock"

    legacy_aliases = {
        'miso': 'misokg',
        'pcm': 'bdpc',
        'pricm': 'bdpc',
        'ei': 'ei',
        'icm': 'bvl'
    }

    data = {}
    cost = np.arange(0, 100000, 100)

    # Find how many we are dealing with
    n_replications = None
    for _, folder, _, _ in NAMES_FOLDERS_IS0:
        fptrs = [int(f.split("_")[0]) for f in os.listdir(folder)
                 if f.endswith(".log")]
        if n_replications is None:
            n_replications = max(fptrs)
        else:
            n_replications = min(n_replications, max(fptrs))

    for name, folder, IS0, legacy in NAMES_FOLDERS_IS0:
        data[name] = {}
        for model in models:
            data[name][model] = {}

            # Handle edge case where we only ran ei in rosenbrock_noise
            if model == "ei":
                num = int(name.split("_")[-1])
                folder = "%s_%d/" % (ei_folder, num)
                # if legacy:
                #     folder = "rosenbrock/rosenbrock_%d/" % num

            for ts in training_IS:
                data[name][model][ts] = {}

                # Legacy was only done for IS0
                if legacy and ts != "is0":
                    continue

                replication_output = []
                for i in range(n_replications):

                    file_name = model + "_" + ts if model != "ei" else model
                    if legacy:
                        file_name = legacy_aliases[model]

                    l_cost, l_values = read_data(
                        folder,
                        file_name,
                        i,
                        IS0
                    )

                    # If data doesn't exist, skip
                    if l_cost is None:
                        continue

                    local_values = np.array(
                        [l_values[get_index(l_cost, c)]
                         for c in cost]
                    )
                    replication_output.append(local_values)

                    if best_so_far:
                        replication_output[-1] = np.array([
                            local_values[0]
                            if j == 0 else min(local_values[:j + 1])
                            for j in range(len(local_values))
                        ])

                # If no data, skip
                if len(replication_output) == 0:
                    print("No output for %s, %s, %s"
                          % (folder, model, ts))
                    continue

                # Now, get statistically significant data
                replication_output = np.array(replication_output)
                data[name][model][ts] = {
                    'cost': cost,
                    'energies_first_above_cost': replication_output,
                    'mean_first_above_cost': np.mean(
                        replication_output, axis=0),
                    'se_first_above_cost': np.std(
                        replication_output, axis=0
                    ) / float(len(replication_output)**0.5),
                }

    appnd = "_" +\
        ".".join(models) +\
        "_" +\
        ".".join(training_IS) +\
        ("_sorted" if best_so_far else "")
    pickle.dump(data, open("rosenbrock%s.pickle" % appnd, 'w'))


def plot_cross_rosenbrock(pfile_name, best_so_far=False):
    '''
    Plot data in processed.pickle
    '''
    # data = pickle.load(open(pfile_name, 'r'))
    data = tar_load(pfile_name)

    label_alias = {
        "Expected Improvement": "EGO",
        "Pearson Coregionalization Model": "PearsonKG",
        "Intrinsic Coregionalization Model": "MultiTaskKG",
        "Standard Gaussian Process": "misoKG",
        "is0": "Costly",
        "overlap": "Intersection",
        "full": "Full",
        # "Standard Gaussian Process": "SGP"
    }

    for sim in data.keys():
        sdata = data[sim]
        for model in sdata.keys():
            if model == "ei":
                continue
            mdata = sdata[model]
            # If we were to only plot one thing, dont.
            if sum([int(len(mdata[x].keys()) > 1) for x in mdata.keys()]) <= 1:
                continue

            for ts in mdata.keys():
                tsdata = mdata[ts]
                cost = tsdata["cost"]
                mean = tsdata["mean_first_above_cost"]
                se = tsdata["se_first_above_cost"]

                plt.plot(
                    cost, mean,
                    label=label_alias[ts],
                    linestyle=linestyles[0],
                    color=color_aliases_IS[ts],
                    linewidth=3
                )

                plt.fill_between(
                    cost, mean - 2.0 * se, mean + 2.0 * se,
                    alpha=0.5,
                    color=color_aliases_IS[ts]
                )

            # Flip axis here
            plt.gca().invert_yaxis()
            plt.ylim(-455.7, -456.3)

            ax = plt.subplot(111)
            ax.legend(
                loc='upper center', bbox_to_anchor=(0.5, 1.05),
                ncol=2, fancybox=True, shadow=True
            )
            plt.xlabel("Cost")
            plt.ylabel("Mean Function Value Exceeding Cost")
            appnd = "_" + sim + "_" + model + "_" + ts +\
                ("_sorted" if best_so_far else "")
            plt.tight_layout()
            plt.savefig("rosenbrock_imgs/cross_rosenbrock%s.png" % appnd)
            plt.close()


def tar_load(fname):
    '''
    Read in the relevant file from the parsed_output.tar.gz file.

    **Parameters**

        fname: *str*
            The file to read in.

    **Returns**

        data:
            The file contents
    '''
    tar = tarfile.open("parsed_output.tar.gz")
    members = tar.getmembers()
    mems = [t.name.replace("parsed_output/", "") for t in members]
    assert fname in mems,\
        "Error - Desired file (%s) not in parsed_output.tar.gz." % fname
    index = mems.index(fname)
    return pickle.load(tar.extractfile(members[index]))
    # return pickle.load(tar.extractfile(members[index]), encoding="latin1")


def plot_rosenbrock(pfile_name, training_IS):
    '''
    Plot data in processed.pickle
    '''
    aliases = {
        "ei": "Expected Improvement",
        "pricm": "Pearson Coregionalization Model",
        "icm": "Intrinsic Coregionalization Model",
        "miso": "Standard Gaussian Process"
    }
    label_alias = {
        "Expected Improvement": "EGO",
        "Pearson Coregionalization Model": "PearsonKG",
        "Intrinsic Coregionalization Model": "MultiTaskKG",
        "Standard Gaussian Process": "misoKG"
        # "Standard Gaussian Process": "SGP"
    }
    sffx_to_line = {
        sffx: (linestyles[i], colors[i], aliases[sffx])
        for i, sffx in enumerate(["ei", "pricm", "icm", "miso"])
    }

    # data = pickle.load(open(pfile_name, 'r'))
    data = tar_load(pfile_name)

    for sim in data.keys():
        sdata = data[sim]

        for model in sdata.keys():
            mdata = sdata[model][training_IS]
            if list(mdata.keys()) == []:
                continue
            cost = mdata["cost"]
            mean = mdata["mean_first_above_cost"]
            se = mdata["se_first_above_cost"]

            ls, c, lname = sffx_to_line[model]

            plt.plot(
                cost,
                mean,
                label=label_alias[lname],
                linestyle=ls,
                color=color_aliases[label_alias[lname].upper()],
                linewidth=3
            )
            plt.fill_between(
                cost,
                mean - 2.0 * se,
                mean + 2.0 * se,
                alpha=0.5,
                color=c
            )

        # Flip axis here
        plt.gca().invert_yaxis()
        #plt.ylim(-455.9, -456.3)
        plt.ylim(-455.7, -456.3)

        plt.gcf().subplots_adjust(left=0.15)

        # plt.legend(loc="lower right")
        plt.legend(
            loc='upper center', bbox_to_anchor=(0.5, 1.05),
            ncol=2, fancybox=True, shadow=True
        )
        plt.xlabel("Cost")
        plt.ylabel("Mean Function Value Exceeding Cost")

        append = "_" + sim + ("_sorted" if "sorted" in pfile_name else "")
        append += "_" + training_IS
        plt.savefig("rosenbrock_imgs/rosenbrock%s.png" % (append))
        plt.close()


if __name__ == "__main__":
    models = ["pricm", "icm", "miso", "ei"]
    training_IS = ["is0", "full", "overlap"]

    if not os.path.exists("rosenbrock_imgs"):
        os.mkdir("rosenbrock_imgs")
    else:
        os.system("rm rosenbrock_imgs/*")

    ##########################################################################

    # Parse the rosenbrock benchmarks into pickle files

    # parse_rosenbrock(
    #     models, training_IS, best_so_far=False)
    # parse_rosenbrock(
    #     models, training_IS, best_so_far=True)

    ##########################################################################

    # # Plot the output for various models
    for tsi in training_IS:
        plot_rosenbrock(
            "rosenbrock_pricm.icm.miso.ei_is0.full.overlap.pickle",
            tsi
        )
    # plot_rosenbrock(
    #     "rosenbrock_pricm.icm.miso.ei_is0.full.overlap_sorted.pickle",
    #     "is0"
    # )

    ##########################################################################

    # Plot the comparison of training sets for the Supplementary Information

    plot_cross_rosenbrock(
        "rosenbrock_pricm.icm.miso.ei_is0.full.overlap.pickle",
        best_so_far=False
    )
    # plot_cross_rosenbrock(
    #     "rosenbrock_pricm.icm.miso.ei_is0.full.overlap_sorted.pickle",
    #     best_so_far=True
    # )
