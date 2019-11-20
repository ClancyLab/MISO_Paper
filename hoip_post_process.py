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

colors = [
    'dodgerblue',  # EI
    'green',  # PRICM
    'orange',  # ICM
    'navy',  # MISO
    'seagreen',
    'saddlebrown',
    'black',
    'purple',
    'pink',
    'red',
    'mediumblue',
    'goldenrod',
    'lightgreen',
]

color_aliases = {
    "EI": colors[0],
    "PCM": colors[1],
    "PRICM": colors[1],
    "ICM": colors[2],
    "SGP": colors[3],
    "PCM0": colors[4],
    "PRICM0": colors[4],
    "PCM1": colors[5],
    "PRICM1": colors[5],
    "PCM2": colors[6],
    "PRICM2": colors[6],
}

line_aliases = {
    "is0": linestyles[0],
    "full": linestyles[1],
    "overlap": linestyles[2],
}


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

    rosen = "rosenbrock" in folder
    COMol = "CO" in folder

    for row in open(fname, 'r').read().strip().split("\n"):
        if identifier not in row:
            continue
        a, b = row.split(identifier)
        if rosen:
            x1, x2 = map(float, a.replace("]", " ").split("Recommendation = [0.0, ")[1].split(","))
            ENERGY[counter] = IS0(x1, x2)
        elif COMol:
            x1 = float(a.split("Recommendation = [0.0, ")[1].split(",")[0][:-1])
            ENERGY[counter] = IS0(x1)
        else:
            ENERGY[counter] = IS0[convert_perov_to_key(a.split()[-1])]
        COST[counter] = float(b.split(",")[0])
        counter += 1
        if counter == len(ENERGY):
            ENERGY = ENERGY + [0.0 for _ in range(1000)]
            COST = COST + [0.0 for _ in range(1000)]

    return COST[:counter], ENERGY[:counter]


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


def parse_hoips(models, best_so_far=True, benchmark_name="N1R3_N1R2_TC"):
    # We want to plot "value first exceeding cost"
    # https://stackoverflow.com/a/2236935
    def get_index(lc, c):
        try:
            return next(x[0] for x in enumerate(lc) if x[1] > c)
        except StopIteration:
            return -1

    IS_N1R3 = pickle.load(open(
        "hoips/enthalpy_N1_R3_Ukcal-mol", 'r'))
    IS_N3R2 = pickle.load(open(
        "hoips/enthalpy_N3_R2_Ukcal-mol", 'r'))
    IS_N5R2 = pickle.load(open(
        "hoips/enthalpy_N5_R2_wo_GBL_Ukcal-mol", 'r'))

    IS_0 = {
        "N1R3": IS_N1R3,
        "N3R2": IS_N3R2,
        "N5R2": IS_N5R2,
    }[benchmark_name.split("_")[0]]

    # Read in all data for each folder and sffx
    NAMES_FOLDERS_IS0_TIS = [
        ("MisoKG_IS0",
            "hoips/%s/" % benchmark_name, IS_0, "is0"),
        ("MisoKG_Full_IS",
            "hoips/%s/" % benchmark_name, IS_0, "full"),
        ("MisoKG_Overlapped_IS",
            "hoips/%s/" % benchmark_name, IS_0, "overlap"),
    ]

    # Find how many we are dealing with
    n_replications = None
    folder = NAMES_FOLDERS_IS0_TIS[0][1]
    fptrs = [int(f.split("_")[0]) for f in os.listdir(folder)
             if f.endswith(".log")]
    if n_replications is None:
        n_replications = max(fptrs)
    else:
        n_replications = min(n_replications, max(fptrs))

    sffx_data = {}
    cost = np.arange(5, 3000, 5)
    # Loop through each model
    for sffx in models:
        sffx_data[sffx] = {}
        # For each variation, read in and analyze all results
        for name, folder, IS0, training_IS in NAMES_FOLDERS_IS0_TIS:
            all_energies = []
            for i in range(n_replications):
                l_cost, l_energy = read_data(
                    folder,
                    sffx + "_" + training_IS if sffx != "ei" else sffx,
                    i, IS0)

                # If nothing read in, skip
                if l_cost is None:
                    continue

                # Parse data so we get "value first exceeding cost"
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
            # If no data, skip
            if len(all_energies) == 0:
                continue

            # Now, get statistically significant data
            all_energies = np.array(all_energies)
            sffx_data[sffx][name] = {
                'cost': cost,
                'energies_first_above_cost': all_energies,
                'mean_first_above_cost': np.mean(all_energies, axis=0),
                'se_first_above_cost': np.std(all_energies, axis=0) / float(len(all_energies)**0.5),
            }

    appnd = "_" + benchmark_name
    if best_so_far:
        appnd += "_sorted"
    pickle.dump(sffx_data, open("cross_processed_miso%s.pickle" % appnd, 'w'))


def plot_hoips(benchmark_name, best_so_far=False):
    '''
    Plot data in processed.pickle
    '''

    aliases = {
        "ei": "Expected Improvement",
        "pricm": "Pearson Coregionalization Model",
        "icm": "Intrinsic Coregionalization Model",
        "miso": "Standard Gaussian Process"
    }

    folder_name_keys = [
        "MisoKG_IS0",
        "MisoKG_Full_IS",
        "MisoKG_Overlapped_IS",
    ]

    label_aliases = {
        "MisoKG_IS0": "IS$_0$",
        "MisoKG_Full_IS": "Full",
        "MisoKG_Overlapped_IS": "Overlap"
    }

    appnd = "_" + benchmark_name
    if best_so_far:
        appnd += "_sorted"

    sffx_data = tar_load("cross_processed_miso%s.pickle" % appnd)

    EI = sffx_data["ei"]["MisoKG_IS0"]

    for sffx, folder_data in sffx_data.items():
        if sffx == "ei":
            continue
        if folder_data == {}:
            continue

        # c = colors[sffx]
        sffx_alias = aliases[sffx]
        for i, lname in enumerate(folder_name_keys):
            if lname not in folder_data:
                continue
            ls = linestyles[i]
            c = colors[2 + i]

            if "is0" in lname.lower():
                ls = line_aliases["is0"]
            elif "full" in lname.lower():
                ls = line_aliases["full"]
            elif "overlap" in lname.lower():
                ls = line_aliases["overlap"]
            else:
                raise Exception("ERROR")

            append = ""
            if any(["pcm" == sffx, "pricm" == sffx]):
                append = str(i)

            plt.plot(
                folder_data[lname]["cost"],
                folder_data[lname]["mean_first_above_cost"],
                label=label_aliases[lname].upper() if label_aliases[lname] != "pricm" else "PearsonKG",
                linestyle=ls,
                color=color_aliases[sffx.upper() + append],
                linewidth=3
            )

            plt.fill_between(
                folder_data[lname]["cost"],
                folder_data[lname]["mean_first_above_cost"] - 2.0 * folder_data[lname]["se_first_above_cost"],
                folder_data[lname]["mean_first_above_cost"] + 2.0 * folder_data[lname]["se_first_above_cost"],
                alpha=0.5,
                color=color_aliases[sffx.upper() + append]
            )

        # Plot EI
        ls = linestyles[0]
        c = colors[0]
        plt.plot(
            EI["cost"],
            EI["mean_first_above_cost"],
            label="EI",
            linestyle=line_aliases["is0"],
            color=color_aliases["EI"],
            linewidth=3
        )
        plt.fill_between(
            EI["cost"],
            EI["mean_first_above_cost"] - 2.0 * EI["se_first_above_cost"],
            EI["mean_first_above_cost"] + 2.0 * EI["se_first_above_cost"],
            alpha=0.5,
            color=color_aliases["EI"]
        )

        # Flip axis here
        plt.gca().invert_yaxis()
        #plt.ylim(-12.5, -13)
        # plt.ylim(-450, -460)
        # plt.ylim(-455.7, -456.4)

#        plt.title("Comparison of Hyperparameter Training Sets for %s" % sffx_alias, y=1.08)
        # plt.legend(loc="upper left")
        ax = plt.subplot(111)
        ax.legend(
            loc='upper center', bbox_to_anchor=(0.7, 0.2),
            # loc='upper center', bbox_to_anchor=(0.5, 0.05),
            ncol=2, fancybox=True, shadow=True
        )
        # plt.legend(loc="lower right")
        plt.xlabel("Cost")
        plt.ylabel("Mean Energy Exceeding Cost (kcal/mol)")
        appnd = "_" + benchmark_name
        if best_so_far:
            appnd = "_sorted"
        plt.savefig("hoip_imgs/hoips_%s%s.png" % (sffx, appnd))
        plt.close()


def plot_hoips_2(benchmark_name, training_IS, best_so_far=False):
    '''
    Plot data in processed.pickle
    '''

    aliases = {
        "ei": "Expected Improvement",
        "pricm": "Pearson Coregionalization Model",
        "icm": "Intrinsic Coregionalization Model",
        "miso": "Standard Gaussian Process"
    }

    folder_name_keys = [
        "MisoKG_IS0",
        "MisoKG_Full_IS",
        "MisoKG_Overlapped_IS",
    ]

    label_aliases = {
        "EI": "EGO",
        "PRICM": "PearsonKG",
        "PCM": "PearsonKG",
        "ICM": "MultiTaskKG",
        "MGP": "misoKG"
    }

    appnd = "_" + benchmark_name
    if best_so_far:
        appnd += "_sorted"

    sffx_data = tar_load("cross_processed_miso%s.pickle" % appnd)

    EI = sffx_data["ei"]["MisoKG_IS0"]

    i = 0
    for sffx, folder_data in sffx_data.items():
        if sffx == "ei":
            continue
        if folder_data == {}:
            continue

        sffx_alias = aliases[sffx]
        for j, lname in enumerate(folder_name_keys):
            if lname != training_IS:
                continue
            if lname not in folder_data:
                continue
            ls = linestyles[i]
            c = colors[2 + i]
            i += 1

            if "is0" in lname.lower():
                ls = line_aliases["is0"]
            elif "full" in lname.lower():
                ls = line_aliases["full"]
            elif "overlap" in lname.lower():
                ls = line_aliases["overlap"]
            else:
                raise Exception("ERROR")

            plt.plot(
                folder_data[lname]["cost"],
                folder_data[lname]["mean_first_above_cost"],
                label=label_aliases[sffx.upper()],
                linestyle=ls,
                color=color_aliases[sffx.upper()],
                linewidth=3
            )

            plt.fill_between(
                folder_data[lname]["cost"],
                folder_data[lname]["mean_first_above_cost"] - 2.0 * folder_data[lname]["se_first_above_cost"],
                folder_data[lname]["mean_first_above_cost"] + 2.0 * folder_data[lname]["se_first_above_cost"],
                alpha=0.5,
                color=color_aliases[sffx.upper()]
            )

    # Plot EI
    ls = linestyles[0]
    c = colors[0]
    plt.plot(
        EI["cost"],
        EI["mean_first_above_cost"],
        label="EGO",
        linestyle=line_aliases["is0"],
        color=color_aliases["EI"],
        linewidth=3
    )
    plt.fill_between(
        EI["cost"],
        EI["mean_first_above_cost"] - 2.0 * EI["se_first_above_cost"],
        EI["mean_first_above_cost"] + 2.0 * EI["se_first_above_cost"],
        alpha=0.5,
        color=color_aliases["EI"]
    )

    # Flip axis here
    plt.gca().invert_yaxis()
    #plt.ylim(-12.5, -13)
    # plt.ylim(-450, -460)
    # plt.ylim(-455.7, -456.4)

#    plt.title("Comparison of Hyperparameter Training Sets for %s" % sffx_alias, y=1.08)
    # plt.legend(loc="upper left")
    ax = plt.subplot(111)
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.7, 0.2),
        # loc='upper center', bbox_to_anchor=(0.5, 0.05),
        ncol=2, fancybox=True, shadow=True
    )
    # plt.legend(loc="lower right")
    plt.xlabel("Cost")
    plt.ylabel("Mean Energy Exceeding Cost (kcal/mol)")
    appnd = "_" + benchmark_name
    if best_so_far:
        appnd += "_sorted"
    plt.tight_layout()
    if "overlap" in training_IS.lower():
        tis = "overlap"
    elif "is0" in training_IS.lower():
        tis = "IS0"
    else:
        tis = "full"
    plt.savefig("hoip_imgs/HOIP_%s%s.png" % (tis, appnd))
    plt.close()


def plot_raw_data(sort_by="N1R2"):
    '''
    This function reads in the enthalpy_N*_R^_Ukcal-mol data files and plots
    them on a single graph.  To make the plot more readable, it is sorted to
    an information source specified (by default N1R2, the single solvent
    at the GGA level of theory).

    Note - The naming convention for enthalpy_N5_R2_wo_GBL_Ukcal-mol is
    done due to the fact that we found out our GBL data was grossly inaccurate
    for that data set, so it was removed.  Inaccuracy was due to unconverged
    data on a cluster that had crashed.

    **Parameters**

        sort_by: *str, optional*
            The data set to sort by.

    **Returns**

        None.
    '''
    N1R2 = pickle.load(open("hoips/enthalpy_N1_R2_Ukcal-mol", 'r'))
    N1R3 = pickle.load(open("hoips/enthalpy_N1_R3_Ukcal-mol", 'r'))
    N3R2 = pickle.load(open("hoips/enthalpy_N3_R2_Ukcal-mol", 'r'))
    N5R2 = pickle.load(open("hoips/enthalpy_N5_R2_wo_GBL_Ukcal-mol", 'r'))

    all_keys = list(N1R2.keys()) + list(N1R3.keys()) +\
        list(N3R2.keys()) + list(N5R2.keys())
    all_keys = list(set(all_keys))

    data = [
        (N1R2[k] if k in N1R2 else None,
         N1R3[k] if k in N1R3 else None,
         N3R2[k] if k in N3R2 else None,
         N5R2[k] if k in N5R2 else None)
        for k in all_keys
    ]

    data.sort(
        key=lambda x: x[["N1R2", "N1R3", "N3R2", "N5R2"].index(sort_by)]
    )

    N1R2, N1R3, N3R2, N5R2 = zip(*data)
    xvals = list(range(len(all_keys)))
    plt.plot(xvals, N1R2, 'bD', markersize=4, label="GGA-1")  # N1R2
    plt.plot(xvals, N1R3, 'gs', markersize=4, label="Hybrid-1")  # N1R3
    plt.plot(xvals, N3R2, 'r.', markersize=4, label="GGA-3")  # N3R2
    plt.plot(xvals, N5R2, 'c*', markersize=4, label="GGA-5")  # N5R2

    plt.xlabel("HOIP-Solvent System")
    plt.ylabel("Intermolecular Binding Energy (kcal/mol)")
    plt.legend()

    plt.savefig("hoips/IS_comparison.png")


if __name__ == "__main__":
    models = ["pricm", "icm", "ei"]
    if not os.path.isdir("hoip_imgs"):
        os.mkdir("hoip_imgs")

    # Generate the pickle files for plotting
    for benchmark_name in ["N1R3_N1R2_TC", "N3R2_N1R2_TC", "N5R2_N3R2_TC"]:
#        parse_hoips(models, best_so_far=False, benchmark_name=benchmark_name)
#        plot_hoips(benchmark_name, best_so_far=False)
        plot_hoips_2(benchmark_name, "MisoKG_Overlapped_IS", best_so_far=False)
        plot_hoips_2(benchmark_name, "MisoKG_IS0", best_so_far=False)


    # Generate the comparison of IS from the enthalpy pickled files.
    plot_raw_data("N1R2")
    # plot_raw_data("N1R3")
    # plot_raw_data("N3R2")
    # plot_raw_data("N5R2")
