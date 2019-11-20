import os
import tarfile
import numpy as np
from squid import units
import cPickle as pickle


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


def get_pickled_filenames():
    return [
        t.name.replace("parsed_output/", "")
        for t in tarfile.open("parsed_output.tar.gz").getmembers()
    ]


def analyze_rosenbrock_data(percentile=0.999):
    '''
    Print out information about the data in regards to mean to max, std to max,
    and percentiles to max.
    '''
    for fptr in get_pickled_filenames():
        if not fptr.endswith(".pickle"):
            continue
        if "rosenbrock" not in fptr:
            continue
        DASH_LEN = 20
        print("-" * DASH_LEN)
        print("FILE: %s" % fptr)
        print("-" * DASH_LEN)
        # folder_data = pickle.load(open(fptr, 'r'))
        folder_data = tar_load(fptr)

        # Find best value across all
        best, worst = float('inf'), float('-inf')
        for folder, sffx_data in folder_data.items():
            for IS, IS_data in sffx_data.items():
                for sffx, data in IS_data.items():
                    if data.keys() == []:
                        continue
                    last_val = [
                        run[-1] for run in data['energies_first_above_cost']]
                    best = min(best, min(last_val))
                    worst = max(worst, max(last_val))

        if "osenbr" in fptr:
            EPS = 1.0
        elif "CO" in fptr:
            EPS = units.convert_energy("kT_300", "Ha", 1.0)
        else:
            EPS = units.convert_energy("kT_300", "kcal/mol", 1.0)
        best += EPS
        worst -= EPS

        # print max(data['energies_first_above_cost'][0])
        # raise Exception

        for folder, sffx_data in folder_data.items():
            for IS, IS_data in sffx_data.items():
                print(IS)
                for sffx, data in IS_data.items():
                    if data.keys() == []:
                        continue
                    cost = data["cost"]
                    cost_to_best = sorted([
                        cost[get_index(run, best, inv=True)]
                        for run in data['energies_first_above_cost']
                    ])
                    index = int(len(cost_to_best) * percentile)
                    mean = round(np.mean(cost_to_best), -3) / 1000.0
                    std = round(np.std(cost_to_best), -3) / 1000.0
                    perc = round(cost_to_best[index], -3) / 1000.0
                    print(folder, sffx, mean, std, perc)
            print("")
        print("-" * DASH_LEN + "\n")


def analyze_data(percentile=0.999):
    '''
    Print out information about the data in regards to mean to max, std to max,
    and percentiles to max.
    '''
    for fptr in get_pickled_filenames():
        if not fptr.endswith(".pickle"):
            continue
        if "rosenbrock" in fptr:
            continue
        DASH_LEN = 20
        print("-" * DASH_LEN)
        print("FILE: %s" % fptr)
        print("-" * DASH_LEN)
        # folder_data = pickle.load(open(fptr, 'r'))
        folder_data = tar_load(fptr)

        # Find best value across all
        best, worst = float('inf'), float('-inf')
        for folder, sffx_data in folder_data.items():
            for sffx, data in sffx_data.items():
                last_val = [
                    run[-1] for run in data['energies_first_above_cost']]
                best = min(best, min(last_val))
                worst = max(worst, max(last_val))

        if "osenbr" in fptr:
            EPS = 1.0
        elif "CO" in fptr:
            EPS = units.convert_energy("kT_300", "Ha", 1.0)
        else:
            EPS = units.convert_energy("kT_300", "kcal/mol", 1.0)
        best += EPS
        worst -= EPS

        for folder, sffx_data in folder_data.items():
            for sffx, data in sffx_data.items():
                cost = data["cost"]
                cost_to_best = sorted([
                    cost[get_index(run, best, inv=True)]
                    for run in data['energies_first_above_cost']
                ])
                # print(cost_to_best)
                # print(data['energies_first_above_cost'])
                # raise Exception("DEBUG")
                index = int(len(cost_to_best) * percentile)
                mean = round(np.mean(cost_to_best), -1) / 10
                std = round(np.std(cost_to_best), -1) / 10
                perc = round(cost_to_best[index], -1) / 10
                print(folder, sffx, mean, std, perc)
            print("")
        print("-" * DASH_LEN + "\n")


if __name__ == "__main__":
    analyze_rosenbrock_data()
    analyze_data()
