'''
This script handles job submission for statistical significance.
'''
import os
import time
from squid.jobs import pysub

def fullreplace(book, word, newword):
    '''
    Simply replace every instance of word in some string with
    a new word.
    '''
    while word in book:
        book = book.replace(str(word), str(newword))
    return book


def run_pricm(infosources, folder, cost, RANGE=range(5), QUEUE=None, WALLTIME="1-00:00:00", N_History=10):
    '''
    Submit a number of miso jobs for HOIP optimization.
    
    **Parameters**

        infosources: *int*
            The information source to use:
                0 - IS_N1R3 vs IS_N1R2
                1 - IS_N3R2 vs IS_N1R2
                2 - IS_N5R2 vs IS_N3R2
                3 - IS_N5R2 vs IS_N3R2 vs IS_N1R2
        folder: *str*
            The name for the output folder.
        cost: *bool*
            The cost of the information source.  Mainly, True if the true cost
            was used in the corresponding run_ei file, and False if it is an
            approximation.  This will only change the naming of files so we
            keep track of this.

    **Returns**

        None
    '''
    script = '''
import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "$MODEL$",
    $IS$,
    folder="$FOLDER$",
    exact_cost=$COST$,
    hp_opt="$HP_OPT$",
    generate_new_historical=$GNH$,
    N_historical=$N_HISTORY$
)
'''.strip()
    systems = [
#       Model   ,  HP_Opt  , generate_new_historical
#        ("pricm", "is0"    , True                   ),
#        ("icm"  , "is0"    , False                  ),
#        ("pricm", "full"   , False                  ),
#        ("icm"  , "full"   , False                  ),
############
#        ("pricm", "overlap", True                   ),
        ("icm"  , "overlap", False                  ),
        ("pricm", "full", False                   ),
    ]
    for model, hp_opt, gen_new_hist in systems:
        local = fullreplace(script, "$MODEL$", model)
        local = fullreplace(local, "$IS$", infosources)
        local = fullreplace(local, "$FOLDER$", folder)
        local = fullreplace(local, "$COST$", cost)
        local = fullreplace(local, "$HP_OPT$", hp_opt)
        local = fullreplace(local, "$GNH$", gen_new_hist)
        local = fullreplace(local, "$N_HISTORY$", N_History)

        if cost:       
            fname = "%d_%s_%s_TC.py" % (infosources, model, hp_opt)
        else:
            fname = "%d_%s_%s.py" % (infosources, model, hp_opt)
    
        fptr = open(fname, 'w')
        fptr.write(local)
        fptr.close()
    
        pysub(
            fname,
            queue=QUEUE, nprocs="4",
            use_mpi=False, walltime=WALLTIME, modules=["pal"],
            unique_name=False, jobarray=RANGE
        )
        print("Submitting %s" % fname)
        time.sleep(0.2)


def run_ei(infosources, folder, cost, RANGE=(0, 5), QUEUE=None, WALLTIME="1-00:00:00"):
    '''
    Submit a number of EI jobs for HOIP optimization.

    **Parameters**

        infosources: *int*
            The information source to use.  In the case of EI, this is only
            the 0th, as there is no need to look at the cheaper alternatives
            (being that this is not a miso approach).
                0 - IS_N1R3 vs IS_N1R2
                1 - IS_N3R2 vs IS_N1R2
                2 - IS_N5R2 vs IS_N3R2
                3 - IS_N5R2 vs IS_N3R2 vs IS_N1R2
        folder: *str*
            The name for the output folder.
        cost: *bool*
            The cost of the information source.  Mainly, True if the true cost
            was used in the corresponding run_ei file, and False if it is an
            approximation.  This will only change the naming of files so we
            keep track of this.

    **Returns**

        None
    '''

    # Recall that infosources 2 and 3 are the same for EI
    if infosources == 3:
        return None

    script = '''
import sys
from run_ei import run
run(int(sys.argv[-1]), folder="$FOLDER$", infosources=$IS$, exact_cost=$ECOST$)
'''
    model = "ei"
    if cost:       
        fname = "%d_%s_TC.py" % (infosources, model)
    else:
        fname = "%d_%s.py" % (infosources, model)

    local = fullreplace(script, "$FOLDER$", folder)
    local = fullreplace(local, "$IS$", infosources)
    local = fullreplace(local, "$ECOST$", cost)
    
    fptr = open(fname, 'w')
    fptr.write(local)
    fptr.close()
    
    pysub(fname, queue=QUEUE, nprocs="4", use_mpi=False, walltime=WALLTIME, priority=20, modules=["pal"], jobarray=RANGE)
    print("Submitting %s" % fname)
    time.sleep(0.1)


if __name__ == "__main__":
    # Parameters for general job submission
    RANGE = (0, 200)
    QUEUE = "shared"
    WALLTIME = "1-00:00:00"
    stuff = [
        (0, "N1R3_N1R2"),
        (1, "N3R2_N1R2"),
        (2, "N5R2_N3R2"),
    ]
    costs = [True]

    for cost in costs:
        for IS, F in stuff:
            if cost:
                F += "_TC"
            if not os.path.exists(F):
                os.mkdir(F)
            run_pricm(IS, F, cost, RANGE=RANGE, QUEUE=QUEUE, WALLTIME=WALLTIME)
   
    for cost in costs:
        for IS, F in stuff:
            if cost:
                F += "_TC"
            run_ei(IS, F, cost, RANGE=RANGE, QUEUE=QUEUE, WALLTIME=WALLTIME)

