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


def run_pricm(folder, RANGE=(0, 5), QUEUE=None, WALLTIME="1-00:00:00", sample_domain=1000):
    '''
    Submit a number of miso jobs for HOIP optimization.
    
    **Parameters**

        folder: *str*
            The name for the output folder.

    **Returns**

        None
    '''
    script = '''
import sys
from run_pricm import run
run(
    int(sys.argv[-1]),
    "$MODEL$",
    folder="$FOLDER$",
    hp_opt="$HP_OPT$",
    sample_domain=$SAMPLE_DOMAIN$
)
'''.strip()
    systems = [
#       Model   ,  HP_Opt 
#        ("pricm", "is0"    ),
#        ("icm"  , "is0"    ),
#        ("miso" , "is0"    ),
#        ("pricm", "full"   ),
#        ("icm" , "full"   ),
#        ("miso" , "full"    ),
        ("pricm", "full"),
#        ("pricm", "overlap"),
#        ("icm" , "overlap"),
#        ("miso" , "overlap"),
    ]
    for model, hp_opt in systems:
        local = fullreplace(script, "$MODEL$", model)
        local = fullreplace(local, "$FOLDER$", folder)
        local = fullreplace(local, "$HP_OPT$", hp_opt)
        local = fullreplace(local, "$SAMPLE_DOMAIN$", sample_domain)

        fname = "%s_%s.py" % (model, hp_opt)
    
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


def run_ei(folder, RANGE=(0, 5), QUEUE=None, WALLTIME="1-00:00:00", sample_domain=1000):
    '''
    Submit a number of EI jobs for HOIP optimization.

    **Parameters**

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
from run_ei import run
run(int(sys.argv[-1]), folder="$FOLDER$", sample_domain=$SAMPLE_DOMAIN$)
'''
    model = "ei"
    local = fullreplace(script, "$FOLDER$", folder)
    local = fullreplace(local, "$SAMPLE_DOMAIN$", sample_domain)

    fname = "%s.py" % (model)
    
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
    folder = "CO"
    sample_domain = 500

    if not os.path.exists(folder):
        os.mkdir(folder)
    run_pricm(folder, RANGE=RANGE, QUEUE=QUEUE, WALLTIME=WALLTIME, sample_domain=sample_domain)
#    run_ei(folder, RANGE=RANGE, QUEUE=QUEUE, WALLTIME=WALLTIME, sample_domain=sample_domain)

