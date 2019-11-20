# MISO_Paper

These results were done using anaconda/python-2.7.  To regenerate plots from
parsed output, you may simply run the following codes:

    - hoip_post_process.py
    - rosenbrock_post_process.py

To regenerate tables, you can run:

    - table_post_process.py

ALl raw output is stored in subdirectories of *hoips*, *rosenbrock*, and *CO*.
Parsing the output takes some time, so pre-parsed output has been tarballed in
the *parsed_output.tar.gz* file.  The python codes above will automatically
grab said output from the tarball and plot them accordingly.
