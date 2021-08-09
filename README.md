# Predicting Genomic Resolution of BSA
This repository stores the codes that run SLiM simulations and plot the graphs for the paper that predicts the genomic resolution of Bulk Segregant Analysis (BSA).

`predict_bsa_resolution.py` is the python script that can be used to calculate the expected genomic resolution of a BSA experiment based on the analytical solutions derived in our paper. No additional python package is needed to run the script. The python script takes five arguments in order: the estimated effective population size _Ne_, the length of the experiment _t_, the average recombination rate _r_, the sample size for genome sequencing _s_ and the analytical model used for the calculation _approx_ or _recursion_. An example is shown below:


`python predict_bsa_resolution.py 100 10 1e-8 20 approx` uses the approximation model to calculate the expected genomic resolution of a BSA experiment with _Ne_=100, running from F0 to F10, an estimated recombination probablity of 1e-8 and sampling a total of 20 diploid individuals for genome sequencing. The output of running the script would be:

`Expected Genomic Resolution of BSA Experiment with Ne=100, Gen=10, R=1.000e-08, s=20 using approximate model is: 7.149e+05 bp`.
