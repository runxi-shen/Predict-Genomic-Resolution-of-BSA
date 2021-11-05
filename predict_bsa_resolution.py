"""
    Author: Runxi Shen
    This script helps the researcher to calculate the expected genomic resolution
    of a predesigned BSA experiment using the analytical results presented in the
    paper Predicting Genomic Resolution of BSA.
    
    The script takes in five required parameters:
    Ne (integer): the estimated effective population size
    T (integer): the number of generations in BSA experiment 
    R (float): recombination rate, R = 1e-8 is equivalent to 1cM/Mb
    S (integer): sample size
    model (str): the analytical equation used to calculate genomic resolution, 'approx' or 'recursion'
"""

import sys
import math

"""
    Calculate the BSA open-window resolution in finite populations
"""
def calc_BSA_open_win_res_fin(Ne, r, s, t):
    D = 1/(2*Ne*r*math.log((2*s*(math.exp(t/(4*Ne))-1)+1)))
    return D


"""    
    Calculate the number of lineages expectation
"""
def calc_nt_recursion_diploid(Ne, t, s):
    if (t == 0):
        return 2*s
    elif (t == 1):
        return 2*Ne - 2*Ne*(1-1/(2*Ne))**(2*s)
    else:
        return 2*Ne - 2*Ne*(1-1/(2*Ne))**(calc_nt_recursion_diploid(Ne, t-1, s))
    
    
"""
    Calculate the BSA open-window resolution in finite populations with recursion
"""
def calc_BSA_open_win_res_recursion(Ne, r, s, t):
    tot_tree_len_bef_3 = sum([calc_nt_recursion_diploid(Ne, t, s) for t in range(t-2)])
    samples_at_3 = calc_nt_recursion_diploid(Ne, t-2, s)
    D = 1/(r*samples_at_3+r/2*tot_tree_len_bef_3)
    return D


def main():
    """
        Get the input experimental parameters
    """
    ## the estimated effective population size Ne (int)
    Ne = int(sys.argv[1]) 
    ## the number of generations in BSA experiment (int): 
    ## starting with zero-index F_0 as described in paper
    T = int(sys.argv[2])
    ## recombination rate (float): defined as the probability of a crossing-over 
    ## event occurring between any two adjacent bases per genome per generation
    ## e.g. R = 1e-8 is equivalent to 1cM/Mb
    R = float(sys.argv[3])
    ## sample size (int): the number of total diploid individuals 
    ## in two bulks sampled for sequencing in the BSA experiment
    S = int(sys.argv[4])
    ## model (str): specify which model is used to calculate the resolution ("integration" or "recursion")
    model = sys.argv[5]
    
    if (model == 'integration'):
        print("Expected Genomic Resolution of BSA Experiment with Ne={}, Gen={}, R={:.3e}, s={} using integration model is: {:.3e} bp".format(Ne, T, R, S, calc_BSA_open_win_res_fin(Ne, R, S, T)))
    elif (model == 'recursion'):
        print("Expected Genomic Resolution of BSA Experiment with Ne={}, Gen={}, R={:.3e}, s={} using recursion model is: {:.3e} bp".format(Ne, T, R, S, calc_BSA_open_win_res_recursion(Ne, R, S, T)))
    else:
        print("No model is selected. Please specify 'integration' or 'recursion' for the model parameter.")
        
        
if __name__ == '__main__':
    main()
