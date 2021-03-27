import msprime, pyslim
import numpy as np
import os
from collections import OrderedDict, defaultdict
import time
import pickle
import multiprocessing as mp
from subprocess import Popen, PIPE, check_output, run
import sys


def output_SLiM_NV_BSA(slim_file, Cap_N, Ne, recom_rate, chr_len, gen, output_tree_file):
    """
        output_SLiM_NV_BSA: output SLiM model for BSA

        All the arguments taken are strings (even for the numerical parameters), to conform with the numerical format required by SLiM
        Args:
            slim_file (string): SLiM file for simulating BSA
            Cap_N (string): string of total individuals in BSA experiment (carrying capacity)
            Ne (string): string of effective population size in BSA experiemnt (number of individuals allowed to reproduce per generation)
            recom_rate (string): string of recombination rate
            chr_len (string): string of length of chromosome
            gen (string): string of number of generation in BSA experiment
            output_tree_file (string): file name of Tree-Sequence output file by SLiM model recording simulation results

        Returns:
            slim_file (string): file name of SLiM script

    """

    f_in = open(slim_file, "w")
    ## initialize
    f_in.write('// Cannings Model simulation for BSA\n')
    f_in.write('initialize() {\n')
    f_in.write('\tinitializeSLiMOptions(keepPedigrees = T);\n')
    f_in.write('\tinitializeTreeSeq();\n')
    f_in.write('\tinitializeSex("A");\n')
    f_in.write('\tinitializeMutationRate(0);\n')
    f_in.write('\tinitializeMutationType("m1", 0.5, "f", 0.0);\n')
    f_in.write('\tinitializeGenomicElementType("g1", m1, 1.0);\n')
    f_in.write('\tinitializeGenomicElement(g1, 0, {});\n'.format(chr_len))
    f_in.write("\tinitializeRecombinationRate({});".format(recom_rate)+"\n}\n\n")

    ## add population
    f_in.write('1 late() {\n')
    f_in.write("\tsim.addSubpop(\"p1\", {});".format(Cap_N)+"\n}\n\n")

    ## cannings model for each generation
    f_in.write('late() {\n')
    f_in.write('\t// after reproduction, everybody lives initially\n')
    f_in.write('\tsim.subpopulations.individuals.tag = 0;\n')
    f_in.write('\t// afterwards, only NE samples survive and eligible for reproduction\n')
    f_in.write('\tsamples = sample(sim.subpopulations.individuals, {});\n'.format(Ne))
    f_in.write('\t// make sure the samples are bisexual for eligible reproduction\n')
    f_in.write("\twhile(sum(samples.sex == 'M') == {} | sum(samples.sex == 'F') == {})".format(Ne, Ne)+'{\n')
    f_in.write('\t\tsamples = sample(sim.subpopulations.individuals, {});\n'.format(Ne)+'\t}\n\n')
    f_in.write('\tsamples.tag = 1;\n}\n\n')
    
    f_in.write('fitness(NULL) {\n')
    f_in.write('\t// individuals tagged for death die here\n')
    f_in.write('\tif (individual.tag == 1)\n')
    f_in.write('\t\treturn 1.0;\n')
    f_in.write('\telse\n')
    f_in.write('\t\treturn 0.0;\n}\n\n')

    # SLiM output tree-seq file
    f_in.write('{} late()'.format(gen)+' {\n')
    f_in.write("\tsim.treeSeqOutput('{}');".format(output_tree_file)+'\n}')

    f_in.close()
    return slim_file


def run_slim(slim_file, remove=True):    
    """
        run_slim: Run a single SLiM models in the command line
        
        Args:
            slim_file (string): the SLiM file name for simulation
            remove (boolean): determine whether to remove SLiM file after running it

        Returns:
            0
        
    """
    process = run(['slim', slim_file], stdout=PIPE, shell=False)
    if remove: os.remove(slim_file)
    return 0


"""
    Run multiple SLiM models in parallel
"""
def run_slims(slim_files, pool):
    """
        run_slims: Run SLiM simulations in parallel
    
        Args:
            slim_files (list): the list of SLiM files for simulation
            pool (object): pool object from multiprocessing package for running SLiM models in parallel

        Returns:
            0
    """
    pool.map(run_slim, slim_files)
    return 0


def BSA_genomic_resolution(tree_file, causal_mut, sample_size, window_type):
    """
        BSA_genomic_resolution: get the genomic resolution of a BSA run
        
        Args:
            tree_file (string): the Tree-sequence output of SLiM model
            causal_mut (int): the position of causal mutation on the chromosome
            sample_size (int): the number of (DIPLOID) individuals selected from the population
            window_type (string): specifies the type of window for BSA genomic resolution

        Returns:
            resolution (float): genomic resolution of a BSA run
        
    """
    ### the window_type can only be open or closed
    assert (window_type == 'open' or window_type == 'closed'), "Window type could only be \'open\' or \'closed\'"
    
    # read in the tree-sequence output file
    ts = pyslim.load(tree_file)

    # get the number of haplotypes in the WF population of last generation
    all_samples = list(ts.first().samples())
    Ne = len(all_samples)//2

    # set the haplotypes of the population in F0
    genome_tag_dict = OrderedDict((x, y) for x, y in zip(range(Ne*2), [0] * Ne + [1] * Ne))

    # randomly select the samples
    sample_inds = list(np.random.choice(list(filter(lambda x : x%2==0, all_samples)), sample_size, replace=False))
    samples = sample_inds + [x + 1 for x in sample_inds]

    # trace back the genealogy of lineages to locate the nearest recombination point to causal mutation 
    recombination_points = []
    sample_parent_dict = {}
    for tree in ts.trees():
        if tree.interval == ts.first().interval:
            for sample in samples:
                u = sample
                while tree.parent(u) != msprime.NULL_NODE:
                    u = tree.parent(u)
                sample_parent_dict[sample] = [genome_tag_dict[u]]
        else:
            for sample in samples:
                u = sample
                while tree.parent(u) != msprime.NULL_NODE:
                    u = tree.parent(u)
                if (sample_parent_dict[sample][-1] != genome_tag_dict[u]):
                    recombination_points.append(tree.interval[0])
                sample_parent_dict[sample].append(genome_tag_dict[u])

    # trace back the genealogy of lineages to locate the nearest recombination point to causal mutation 
    if (window_type == 'open'):
        # return the open-window resolution of the random samples
        if (len(list(filter(lambda x:x>=causal_mut, recombination_points))) > 0):
            return min(list(filter(lambda x:x>=causal_mut, recombination_points))) - causal_mut
        else:
            return np.inf
    elif (window_type == 'closed'):
        # return the closed-window resolution of the random samples
        if (len(list(filter(lambda x:x>=causal_mut, recombination_points))) > 0 and len(list(filter(lambda x:x<=causal_mut, recombination_points))) > 0):
            return min(list(filter(lambda x:x>=causal_mut, recombination_points))) - max(list(filter(lambda x:x<=causal_mut, recombination_points)))
        else:
            return np.inf


def main():
    """
        Define some simulation parameters
    """
    Cap_N = '2000'
    Ne = sys.argv[1]
    # Remember: T = 11 is for generation starting index as 1 in SLiM model
    # This is equivalent to t = 10 for zero-index in our paper's analytical model
    T = sys.argv[2]
    R = '1e-8'
    chr_len = '1e8-1'
    mut_pos = '1e2'
    window_type = 'open'
    # number of simulations
    num_sim = 5000

    slim_file = 'SLiM_NV_BSA'
    
    # multiprocessing pool to run slim simulations
    count = mp.cpu_count()
    pool = mp.Pool(processes=count)
    print("Pool started for Population Size {}, Experimental length {}".format(Ne, T))

    # create the SLiM models from template
    slim_sims = [pool.apply_async(output_SLiM_NV_BSA, args=(slim_file+'{}.slim'.format(_), Cap_N, Ne, R, chr_len, T, slim_file+'{}.trees'.format(_))) for _ in range(num_sim)]
    slim_files = [s.get() for s in slim_sims]
    print("Number of SLiM files:", len(slim_files))
    
    # run SLiM models in parallel
    SLIM = run_slims(slim_files, pool)
    print("SLiM Files run and removed")
    tree_files = list(filter(lambda x : slim_file in x and x.endswith('.trees'),os.listdir()))
    print("Number of Tree files:", len(tree_files))
    
    # find the genomic resolution for different sample sizes
    S_s = [2**x for x in range(1,11)]
    resolution_all = OrderedDict([(S,[]) for S in S_s])
    for S in S_s:
        random_trees = np.random.choice(tree_files, num_sim, replace=False)
        results_slim = [pool.apply_async(BSA_genomic_resolution, args=(pop_tree_file, int(float(mut_pos)), S, window_type, )) for pop_tree_file in random_trees]
        resolution_slim = [s.get() for s in results_slim]
        resolution_all[S] += resolution_slim
        print('{} Sample Size finished'.format(S), len(resolution_all[S]))
    
    # remove tree files
    remove_trees = pool.map(os.remove, tree_files)

    # output results into pickle file for plotting
    with open('BSA_Open-win_Res_100MbChrom_Ne{}_Gen{}_{}cM_{}kSims.pckl'.format(Ne, T, float(R)*1e8, num_sim//1000), 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(resolution_all, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
