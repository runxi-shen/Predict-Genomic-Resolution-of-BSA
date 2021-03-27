import msprime, pyslim
import numpy as np
import os
from collections import OrderedDict, defaultdict
import time
import pickle
import multiprocessing as mp
from subprocess import Popen, PIPE, check_output, run
import sys

"""
    SLiM Simulation functions for HC
"""
def output_SLiM_NV_HC(slim_file, Ne, recom_rate, chr_len, mut_pos, gen, output_tree_file):
    """
        output_SLiM_NV_HC: output SLiM model for BSA

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
    f_in.write("\tsim.addSubpop(\"p1\", {});".format(Ne)+"\n")
    f_in.write("\tfemales = p1.sampleIndividuals({}, sex='F');\n".format(int(Ne)//2))
    f_in.write('\tfemales.genomes.addNewDrawnMutation(m1, {});'.format(mut_pos)+'\n}\n\n')

    f_in.write('3:{} late()'.format(int(gen)-1)+'{\n')    
    f_in.write('\tfor (individual in p1.individuals) {\n')
    f_in.write('\t\tif (individual.genome1.containsMarkerMutation(m1, {}) == individual.genome2.containsMarkerMutation(m1, {}))\n'.format(mut_pos, mut_pos))
    f_in.write('\t\t\tindividual.fitnessScaling = 0;\n')
    f_in.write('\t\telse\n')
    f_in.write('\t\t\tindividual.fitnessScaling = 1;\n\t}\n}\n\n')

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


def HC_genomic_resolution_treeseq(tree_file, causal_mut, sample_size, window_type):
    """
        HC_genomic_resolution_treeseq: get the open-window resolution of a given sample size
        
        Args:
            tree_file (string): the Tree-seq population file
            Ne (int): the effective population size for IM
            sample_size (int): the number of (DIPLOID) individuals selected from the population
            window_type (string): the type of genomic resolution (open or closed)
            
        Returns:
            resolution (float): genomic resolution of HC experiment
        
    """
    assert (window_type == 'open' or window_type == 'closed'), "Window type could only be \'open\' or \'closed\'"
    
    # read in the tree-sequence output file
    ts = pyslim.load(tree_file)

    # select the samples from population
    all_samples = sorted(list(ts.first().samples()))
    Ne = len(all_samples) // 2

    # set the haplotypes of the population in the 1st generation
    genome_tag_dict = OrderedDict((x, y) for x, y in zip(range(Ne*2), [0]*Ne + [1]*Ne))

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
            return (min(list(filter(lambda x:x>=causal_mut, recombination_points))) - causal_mut)
        else:
            return (np.inf, len(set(recombination_points)))
    elif (window_type == 'closed'):
        # return the closed-window resolution of the random samples
        if (len(list(filter(lambda x:x>=causal_mut, recombination_points))) > 0 and len(list(filter(lambda x:x<=causal_mut, recombination_points))) > 0):
            return min(list(filter(lambda x:x>=causal_mut, recombination_points))) - max(list(filter(lambda x:x<=causal_mut, recombination_points)))
        else:
            return np.inf


def main():
    """
        Define some constants
    """
    Ne = '100'
    # T = 11 is for generation starting index as 1 in SLiM model
    # This is equivalent to t = 10 for zero-index in our paper's analytical model
    T = '11'
    R = '1e-8'
    chr_len = '1e8-1' 
    mut_pos = '1e4'
    window_type = 'open'
    # number of simulations per each run
    num_sim = 5000

    slim_file = 'SLiM_NV_HC'

    count = mp.cpu_count()
    pool = mp.Pool(processes=count//2)
    print("Pool started")

    slim_sims = [pool.apply_async(output_SLiM_NV_HC, args=(slim_file+'{}.slim'.format(_), Ne, R, chr_len, mut_pos, T, slim_file+'{}.trees'.format(_))) for _ in range(num_sim)]
    slim_files = [s.get() for s in slim_sims]
    print("Number of SLiM files:", len(slim_files))
    
    SLIM = run_slims(slim_files, pool)
    print("SLiM Files run and removed")
    tree_files = sorted(list(filter(lambda x : slim_file in x and x.endswith('.trees'),os.listdir())))
    print("Number of Tree files:", len(tree_files))
    
    S_s = [2**x for x in range(1,7)] 
    resolution_all = OrderedDict([(S,[]) for S in S_s])

    for S in S_s:
        random_trees = np.random.choice(tree_files, num_sim, replace=False)
        results_slim = [pool.apply_async(HC_genomic_resolution_treeseq, args=(pop_tree_file, int(float(mut_pos)), S, window_type, )) for pop_tree_file in random_trees]
        resolution_slim = [s.get() for s in results_slim]
        resolution_all[S] += resolution_slim
        print('{} Sample Size finished'.format(S), len(resolution_all[S]))

    remove_trees = pool.map(os.remove, tree_files)
    print("Number of Tree files:", len(list(filter(lambda x : 'Tree_Seq_Popsize_'+str(Ne)+'_' in x and x.endswith('_gen'+str(T+1)+'.trees'),os.listdir()))))
    f_out = 'HC_Open-win_Res_100MbChrom_Ne{}_Gen{}_{}cM_{}kSims.pckl'.format(Ne, T, float(R)*1e8, num_sim//1000)
    with open(f_out, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(resolution_all, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
