import msprime, pyslim
import numpy as np
import os
from collections import OrderedDict
import time
import pickle
from subprocess import PIPE, run
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

        
def BSA_stats_tree_seq(tree_file, mut_rate, sample_size, causal_mut, coverage=None, chr_len=int(1e7)):
    """
        BSA_stats_tree_seq: output summary statistics for BSA analysis

        Args:
            tree_file (string): tree-seq file from simulating BSA output
            mut_rate (float): the rate of snps on the chromosome (e.g. 1e-2 means 1 snp per 100bp on average)
            sample_size (int): the number of diploid individuals sampled from each phenotype
            causal_mut (int): location of causal mutation
            win_size (int): window size for ancestry diff calculation
            coverage (int): the coverage for pool sequencing

        Returns:
            all_snp_g_dict (dict): G statistic per snp
            ancestry_diff_dict (dict): A_d statistic per window
    """
    # read tree file
    ts = pyslim.load(tree_file)
    tree = ts.first()
    all_samples = list(tree.samples())
    genome_pop_size = len(all_samples)

    # Create the phenotype tag for each ancestral genome
    genome_tag_dict = OrderedDict((x, y) for x, y in zip(range(genome_pop_size), [1] * (genome_pop_size // 2) + [0] * (genome_pop_size // 2)))

    # create the mutations to lay on the genomes
    snp_pos_list = list(np.random.choice(range(chr_len), int(chr_len*mut_rate+np.random.normal(1/mut_rate, 10)*np.random.choice([1,-1])), replace=False))
    snp_pos_list.append(causal_mut)

    # select for the resistant vs susceptible samples
    # sampling process
    resistance_genomes = []
    susceptible_genomes = []

    # locate the tree interval for causal_mut
    # select all the resistant/susceptible genomes
    for t in ts.trees():
        if (t.interval[0] <= causal_mut <= t.interval[1]):
            tree_with_mut = t
            for root in tree_with_mut.roots:
                if (genome_tag_dict[root] == 1):
                    resistance_genomes += tree_with_mut.samples(root)
                elif (genome_tag_dict[root] == 0):
                    susceptible_genomes += tree_with_mut.samples(root)
            break

    # choose the individuals' genomes
    homo_res_ind_genome1s = list(filter(lambda x : x%2==0 and (x+1) in resistance_genomes, resistance_genomes))
    if (len(homo_res_ind_genome1s) < sample_size):
        homo_res_ind_sample = [x for x in np.random.choice(homo_res_ind_genome1s, sample_size, replace=True)]
    else:
        homo_res_ind_sample = [x for x in np.random.choice(homo_res_ind_genome1s, sample_size, replace=False)]
    homo_res_ind_genomes = homo_res_ind_sample + [x + 1 for x in homo_res_ind_sample]

    homo_sus_ind_genome1s = list(filter(lambda x : x%2==0 and (x+1) in susceptible_genomes, susceptible_genomes))
    if (len(homo_sus_ind_genome1s) < sample_size):
        homo_sus_ind_sample = [x for x in np.random.choice(homo_sus_ind_genome1s, sample_size, replace=True)]
    else:
        homo_sus_ind_sample = [x for x in np.random.choice(homo_sus_ind_genome1s, sample_size, replace=False)]
    homo_sus_ind_genomes = homo_sus_ind_sample + [x + 1 for x in homo_sus_ind_sample]

    # calculate Gprime statistics
    tree_along_chr = OrderedDict()
    all_snp_g_dict = OrderedDict()
    start_time = time.time()

    ## with short-read sequencing
    if (coverage != None):
        for tree in filter(lambda tree: tree.interval[0] >= causal_mut - 2e6 and tree.interval[1] <= causal_mut + 2e6, ts.trees()): 
            snp_in_interval = list(filter(lambda x: tree.interval[0] <= x <= tree.interval[1], snp_pos_list))
            ## each snp is sequenced independently by short-read sequencing
            for snp in snp_in_interval:
                n3 = 0
                n4 = 0
                sequenced_samples_res = np.random.choice(homo_res_ind_genomes, coverage, replace=True)
                for sample in sequenced_samples_res:
                    u = sample
                    while tree.parent(u) != msprime.NULL_NODE:
                        u = tree.parent(u)
                    n3 += genome_tag_dict[u]
                if (n3 == coverage): n3 -= 1e-10

                sequenced_samples_sus = np.random.choice(homo_sus_ind_genomes, coverage, replace=True)
                for sample in sequenced_samples_sus:
                    u = sample
                    while tree.parent(u) != msprime.NULL_NODE:
                        u = tree.parent(u)
                    n4 += genome_tag_dict[u]
                if (n4 == 0): n4 += 1e-10
                n1 = coverage - n3
                n2 = coverage - n4

                exp_n1 = (n1+n2)*(n1+n3) / (n1+n2+n3+n4)
                exp_n2 = (n1+n2)*(n2+n4) / (n1+n2+n3+n4)
                exp_n3 = (n1+n3)*(n4+n3) / (n1+n2+n3+n4)
                exp_n4 = (n4+n2)*(n4+n3) / (n1+n2+n3+n4)

                obs_n = [n1, n2, n3, n4]
                exp_n = [exp_n1, exp_n2, exp_n3, exp_n4]

                G = 2 * sum([obs_n[i]*np.log(obs_n[i]/exp_n[i]) for i in range(len(obs_n))])
                tree_along_chr[tree.interval] = G
                all_snp_g_dict[snp] = G
    else:
        for tree in filter(lambda tree: tree.interval[0] >= causal_mut - 2e6 and tree.interval[1] <= causal_mut + 2e6, ts.trees()):
            n3 = 0
            n4 = 0
            
            for sample in homo_res_ind_genomes:
                u = sample
                while tree.parent(u) != msprime.NULL_NODE:
                    u = tree.parent(u)
                n3 += genome_tag_dict[u]
            if (n3 == sample_size * 2): n3 -= 1e-10

            for sample in homo_sus_ind_genomes:
                u = sample
                while tree.parent(u) != msprime.NULL_NODE:
                    u = tree.parent(u)
                n4 += genome_tag_dict[u]

            if (n4 == 0): n4 += 1e-10
            n1 = sample_size * 2 - n3
            n2 = sample_size * 2 - n4
            
            exp_n1 = (n1+n2)*(n1+n3) / (n1+n2+n3+n4)
            exp_n2 = (n1+n2)*(n2+n4) / (n1+n2+n3+n4)
            exp_n3 = (n1+n3)*(n4+n3) / (n1+n2+n3+n4)
            exp_n4 = (n4+n2)*(n4+n3) / (n1+n2+n3+n4)

            obs_n = [n1, n2, n3, n4]
            exp_n = [exp_n1, exp_n2, exp_n3, exp_n4]

            G = 2 * sum([obs_n[i]*np.log(obs_n[i]/exp_n[i]) for i in range(len(obs_n))])

            tree_along_chr[tree.interval] = G
            snp_in_interval = list(filter(lambda x: tree.interval[0] <= x <= tree.interval[1], snp_pos_list))
            all_snp_g_dict.update(dict.fromkeys(snp_in_interval, G))

    print("SNP G stats finished for sample size {} coverage {}!".format(sample_size, coverage), "--- %s seconds ---" % (time.time() - start_time))
    return all_snp_g_dict


"""
    Smoothing functions for G statistic
"""
def tricube_smooth_G(all_snp_g_dict, win_size=10000, chr_len=10**7):
    """
        tricube_smooth_G(all_snp_g_dict, win_size=10000, chr_len=10**7) smooths G statistic of each SNP using a tricube-weighted smoothing function
    """
    tricube_weighted_Gprime_dict = OrderedDict()
    min_ = min(all_snp_g_dict.keys())
    max_ = max(all_snp_g_dict.keys())
    all_snp_list = sorted(list(filter(lambda x : min_ < x < max_, all_snp_g_dict.keys())))
    for focal_snp in all_snp_list:
        window = (focal_snp - win_size//2, focal_snp + win_size//2)
        snp_in_window = list(filter(lambda x : window[0] <= x <= window[1], all_snp_list))
        weight_kjs_ = [(1-(abs(snp-focal_snp)/(win_size // 2))**3)**3 for snp in snp_in_window]
        sum_kjs = sum(weight_kjs_)
        weight_kjs_dict = OrderedDict([(snp, k_j / sum_kjs) for snp, k_j in zip(snp_in_window, weight_kjs_)])
        snp_Gprime = sum([weight_kjs_dict[snp] * all_snp_g_dict[snp] for snp in snp_in_window])

        for snp in snp_in_window:
            tricube_weighted_Gprime_dict[snp] = snp_Gprime
#             print(snp, snp_Gprime)

    # print("SNP G\' finished!", "--- %s seconds ---" % (time.time() - start_time))
    peak_snps = [x[0] for x in list(filter(lambda x: abs(x[1]-max(tricube_weighted_Gprime_dict.values()))<=0.01, tricube_weighted_Gprime_dict.items()))]

#     max_window = list(filter(lambda x: np.isclose(x[1],max(tricube_weighted_Gprime_dict.values())), tricube_weighted_Gprime_dict.items()))
#     max_window_bounds = (max_window[0][0], max_window[-1][0])
    
    return tricube_weighted_Gprime_dict


def main():
    """
        Define some simulation parameters
    """
    Cap_N = '2000'
    MUT_RATE = 1e-3
    Ne = 100
    # Remember: T = 11 is for generation starting index as 1 in SLiM model
    # This is equivalent to t = 10 for zero-index in our paper's analytical model
    T = 11
    R = '1e-8'
    chr_len = '1e7-1'
    mut_pos = 5e6
    slim_file = 'SLiM_NV_BSA.slim'
    tree_file = 'SLiM_NV_BSA.trees'
    
    # create the SLiM file
    slim_file = output_SLiM_NV_BSA(slim_file, Cap_N, Ne, R, chr_len, T, tree_file)
    # run the SLiM file
    run_slim_file = run_slim(slim_file)
    
    # get the summary statistic
    all_snp_g_dict_s1000 = BSA_stats_tree_seq(tree_file, MUT_RATE, 500, mut_pos, coverage=None)
    all_snp_gprime_dict_s1000 = tricube_smooth_G(all_snp_g_dict_s1000)
    with open('BSA_Resolution_vs_Statistics_s1000_noCov.pckl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump((all_snp_g_dict_s1000, all_snp_gprime_dict_s1000), f, pickle.HIGHEST_PROTOCOL)
    
    all_snp_g_dict_s1000_c100 = BSA_stats_tree_seq(tree_file, MUT_RATE, 500, mut_pos, coverage=100)
    all_snp_gprime_dict_s1000_c100 = tricube_smooth_G(all_snp_g_dict_s1000_c100)
    with open('BSA_Resolution_vs_Statistics_s1000_cov100.pckl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump((all_snp_g_dict_s1000_c100,all_snp_gprime_dict_s1000_c100), f, pickle.HIGHEST_PROTOCOL)
    
    all_snp_g_dict_s100 = BSA_stats_tree_seq(tree_file, MUT_RATE, 50, mut_pos, coverage=None)
    all_snp_gprime_dict_s100 = tricube_smooth_G(all_snp_g_dict_s100)
    with open('BSA_Resolution_vs_Statistics_s100_noCov.pckl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump((all_snp_g_dict_s100, all_snp_gprime_dict_s100), f, pickle.HIGHEST_PROTOCOL)

    print("BSA Summary Statistics output successfully!")
    
    
if __name__ == '__main__':
    main()
