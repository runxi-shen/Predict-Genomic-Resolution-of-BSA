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

    # print(genome_tag_dict)
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
#     selected_snp_g_dict = OrderedDict()
    tree_along_chr = OrderedDict()
    all_snp_g_dict = OrderedDict()
    all_snp_g_dict_w_cov = OrderedDict()
    start_time = time.time()
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
        
        ## with short-read sequencing
        if (coverage != None):
            snp_in_interval = list(filter(lambda x: tree.interval[0] <= x <= tree.interval[1], snp_pos_list))
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
                # all_snp_g_dict_w_cov.update(dict.fromkeys(snp, G))
                all_snp_g_dict_w_cov[snp] = G

    print("SNP G stats finished!", "--- %s seconds ---" % (time.time() - start_time))
    return(all_snp_g_dict, all_snp_g_dict_w_cov)


"""
    Smoothing functions for G and A_d
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
#     print(peak_snps)
#     print(np.min(peak_snps), np.max(peak_snps))
    resolution = np.max(peak_snps) - np.min(peak_snps)
    return(tricube_weighted_Gprime_dict, resolution)


# def smooth_window_ad(ancestry_diff_dict, num_flanking=4):
#     """
#         smooth_window_ad(all_snp_g_dict, win_size=10000, chr_len=10**7) smooths G statistic of each SNP using a tricube-weighted smoothing function
#     """
#     window_keys = list(ancestry_diff_dict.keys())
#     window_values = list(ancestry_diff_dict.values())
    
#     # smoothing weights
#     weights = list(range(1,num_flanking+2))+list(range(num_flanking,0,-1))
#     sum_weights = float(sum(weights))

#     assert num_flanking <= len(window_values)-num_flanking-1
#     for win_idx in range(num_flanking, len(window_values)-num_flanking-1):
#         windows_select = [window_values[i] for i in range(win_idx-num_flanking, win_idx+num_flanking+1)]
#         weighted_window = [val*weight for val, weight in zip(windows_select, weights)]
#         window_values[win_idx] = float(sum(weighted_window)) / sum_weights
    
#     smoothed_dict = OrderedDict([(key, val) for key, val in zip(window_keys, window_values)])
#     peak_snps = sorted([x[0] for x in list(filter(lambda x: abs(x[1]-max(smoothed_dict.values()))<=0.01, smoothed_dict.items()))])
# #     print(peak_snps[0][0], peak_snps[-1][-1])
#     resolution = peak_snps[-1][-1] - peak_snps[0][0]
#     return(smoothed_dict, resolution)


def main():
    """
        Define some simulation parameters
    """
    Cap_N = '2500'
    MUT_RATE = 1e-3
    Ne = 100
    # Remember: T = 11 is for generation starting index as 1 in SLiM model
    # This is equivalent to t = 10 for zero-index in our paper's analytical model
    T = 11
    R = '1e-8'
    chr_len = '1e7-1'
    mut_pos = 5e6
#     window_type = 'open'
    slim_file = 'SLiM_NV_BSA.slim'
    tree_file = 'SLiM_NV_BSA.trees'
    
    # create the SLiM file
    slim_file = output_SLiM_NV_BSA(slim_file, Cap_N, Ne, R, chr_len, T, tree_file)
    # run the SLiM file
    run_slim_file = run_slim(slim_file)
    
    # get the summary statistics
    sample_size = 1000
    coverage = 50
    
#     all_snp_g_dict, all_snp_g_dict_w_cov = BSA_stats_tree_seq(tree_file, MUT_RATE, sample_size, mut_pos, coverage=coverage)
    two_g_stats_dicts = BSA_stats_tree_seq(tree_file, MUT_RATE, sample_size, mut_pos, coverage=coverage)
    
#     smoothed_dicts = ()
#     tricube_weighted_Gprime_dict, resolution_Gprime = tricube_smooth_G(all_snp_g_dict)
#     if (coverage != None):
#         tricube_weighted_Gprime_w_cov, resolution_Gprime_w_cov = tricube_smooth_G(all_snp_g_dict_w_cov)
#         smoothed_dicts.append((tricube_weighted_Gprime_w_cov, tricube_weighted_Gprime_dict))
#     else:
#         smoothed_dicts.append((dict(), tricube_weighted_Gprime_dict))
    
    with open('BSA_Resolution_vs_Statistics_SampleSize-%s_Coverage-%s' % (sample_size*2, coverage)+'.pckl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(two_g_stats_dicts, f, pickle.HIGHEST_PROTOCOL)

    print("File output successfully!")
    
if __name__ == '__main__':
    main()
