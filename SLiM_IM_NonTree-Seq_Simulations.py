import numpy as np
import os
from collections import OrderedDict, defaultdict
import time
import pickle
import multiprocessing as mp
from subprocess import Popen, PIPE, check_output, run


"""
    SLiM Simulation functions for Introgression Mapping (IM)
"""
def output_SLiM_IM(slim_file, Ne, recom_rate, chr_len, snp_sep, mut_pos, gen, full_output_file):
    """
        output_SLiM_IM: output SLiM model for backcross (BC)

        All the arguments taken are strings (even for the numerical parameters), to conform with the numerical format required by SLiM
        Args:
            slim_file (string): SLiM file for simulating introgression mapping
            Ne (string): string of effective population size in BSA experiemnt (number of individuals allowed to reproduce per generation)
            recom_rate (string): string of recombination rate
            chr_len (string): string of length of chromosome
            snp_sep (string): string of separating distance between SNPs
            mut_pos (string): string of causal mutation position
            gen (string): string of number of generation in BSA experiment
            full_output_file (string): the file to store SLiM simulation output

        Returns:
            slim_file (string): file name of SLiM script

    """
    f_in = open(slim_file, "w")
    ## initialize
    f_in.write('initialize() {\n')
    f_in.write('\tinitializeSex("A");\n')
    f_in.write('\tinitializeMutationRate(0);\n')
    f_in.write('\tinitializeMutationType("m1", 0.5, "f", 0.0);\n')
    f_in.write('\tinitializeGenomicElementType("g1", m1, 1.0);\n')
    f_in.write('\tinitializeGenomicElement(g1, 0, {});\n'.format(chr_len))
    f_in.write("\tinitializeRecombinationRate({});".format(recom_rate)+"\n}\n\n")
    
    f_in.write('1 late() {\n')
    ## add population
    f_in.write("\tsim.addSubpop(\"p1\", {});".format(Ne)+"\n")
    ## add mutations to create two different strains
    f_in.write("\tfemales = p1.sampleIndividuals({}, sex='F');\n".format(int(Ne)//2))
    f_in.write('\tfemales.genomes.addNewDrawnMutation(m1, seq(0,{},{}));'.format(chr_len, snp_sep)+'\n}\n\n')

    f_in.write('3:{} modifyChild()'.format(int(gen)-1)+'{\n')
    f_in.write('\tif (sim.generation % 2 == 0) {\n')
    f_in.write('\t\tMutsOnChromosome1 = childGenome1.mutations;\n')
    f_in.write('\t\tMutsOnChromosome2 = childGenome2.mutations;\n')
    f_in.write('\t\tif (parent1Genome1.containsMarkerMutation(m1, {}) & parent1Genome2.containsMarkerMutation(m1, {}))'.format(mut_pos, mut_pos)+' {\n')
    f_in.write('\t\t\tif (all(parent1.containsMutations(MutsOnChromosome1)))\n')
    f_in.write('\t\t\t\tchildGenome2.removeMutations();\n')
    f_in.write('\t\t\telse\n')
    f_in.write('\t\t\t\tchildGenome1.removeMutations();\n')
    f_in.write('\t\t\treturn T;\n\t\t}\n')
    f_in.write('\t\telse if (parent2Genome1.containsMarkerMutation(m1, {}) & parent2Genome2.containsMarkerMutation(m1, {}))'.format(mut_pos, mut_pos)+'{\n')
    f_in.write('\t\t\tif (all(parent2.containsMutations(MutsOnChromosome2)))\n')
    f_in.write('\t\t\t\tchildGenome1.removeMutations();\n')
    f_in.write('\t\t\telse\n')
    f_in.write('\t\t\t\tchildGenome2.removeMutations();\n')
    f_in.write('\t\t\treturn T;\n\t\t}\n')
    f_in.write('\t\telse\n\t\t\treturn F;\n\t}\n')
    f_in.write('\telse\n\t\treturn T;\n}\n\n')

    f_in.write('{} late()'.format(gen)+' {\n')
    f_in.write("\tsim.outputFull('{}');".format(full_output_file)+'\n}')

    f_in.close()
    return slim_file


def run_slim(slim_file, remove=True):    
    """
        run_slim: Run a single SLiM models in the command line
        
        Args:
            slim_file (string): the SLiM file name for simulation
            remove (boolean): determine whether to remove SLiM file after running it to save space

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


def IM_genomic_resolution_non_treeseq(pop_file, causal_mut, snp_sep, sample_size, window_type='open'):
    """
        IM_genomic_resolution_non_treeseq: get the open-window resolution of a given sample size
        
        Args:
            pop_file (string): the population's genomes output by SLiM
            causal_mut (int): causal mutation position
            snp_sep (int): separating base pairs between simulated SNPs
            sample_size (int): the number of (DIPLOID) individuals selected from the population
            window_type (string): the type of genomic resolution (open or closed)
            
        Returns:
            resolution (float): genomic resolution of BC experiment
        
    """
    read_in = open(pop_file, 'r')
    lines = read_in.readlines()
    read_in.close()

    # read in the genomes from SLiM output full file
    # get the SNPs on each individual
    mutation_start_idx = lines.index("Mutations:\n")
    individual_start_idx = lines.index("Individuals:\n")
    genome_start_idx = lines.index("Genomes:\n")

    # map the SNPs' IDs with their positions on the genome
    mut_idx_pos_tuples = [(int(x.split(" ")[0]), int(x.split(" ")[3])) for x in lines[mutation_start_idx+1:individual_start_idx]]
    mut_idx_map_pos = dict((x,y) for x,y in mut_idx_pos_tuples)
    
    ind_genomes = lines[genome_start_idx+1:]
    recom_points = {'AA homo': [], 'hetero': [], 'aa homo': []}

    # get the closest recombination point of each individual's two haploid genomes
    for ind in range(0, len(ind_genomes), 2):
        mut_on_strain_1 = [int(x) for x in ind_genomes[ind].split(" ")[2:]]
        mut_on_strain_2 = [int(x) for x in ind_genomes[ind+1].split(" ")[2:]]
        
        mut_strain_1_pos = np.array(sorted([mut_idx_map_pos[mut] for mut in mut_on_strain_1]))
        mut_strain_2_pos = np.array(sorted([mut_idx_map_pos[mut] for mut in mut_on_strain_2]))
        
        mut_strain_1_pos = mut_strain_1_pos[mut_strain_1_pos >= causal_mut]
        if (causal_mut in mut_strain_1_pos):
            idx1 = np.where(np.diff(mut_strain_1_pos) != snp_sep)[0]
            if (len(idx1) == 0):
                recom_point_1_ = mut_strain_1_pos[-1]
            else:
                recom_point_1_ = mut_strain_1_pos[idx1[0]]
        else:
            if (len(mut_strain_1_pos) > 0):
                recom_point_1_ = mut_strain_1_pos[0]
            else:
                recom_point_1_ = np.inf

        mut_strain_2_pos = mut_strain_2_pos[mut_strain_2_pos >= causal_mut]
        if (causal_mut in mut_strain_2_pos):
            idx2 = np.where(np.diff(mut_strain_2_pos) != snp_sep)[0]
            if (len(idx2) == 0):
                recom_point_2_ = mut_strain_2_pos[-1]
            else:
                recom_point_2_ = mut_strain_2_pos[idx2[0]]
        else:
            if (len(mut_strain_2_pos) > 0):
                recom_point_2_ = mut_strain_2_pos[0]
            else:
                recom_point_2_ = np.inf

        if (causal_mut in mut_strain_1_pos and causal_mut in mut_strain_2_pos):
            recom_points['AA homo'].append(min(recom_point_1_, recom_point_2_))
        elif (causal_mut not in mut_strain_1_pos and causal_mut not in mut_strain_2_pos):
            recom_points['aa homo'].append(min(recom_point_1_, recom_point_2_))
        else:
            recom_points['hetero'].append(min(recom_point_1_, recom_point_2_))
        
    # print('AA Homozygous: ', len(recom_points['AA homo']))
    # print('aa Homozygous: ', len(recom_points['aa homo']))

    if (sample_size <= len(recom_points['AA homo'])):
        recom_point = min(np.random.choice(recom_points['AA homo'], sample_size, replace=False))
    elif (sample_size <= len(recom_points['AA homo']) + len(recom_points['hetero'])):
        recom_point = min(np.random.choice(recom_points['hetero'], 2*(sample_size-len(recom_points['AA homo'])), replace=False))
        recom_point = min(min(recom_points['AA homo']), recom_point)
    else:
        return np.nan

    return (recom_point - causal_mut)


def main():
    """
        Define some SLiM simulation population parameters
    """
    Ne = '100'
    T = '11'
    R = '1e-8'
    chr_len = '1e8-1'
    snp_sep = '1e4'
    mut_pos = '1e4'

    slim_file = 'SLiM_NV_IM_NonTree-seq'
    output_file_prefix = 'SLiM_NV_IM'

    # number of simulations per each run
    num_sim = 1000

    count = mp.cpu_count()
    pool = mp.Pool(processes=count//2)
    print("Pool started")

    slim_sims = [pool.apply_async(output_SLiM_IM, args=(slim_file+'{}.slim'.format(_), Ne, R, chr_len, snp_sep, mut_pos, T, output_file_prefix+'{}.txt'.format(_))) for _ in range(num_sim)]
    slim_files = [s.get() for s in slim_sims]
    print("Number of SLiM files:", len(slim_files))
    
    SLIM = run_slims(slim_files, pool)
    print("SLiM Files run and removed")
    slim_output_files = list(filter(lambda x : output_file_prefix in x and x.endswith('.txt'),os.listdir()))
    print("Number of SLiM files:", len(slim_output_files))
    
    S_s = [2**x for x in range(1,5)] 
    resolution_all = OrderedDict([(S,[]) for S in S_s])

    for S in S_s:
        random_slim_output_files = np.random.choice(slim_output_files, 1000)
        results_slim = [pool.apply_async(IM_genomic_resolution_non_treeseq, args=(pop_file, int(float(mut_pos)), int(float(snp_sep)), S, )) for pop_file in random_slim_output_files]
        resolution_slim = [s.get() for s in results_slim]
        resolution_all[S] += resolution_slim
        print('{} Sample Size finished'.format(S), len(resolution_all[S]))
    
    remove_txt = pool.map(os.remove, slim_output_files)
    print("Number of SLiM files:", len(list(filter(lambda x : 'Tree_Seq_Popsize_'+str(Ne)+'_' in x and x.endswith('_gen'+str(T+1)+'.txt'),os.listdir()))))

    f_out = 'IM_Open-win_Res_100MbChrom_Ne%s_%scM_Non-TreeSeq_s-top16' % (Ne, float(R)*1e8)+'.pckl'
    print(f_out)

    with open(f_out, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(resolution_all, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
