print('\nLoading packages')
import pandas as pd
import numpy as np
import math
from scipy import interpolate
import scanpy as sc; sc.settings.verbosity = 0
from scanpy.external.pp import hashsolo
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm
import os
import shutil
import anndata as ad
import argparse
import time
from itertools import product
from functools import reduce
from scipy.sparse import issparse
from bioservices import KEGG
from CRISPR_functions import *
from autoencoder.Run_SSAE_alldata import run_SSAE, new_SSAE

start = time.time()

def main() :

    """
    Load, normalize, merge and demultiplex CRISPR counts matrices
    Find discriminant features between perturbed and control with a Supervised AutoEncoder 

    Requird inputs :
        
        - Counts library
        - Negative control guides as bash inputs or python inputs
        
    Optional inputs :
        - Separate gRNA library
        - Separate HTO library
        - Number of paired libraries to concatenate and treat simultaneously
        - Priors for gRNA demultiplexing

    Outputs :
    - Classification for each cell
    - AutoEncoder scores (Precision, Recall, F1...)
    - Ranking of most discriminant features
    - Pathways associated to most discriminant features
    """    

    parser = argparse.ArgumentParser(
        prog = 'demux', 
        formatter_class=argparse.MetavarTypeHelpFormatter,
        description = 'Detects HTO and gRNA present in each cell and creates a classification dataframe')
    
    parser.add_argument('-libs', type = int, help = 'Number of libraries to treat at the same time', default = 1)
    parser.add_argument('-counts', type = dir_path, help = 'Path/to/counts/library_1/', required = True)
    parser.add_argument('-grna', type = dir_path, help = 'Path/to/gRNA/library_1/')
    parser.add_argument('-hto', type = dir_path, help = 'Path/to/hto/library_1/') 
    parser.add_argument('-nohto', action='store_true', help = 'Add -nohto i you do not have HTO to demultiplex in yout dataset', default = False)
    parser.add_argument('-priors', nargs = '+', type =float, help = 'Define priors for gRNA negatives, singlets and doublets ratio', default = [0.01, 0.8, 0.19])
    parser.add_argument('-neg', nargs = '+', type =str, help = 'Name of negative control gRNAs', default = None)
    parser.add_argument('-runs', type = int, help = 'Number of random samplings and AutoEncoder runs to perform', default = 1)
    parser.add_argument('-eta', action='store_true', help = 'Test multiple eta values to get highest accuracy', default = False)
    parser.add_argument('-plot', action='store_true', help = 'Add -plot to save demultiplexing distribution plots', default = False)
    parser.add_argument('-pathways', action='store_true', help = 'Add -pathways if you want to find pathways associated to te top genes', default = False)
    
    args = parser.parse_args()

    cfolder = args.counts

    if args.grna :
        gfolder = args.grna
        #Find and gRNA names file
        features_file = next((file for file in os.listdir(gfolder) if 'features' in file), None)
        grna_names = list(pd.read_csv(gfolder + features_file, sep = '\t', names = ['Names']).Names)
        #Uniformize the gRNA names and store the target genes in "targets" 
        grna_names, targets = clean_guides(grna_names, args.neg)

    if args.hto :
        hfolder = args.hto
        #Find and HTO names file
        features_file = next((file for file in os.listdir(hfolder) if 'features' in file), None)
        hto_names = list(pd.read_csv(hfolder + features_file, sep = '\t', names = ['Names']).Names)
        #Remove the nucleotide sequence from the name 
        hto_names = [hto.split('-')[0] for hto in hto_names]

    if args.libs > 1 :
        #If multiple datasets, store them in a dictionary
        counts_matrices = {}
        for i in range(args.libs) :
            i+=1
            print(f"\nLoading counts matrix n°{i}")
            #Change the directory name to iteratively load the counts matrices
            folder = replace_digit(cfolder, i)
            #Find the prefix before 'matrix.mtx.gz', 'barcode.tsv.gz' and 'features.tsv.gz'
            prefix = get_prefix(folder)
            #Load the counts matrix as an AnnData object with cell barcodes in .obs and genes names in .var
            matrix = sc.read_10x_mtx(folder, prefix=prefix, cache_compression='gzip', gex_only=False)
            #Remove the nucleotide sequence from the HTO names
            matrix.obs_names = [barcode.split('-')[0] + f"-{i}" for barcode in matrix.obs_names]
            #CPM normalization on the gene counts
            sc.pp.normalize_total(matrix, target_sum=1e6)
            #Add the current matrix to the matruces dictionary
            counts_matrices[f"matrix_{i}"] = matrix
        
    elif args.libs == 1 :
        print('\nLoading counts matrix')
        #Find the prefix before 'matrix.mtx.gz', 'barcode.tsv.gz' and 'features.tsv.gz'
        prefix = get_prefix(cfolder)
        #Load the counts matrix as an AnnData object
        counts_adata = sc.read_10x_mtx(cfolder, prefix=prefix, cache_compression='gzip', gex_only=False)
        #CPM normlaization on the gene counts
        sc.pp.normalize_total(counts_adata, target_sum=1e6)
        # Remove the nucleoide sequence from the HTO names
        counts_adata.obs.index = [barcode.split('-')[0] for barcode in counts_adata.obs.index]


    ####################################################
    #If you do not use oligonuleotides in yor experiment
    ####################################################
    if args.nohto :

        print('Loading gRNA counts')
        
        if args.libs > 1 :
            i+=1
            for i in range(args.libs) :
                #If the gRNA counts are in a separate matrix
                
                if args.grna :
                    print(f"\nLoading gRNA matrix n°{i}")

                    #Change the directory name to iteratively load the gRNA matrices
                    folder = replace_digit(gfolder, i)
                    #Find the gRNA counts matrix in the directory
                    matrix_file = next((file for file in os.listdir(folder) if 'matrix' in file), None)
                    #Find the cell barcodes file in the directory
                    barcodes_file = next((file for file in os.listdir(folder) if 'barcodes' in file), None)
                    matrix_name = f"matrix_{i}"
                    #Load the gRNA counts matrix as an AnnData object 
                    matrix = sc.read_mtx(folder + matrix_file).T
                    #CPM normalization on gRNA counts 
                    sc.pp.normalize_total(matrix, target_sum = 1e6)
                    
                    #Load the cell barcodes and add them to the dataset 
                    barcodes = list(pd.read_csv(folder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                    #Replace the barcode index to differrenciate the libraries
                    matrix.obs.index = [barcode.split('-')[0] + f"-{i}" for barcode in barcodes]

                    #Store the gRNA counts as integers in .obs for demultiplexing 
                    matrix.obs = matrix.to_df().astype(int)
                    #Rename the columns as the gRNA names
                    matrix.obs.columns = grna_names
                    #Remove the cells that have 0 counts for all gRNAs
                    matrix = matrix[matrix.obs.sum(axis=1) > 0]

                    #Demultiplex the gRNA using Hashsolo from Scanpy (Bernstein, et al. 2020)
                    hashsolo(matrix, grna_names, priors=args.priors)

                    #Store the gRNA labels in the main counts matrix
                    counts_matrices[matrix_name].obs['Classif_gRNA'] = matrix.obs['Classification']

                else :
                    #If the gRNA counts are part of the main counts matrix

                    #Find the rows corresponding to gRNAs in the matrix
                    grna_rows = find_guides(counts_matrices[f"matrix_{i}"])
                    #Store these rows names as grna_names
                    grna_names = [counts_matrices[f"matrix_{i}"].var_names[row] for row in grna_rows]
                    #Uniformize the gRNA names and store the target genes in "targets"
                    grna_names, targets = clean_guides(grna_names, args.neg)

                    #Store each gRNA row as a column in .obs
                    for row, name in zip(grna_rows, grna_names) :
                        if hasattr(counts_matrices[f"matrix_{i}"].X, 'toarray') :
                            counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].A.ravel().astype(int)
                        else :
                            counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].astype(int)

                    #Remove the gRNA counts from the matrix
                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][:, ~counts_matrices[f"matrix_{i}"].var_names.isin(grna_names)]
                    #Remove the cells that have 0 counts for all gRNAs
                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][counts_matrices[f"matrix_{i}"].obs.sum(axis=1) > 0]

                    #Demultiplex the gRNA using Hashsolo from Scanpy (Bernstein, et al. 2020)
                    hashsolo(counts_matrices[f"matrix_{i}"], grna_names, priors=args.priors)

                    #Remove the columns other than the gRNA classification from .obs
                    counts_matrices[f"matrix_{i}"].obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace = True)
            
            #Concatenate the AnnData objects as a single objet
            counts_adata = ad.concat(list(counts_matrices.values()), label = 'Library')
        
        else :
            #Same processes with a single library and no concatenation

            if args.grna :
                matrix_file = next((file for file in os.listdir(gfolder) if 'matrix' in file), None)
                barcodes_file = next((file for file in os.listdir(gfolder) if 'barcodes' in file), None)
                grna_adata = sc.read_mtx(gfolder + matrix_file).T
                sc.pp.normalize_total(grna_adata, target_sum = 1e6)

                barcodes = list(pd.read_csv(gfolder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                grna_adata.obs.index = [barcode.split('-')[0] for barcode in barcodes]

                grna_adata.obs = grna_adata.to_df()
                grna_adata.obs.columns = grna_names

                grna_adata = grna_adata[grna_adata.obs.sum(axis=1) > 0]

                hashsolo(grna_adata, grna_names, priors=args.priors)

                counts_adata.obs['Classif_gRNA'] = grna_adata.obs['Classification']  

            else :
                grna_rows = find_guides(counts_adata)
                grna_names = [counts_adata.var_names[row] for row in grna_rows]
                grna_names, targets = clean_guides(grna_names, args.neg)
                for row, name in zip(grna_rows, grna_names) :
                    if hasattr(counts_matrices[f"matrix_{i}"].X, 'toarray') :
                        counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].A.ravel().astype(int)
                    else :
                        counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].astype(int)

                counts_adata = counts_adata[:, ~counts_adata.var_names.isin(grna_names)]
                counts_adata = counts_adata[counts_adata.obs.sum(axis=1) > 0]

                hashsolo(counts_adata, grna_names, priors=args.priors)

                counts_adata.obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace = True)
                
        if args.plot :

            distrib_dir = f"{results_dir}/Distribution_plots/"

            #Check if the plots directory already exists
            if not os.path.exists(distrib_dir):
                os.makedirs(distrib_dir)
            else :
                print(f"Directory '{distrib_dir}' already exists.")

            #Put the gRNA classification results in a Pandas dataframe for plotting 
            hashed_grna = pd.DataFrame(counts_adata.obs)

            plt.figure(figsize=(25, 9))

            if args.libs > 1 :
                sns.countplot(data = hashed_grna, x = 'Classification',
                        hue = 'Library', palette = 'Set2')
                plt.legend(title='Libraries')
            elif args.libs == 1 :
                sns.countplot(data = hashed_grna, x = 'Classification', palette = 'Set2')   
            
            plt.xlabel('Guide')
            plt.ylabel('Count')
            plt.title('Distribution of gRNA classification by Hashsolo')
            plt.savefig(f"{distrib_dir}gRNA_distribution.png")

        #Remove the doublets, negatives and unmapped reads
        counts_adata = counts_adata[counts_adata.obs.Classification.isin(grna_names)]
        counts_adata.obs.index = counts_adata.obs.Barcode
        #Keep only the classification in .obs
        counts_adata.obs = counts_adata.obs[['Classification']]
        counts_adata.obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace = True)

        print('\nSelecting 10k most expressed genes')
        #Select the 10,000 most expressed genes in the counts matrix to reduce the computational cost

        #Make the matrix dense if it is sparse
        gene_counts_array = counts_adata.X.A if issparse(counts_adata.X) else counts_adata.X
        gene_counts = gene_counts_array.sum(axis=0)
        gene_counts_array_1d = gene_counts.A1 if issparse(gene_counts) else gene_counts.flatten()

        #make a Pandas dataframe containing the genes names and their counts sum
        gene_expression_df = pd.DataFrame({
            'Gene': counts_adata.var_names,
            'ExpressionSum': gene_counts_array_1d
        })
        
        #Sort the genes by descending counts sum
        sorted_genes = gene_expression_df.sort_values(by='ExpressionSum', ascending=False)
        #Store the names of the 10,000 most expressed genes in a list 
        top_10k_genes = sorted_genes.head(10000)['Gene'].tolist()
        #Select the rows corresponding to the 10,000 most expressed genes from the counts matrix
        top10k =counts_adata[:, counts_adata.var_names.isin(top_10k_genes)].copy()    

        #list of eta parameters to test as hyperparameter for the autoencoder if '-eta' is specified 
        list_ETA = [0.05,0.1,0.5,1]
        #dictionary that contains the most differentially expressed genes for each condition and their rank
        allresults = {}
        #dictionary that contains the log(fold change) of the studied genes between the perturbed cells and control cells 
        expression = {}

        for condition in targets :
            #create a list for each condition in each dictionary
            allresults[condition] = []
            expression[condition] = []
        
        #Create a dataframe in which the accuracies by eta will be stored for each condition
        acc_df = pd.DataFrame(columns=[f'eta_{ETA}' for ETA in list_ETA])
        
        #Take the cells with non-targeting gRNA as negative control
        Neg = top10k[top10k.obs['Classif_gRNA'].str.contains('Neg')].to_df().T
        #Add the label 0 for 'control cell' for each cell in the dataframe
        Neg.loc['Label'] = pd.Series(np.zeros(len(Neg.columns)), index=Neg.columns)
        Neg = pd.concat([Neg.loc[['Label']], Neg.drop('Label')])

        if args.eta :
            for target in targets : 
                target_data = top10k[top10k.obs['Classif_gRNA'].str.contains(target)].to_df().T
                #Add the label 1 for 'perturbded' for each cell in the dataframe
                target_data.loc['Label'] = pd.Series(np.ones(len(target_data.columns)), index=target_data.columns)
                target_data = pd.concat([target_data.loc[['Label']], target_data.drop('Label')])
                if len(target_data.columns) > 0 and len(Neg.columns) > 0:    
                    #Take a random sample of both datasets to have a matching number of control and perturbed cells
                    Neg_cut = Neg.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=0)
                    target_data_cut = target_data.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=0)
                    #Concatenate the negative control and the perturbed cells counts
                    dataset = pd.concat([Neg_cut, target_data_cut], axis=1)
                    
                    #create a list that will contain the accuracy for each eta parameter tested
                    list_acc = []
                    for ETA in list_ETA :
                        try :
                            new_SSAE(target, dataset, results_dir, eta=ETA)
                            #Add the accuracy of the run to the list of accuracies
                            list_acc.append(pd.read_csv(f'{results_dir}/{target}/bilevel_proj_l1Inftyball_acctest.csv', header=0, index_col=0, sep=';').Global.loc['Mean'])
                            #Keep the results files only if no other runs are expected
                            if args.runs >= 1 :
                                shutil.rmtree(f"{results_dir}/{target}")
                            else :
                                if ETA != list_ETA[-1] :
                                    shutil.rmtree(f"{results_dir}/{target}")
                            
                        except Exception :
                            print(f"Error for {target}_{HTO} ! Not enough data")
                            shutil.rmtree(f"{results_dir}/{target}")
                            pass

                    #Add the list of accuracies as a row in the accuracy dataframe 
                    acc_df.loc[target] = list_acc
                    #Save the dataframe after each condition to be able to check the advancement of the script
                    acc_df.to_csv(f'{results_dir}/accuracies.csv')

                else : pass

            #plot accuracy = f(eta) for each condition
            eta_fig(acc_df)
            
            #Add a row containing the number of times each eta parameter has the max accuracy
            acc_df = pd.concat([acc_df, pd.DataFrame(acc_df.idxmax(axis=1).value_counts()).T])
            acc_df.to_csv(f'{results_dir}/accuracies.csv')

        if args.runs>1 :
            for run in range(args.runs) :
                for target in targets : 
                    target_data = top10k[top10k.obs['Classif_gRNA'].str.contains(target)].to_df().T
                    #Add the label 1 for 'perturbded' for each cell in the dataframe
                    target_data.loc['Label'] = pd.Series(np.ones(len(target_data.columns)), index=target_data.columns)
                    target_data = pd.concat([target_data.loc[['Label']], target_data.drop('Label')])
                    if len(target_data.columns) > 0 and len(Neg.columns) > 0:    
                        #Take a random sample of both datasets to have a matching number of control and perturbed cells
                        Neg_cut = Neg.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=run)
                        target_data_cut = target_data.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=run)
                        #Concatenate the negative control and the perturbed cells counts
                        dataset = pd.concat([Neg_cut, target_data_cut], axis=1)
                        #create a dataframe that contains the expression of each gene in perturbed and control cells for the current run
                        expression_df = pd.DataFrame({'Gene' : Neg.index.to_list()[1:], 
                                                      'Perturbed_expression' : target_data.iloc[1:, :].mean(axis=1), 
                                                      'Control_expression' : Neg.iloc[1:, :].mean(axis=1)})
                        #Add a column that contains the log fold change of each gene between control and perturbed cells
                        expression_df['log2_ratio'] = np.log2(expression_df['Perturbed_expression'] / expression_df['Control_expression'])
                        #Add the dataframe to the list of dataframes of the current condition
                        expression[target].append(expression_df)
                        
                        try :
                            #If the best eta was determined beforehand, use it as hyperparameter
                            if args.eta :
                                new_SSAE(target, dataset, results_dir, HTO, eta=float(acc_df.loc[target].idxmax().split('_')[-1]))
                            else :
                                new_SSAE(target, dataset, results_dir, HTO, eta=0.25)

                            #Store the most differentially expressed genes, their weight and their rank in a dataframe
                            scores = pd.read_csv(f'{results_dir}/{target}/bilevel_proj_l1Inftyball_topGenes_Captum_dl_300.csv', header=0, sep=';')[['Features', 'Mean']]
                            scores['Rank'] = scores.index + 1
                            #Add the dataframe to the list of dataframes corresponding to the current condition
                            allresults[target].append(scores)
                            #Keep the results for the last run only
                            if run != args.runs-1 :
                                shutil.rmtree(f"{results_dir}/{target}")
                        except Exception :
                            print(f"Error for {target}_{HTO} ! Not enough data")
                            shutil.rmtree(f"{results_dir}/{target}")
                            pass
        
            for condition in tqdm(targets) :
                
                #For each condition, make a dataframe that contains the mean weight, weight std, mean rank, rank std and log2 fold change of each gene 
                if allresults[condition] and expression[condition]:
                    results1=allresults[condition][0]

                    for idx, results in enumerate(allresults[condition]) :
                        results.index.name = 'Features'
                        if idx != 0 :
                            results = results.reindex(allresults[condition][0].index)                        
                    
                    df = pd.DataFrame({'Gene' : results1['Features'], 
                                    'Mean_Weight' : pd.concat(allresults[condition], axis=1)['Mean'].mean(axis=1),
                                    'Weight_Std' : pd.concat(allresults[condition], axis=1)['Mean'].std(axis=1),
                                    'Mean_Rank' : pd.concat(allresults[condition], axis=1)['Rank'].mean(axis=1),
                                    'Rank_Std' : pd.concat(allresults[condition], axis=1)['Rank'].std(axis=1)})
                    df = df.sort_values(by='Mean_Rank')
                    df.index.name = 'Gene'

                    df2 = pd.DataFrame({'Gene' : expression[condition][0]['Gene'].to_list(), 
                                        'log2_ratio' : pd.concat(expression[condition], axis=1)['log2_ratio'].mean(axis=1)})
                    df2.index.name='Gene'
                    df2 = df2.reindex(df.index)
                    df['log2_ratio'] = df2['log2_ratio']
                    df.index.name = 'Gene'
                    parts 
                    df.to_csv(f'{ranks_dir}/{condition}.csv', index = True)   
                else :
                    pass 
        
        else : 

            for target in targets : 
                target_data = top10k[top10k.obs['Classif_gRNA'].str.contains(target)].to_df().T
                #Add the label 1 for 'perturbded' for each cell in the dataframe
                target_data.loc['Label'] = pd.Series(np.ones(len(target_data.columns)), index=target_data.columns)
                target_data = pd.concat([target_data.loc[['Label']], target_data.drop('Label')])
                if len(target_data.columns) > 0 and len(Neg.columns) > 0:    
                    
                    Neg_cut = Neg.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=run)
                    target_data_cut = target_data.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=run)
                    #Concatenate the negative control and the perturbed cells counts
                    dataset = pd.concat([Neg_cut, target_data_cut], axis=1)
                    if run != args.runs-1 :
                        try :
                            if args.eta :
                                new_SSAE(target, dataset, results_dir, eta=float(acc_df.loc[target].idxmax().split('_')[-1]))
                            else :
                                new_SSAE(target, dataset, results_dir, eta=0.25)
                        except Exception :
                            print(f"Error for {target}_{HTO} ! Not enough data")
                            shutil.rmtree(f"{results_dir}/{target}")
                            pass

        results_files(results_dir, targets, pathways = args.pathways)       

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Running time - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) 


    else :
        if args.libs > 1 :
            for i in range(args.libs) :
                i+=1
                if args.hto :
                    #If the HTO counts are in a separate matrix

                    print(f"\nLoading HTO matrix n°{i}")

                    #Change the directory name to iteratively load the HTO matrices
                    folder = replace_digit(hfolder, i)
                    #Find the HTO counts matrix in the directory
                    matrix_file = next((file for file in os.listdir(folder) if 'matrix' in file), None)
                    #Find the cell barcodes file in the directory
                    barcodes_file = next((file for file in os.listdir(folder) if 'barcodes' in file), None)
                    matrix_name = f"matrix_{i}"
                    #Load the gRNA counts matrix as an AnnData object 
                    matrix = sc.read_mtx(folder + matrix_file).T
                    #CPM normalization on all HTO counts 
                    sc.pp.normalize_total(matrix, target_sum = 1e6)
                    
                    #Load the cell barcodes and add them to the dataset 
                    barcodes = list(pd.read_csv(folder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                    #Replace the barcode index to differenciate the libraries
                    matrix.obs.index = [barcode.split('-')[0] + f"-{i}" for barcode in barcodes]

                    #Store the HTO counts as integers in .obs for demultiplexing 
                    matrix.obs = matrix.to_df().astype(int)
                    #Rename the columns as the HTO names
                    matrix.obs.columns = hto_names
                    #Remmove the cells that have 0 counts for all HTOs
                    matrix = matrix[matrix.obs.sum(axis=1) > 0]

                    #Demultiplex the gRNA using Hashsolo from Scanpy (Bernstein, et al. 2020)
                    hashsolo(matrix, hto_names)

                    #Store the HTO labels in the main counts matrix
                    counts_matrices[matrix_name].obs['Classif_HTO'] = matrix.obs['Classification']

                else :
                    #If the HTO counts are part of the main counts matrix

                    #Find the rows correpondung to HTOs in the matrx
                    hto_rows = find_HTOs(counts_matrices[f"matrix_{i}"])
                    #Store these rows names as hto_names
                    hto_names = [counts_matrices[f"matrix_{i}"].var_names[row] for row in hto_rows]

                    #Store each HTO row as a column in .obs
                    for row, name in zip(hto_rows, hto_names) :
                        if hasattr(counts_matrices[f"matrix_{i}"].X, 'toarray') :
                            counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].A.ravel().astype(int)
                        else :
                            counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].astype(int)

                    #Remove the HTO counts from the matrix 
                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][:, ~counts_matrices[f"matrix_{i}"].var_names.isin(hto_names)]
                    #Remove the cells that have 0 counts for all HTOs
                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][counts_matrices[f"matrix_{i}"].obs.sum(axis=1) > 0]

                    #Demultiplex the HTO using Hashsolo from Scanpy (Bernstein, et al. 2020)
                    hashsolo(counts_matrices[f"matrix_{i}"], hto_names)

                    #Remove the columns other than the HTO classification from .obs
                    counts_matrices[f"matrix_{i}"].obs.rename(columns={'Classification' : 'Classif_HTO'}, inplace = True)
                
                if args.grna :
                    print(f"\nLoading gRNA matrix n°{i}")

                    #Change the directory name to iteratively load the gRNA matrices
                    folder = replace_digit(gfolder, i)
                    #Find the gRNA counts matrix in the directory
                    matrix_file = next((file for file in os.listdir(folder) if 'matrix' in file), None)
                    #Find the cell barcodes file in the directory
                    barcodes_file = next((file for file in os.listdir(folder) if 'barcodes' in file), None)
                    matrix_name = f"matrix_{i}"
                    #Load the gRNA counts matrix as an AnnData object 
                    matrix = sc.read_mtx(folder + matrix_file).T
                    #CPM normalization on gRNA counts 
                    sc.pp.normalize_total(matrix, target_sum = 1e6)
                    
                    #Load the cell barcodes and add them to the dataset 
                    barcodes = list(pd.read_csv(folder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                    #Replace the barcode index to differrenciate the libraries
                    matrix.obs.index = [barcode.split('-')[0] + f"-{i}" for barcode in barcodes]

                    #Store the gRNA counts as integers in .obs for demultiplexing 
                    matrix.obs = matrix.to_df().astype(int)
                    #Rename the columns as the gRNA names
                    matrix.obs.columns = grna_names
                    #Remove the cells that have 0 counts for all gRNAs
                    matrix = matrix[matrix.obs.sum(axis=1) > 0]

                    #Demultiplex the gRNA using Hashsolo from Scanpy (Bernstein, et al. 2020)
                    hashsolo(matrix, grna_names, priors=args.priors)

                    #Store the gRNA labels in the main counts matrix
                    counts_matrices[matrix_name].obs['Classif_gRNA'] = matrix.obs['Classification']    

                else :
                    #If the gRNA counts are part of the main counts matrix

                    #Find the rows corresponding to gRNAs in the matrix
                    grna_rows = find_guides(counts_matrices[f"matrix_{i}"])
                    #Store these rows names as grna_names
                    grna_names = [counts_matrices[f"matrix_{i}"].var_names[row] for row in grna_rows]
                    #Uniformize the gRNA names and store the target genes in "targets"
                    grna_names, targets = clean_guides(grna_names, args.neg)

                    #Store each gRNA row as a column in .obs
                    for row, name in zip(grna_rows, grna_names) :
                        if hasattr(counts_matrices[f"matrix_{i}"].X, 'toarray') :
                            counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].A.ravel().astype(int)
                        else :
                            counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].astype(int)

                    #Remove the gRNA counts from the matrix
                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][:, ~counts_matrices[f"matrix_{i}"].var_names.isin(grna_names)]
                    #Remove the cells that have 0 counts for all gRNAs
                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][counts_matrices[f"matrix_{i}"].obs.sum(axis=1) > 0]

                    #Demultiplex the gRNA using Hashsolo from Scanpy (Bernstein, et al. 2020)
                    hashsolo(counts_matrices[f"matrix_{i}"], grna_names, priors=args.priors)

                    #Remove the columns other than the gRNA classification from .obs
                    counts_matrices[f"matrix_{i}"].obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace = True)
            
            #Concatenate the AnnData objects as a single objet
            counts_adata = ad.concat(list(counts_matrices.values()), label = 'Library')

        elif args.libs == 1 :
            #Same processes with a single library and no concatenation

            if args.hto :
                matrix_file = next((file for file in os.listdir(hfolder) if 'matrix' in file), None)
                barcodes_file = next((file for file in os.listdir(hfolder) if 'barcodes' in file), None)
                hto_adata = sc.read_mtx(hfolder + matrix_file).T
                sc.pp.normalize_total(hto_adata, target_sum = 1e6)
 
                barcodes = list(pd.read_csv(hfolder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                hto_adata.obs.index = [barcode.split('-')[0] for barcode in barcodes]
                counts_adata = counts_adata[counts_adata.obs.index.isin(hto_adata.obs.index)]
                #hto_adata.obs = hto_adata.obs.set_index(counts_adata.obs.index)

                hto_adata.obs = hto_adata.to_df().astype(int)
                hto_adata.obs.columns = hto_names

                hto_adata = hto_adata[hto_adata.obs.sum(axis=1) > 0]

                hashsolo(hto_adata, hto_names)

                counts_adata.obs['Classif_HTO'] = hto_adata.obs['Classification']     

            else :
                hto_rows = find_HTOs(counts_adata)
                hto_names = [counts_adata.var_names[row] for row in hto_rows]
                for row, name in zip(hto_rows, hto_names) :
                    if hasattr(counts_adata.X, 'toarray') :
                        counts_adata.obs[name] = counts_adata.X[:, row].A.ravel().astype(int)
                    else :
                        counts_adata.obs[name] = counts_adata.X[:, row].astype(int)

                counts_adata = counts_adata[:, ~counts_adata.var_names.isin(hto_names)]
                counts_adata = counts_adata[counts_adata.obs.sum(axis=1) > 0]

                hashsolo(counts_adata, hto_names)

                counts_adata.obs.rename(columns={'Classification' : 'Classif_HTO'}, inplace = True)
            
            if args.grna :
                matrix_file = next((file for file in os.listdir(gfolder) if 'matrix' in file), None)
                barcodes_file = next((file for file in os.listdir(gfolder) if 'barcodes' in file), None)
                grna_adata = sc.read_mtx(gfolder + matrix_file).T
                sc.pp.normalize_total(grna_adata, target_sum = 1e6)

                barcodes = list(pd.read_csv(gfolder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                grna_adata.obs.index = [barcode.split('-')[0] for barcode in barcodes]

                grna_adata = grna_adata[grna_adata.obs.index.isin(counts_adata.obs.index)]

                grna_adata.obs = grna_adata.to_df().astype(int)
                grna_adata.obs.columns = grna_names

                grna_adata = grna_adata[grna_adata.obs.sum(axis=1) > 0]

                hashsolo(grna_adata, grna_names, priors=args.priors)

                counts_adata.obs['Classif_gRNA'] = grna_adata.obs['Classification'] 
            
            else :
                grna_rows = find_guides(counts_adata)
                grna_names = [counts_adata.var_names[row] for row in grna_rows]
                grna_names, targets = clean_guides(grna_names, args.neg)
                for row, name in zip(grna_rows, grna_names) :
                    if hasattr(counts_matrices[f"matrix_{i}"].X, 'toarray') :
                        counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].A.ravel().astype(int)
                    else :
                        counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].astype(int)

                counts_adata = counts_adata[:, ~counts_adata.var_names.isin(grna_names)]
                counts_adata = counts_adata[counts_adata.obs.sum(axis=1) > 0]

                hashsolo(counts_adata, grna_names, priors=args.priors)

                counts_adata.obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace = True)        

        results_dir = create_results_folder()

        if args.plot == True :

            distrib_dir = f"{results_dir}/Distribution_plots/"

            #Check if the plots directory already exists
            if not os.path.exists(distrib_dir):
                os.makedirs(distrib_dir)
            else :
                print(f"Directory {distrib_dir} already exists.")

            #Put the HTO & gRNA classification results in a dataframe for plotting
            
            hashed_data = pd.DataFrame(counts_adata.obs)
            
            #Plot the distribution of HTO Classification
            plt.figure(figsize=(25, 9))
            if args.libs > 1 :
                sns.countplot(data=hashed_data,
                        x='Classif_HTO', hue='Library', palette='Set2')
                plt.legend(title='Libraries')
            elif args.libs == 1 :
                sns.countplot(data=hashed_data,
                        x='Classif_HTO', palette='Set2')
            plt.xlabel('HTO')
            plt.ylabel('Count')
            plt.title('Distribution of HTO classification by Hashsolo')
            plt.savefig(distrib_dir + 'HTO_distribution.png') 
            plt.close()

            #Plot the ditribution of gRNA Classification
            plt.figure(figsize=(25, 9))
            if args.libs > 1 :
                sns.countplot(data=hashed_data,
                        x='Classif_gRNA', hue='Library', palette='Set2')
                plt.legend(title='Libraries')
            elif args.libs == 1 :
                sns.countplot(data=hashed_data,
                        x='Classif_gRNA', palette='Set2')
            plt.xlabel('Guide')
            plt.ylabel('Count')
            plt.title('Distribution of gRNA classification by Hashsolo')
            plt.savefig(f"{distrib_dir}gRNA_distribution.png")
            plt.close()
        
        #Keep only the classifications in .obs
        
        counts_adata.obs = counts_adata.obs[['Classif_HTO', 'Classif_gRNA']]
        #Remove 'unmapped' from the list of HTO names
        if 'unmapped' in hto_names :
            hto_names.remove('unmapped')

        #Remove the doublets, negatives and unmapped reads
        counts_adata = counts_adata[counts_adata.obs.Classif_HTO.isin(hto_names)]
        counts_adata = counts_adata[counts_adata.obs.Classif_gRNA.isin(grna_names)]

        #Select the 10,000 most expressed genes in the counts matrix to reduce the computational cost
        gene_counts_array = counts_adata.X.A if issparse(counts_adata.X) else counts_adata.X
        gene_counts = gene_counts_array.sum(axis=0)
        gene_counts_array_1d = gene_counts.A1 if issparse(gene_counts) else gene_counts.flatten()
        gene_expression_df = pd.DataFrame({
            'Gene': counts_adata.var_names,
            'ExpressionSum': gene_counts_array_1d
        })
        print('\nSelecting 10k most expressed genes')
        sorted_genes = gene_expression_df.sort_values(by='ExpressionSum', ascending=False)
        top10k = counts_adata[:, counts_adata.var_names.isin(sorted_genes.head(10000)['Gene'].tolist())].copy()

        #list of eta parameters to test as hyperparameter for the autoencoder if '-eta' is specified 
        list_ETA = [0.05,0.1,0.5,1]
        #dictionary that contains the most differentially expressed genes for each condition and their rank
        allresults = {}
        #dictionary that contains the log(fold change) of the studied genes between the perturbed cells and control cells 
        expression = {}

        for condition in product(targets, hto_names) :
            condition = '/'.join(condition)
            #create a list for each condition in each dictionary
            allresults[condition] = []
            expression[condition] = []
        
        #Create a dataframe in which the accuracies by eta will be stored for each condition
        acc_df = pd.DataFrame(columns=[f'eta_{ETA}' for ETA in list_ETA])


        if args.eta :
            #Separate the cells according to their condition (HTO)
            for HTO in hto_names :
                #Select all the cells with the HTO
                full_HTO = top10k[top10k.obs['Classif_HTO'].str.contains(HTO)]
                #Take the cells with non-targeting gRNA as negative control
                Neg = full_HTO[full_HTO.obs['Classif_gRNA'].str.contains('Neg')].to_df().T
                #Add the label 0 for 'control cell' for each cell in the dataframe
                Neg.loc['Label'] = pd.Series(np.zeros(len(Neg.columns)), index=Neg.columns)
                Neg = pd.concat([Neg.loc[['Label']], Neg.drop('Label')])

                for target in targets : 
                    target_data = full_HTO[full_HTO.obs['Classif_gRNA'].str.contains(target)].to_df().T
                    #Add the label 1 for 'perturbded' for each cell in the dataframe
                    target_data.loc['Label'] = pd.Series(np.ones(len(target_data.columns)), index=target_data.columns)
                    target_data = pd.concat([target_data.loc[['Label']], target_data.drop('Label')])
                    if len(target_data.columns) > 0 and len(Neg.columns) > 0:    
                        #Take a random sample of both datasets to have a matching number of control and perturbed cells
                        Neg_cut = Neg.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=0)
                        target_data_cut = target_data.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=0)
                        #Concatenate the negative control and the perturbed cells counts
                        dataset = pd.concat([Neg_cut, target_data_cut], axis=1)
                        condition = f'{target}/{HTO}'
                        
                        #Create a list that will contain the accuracy for each eta parameter tested
                        list_acc = []
                        for ETA in list_ETA :
                            try :
                                new_SSAE(target, dataset, results_dir, HTO, eta=ETA)
                                #Add the accuracy of the run to the list of accuracies
                                list_acc.append(pd.read_csv(f'{results_dir}/{condition}/bilevel_proj_l1Inftyball_acctest.csv', header=0, index_col=0, sep=';').Global.loc['Mean'])
                                #Keep the results files only if no other runs are expected
                                if args.runs >= 1 :
                                    shutil.rmtree(f"{results_dir}/{condition}")
                                else :
                                    if ETA != list_ETA[-1] :
                                        shutil.rmtree(f"{results_dir}/{condition}")
                            except Exception :
                                print(f"Error for {target}_{HTO} ! Not enough data")
                                shutil.rmtree(f"{results_dir}/{condition}")
                                pass

                        #Add the list of accuracies as a row in the accuracy dataframe
                        acc_df.loc[condition] = list_acc
                        #Save the dataframe after each condition to be able to check the advancement of the script
                        acc_df.to_csv(f'{results_dir}/accuracies.csv')
                        
                    else : pass

            #Plot accuracy = f(eta) for each condition
            eta_fig(acc_df)

            #Add a row containing the number of times each eta parameter has the max accuracy
            acc_df = pd.concat([acc_df, pd.DataFrame(acc_df.idxmax(axis=1).value_counts()).T])
            acc_df.to_csv(f'{results_dir}/accuracies.csv')

        if args.runs>1 :
            ranks_dir = f'{results_dir}/genes_ranks/'
            os.makedirs(ranks_dir)

            for run in range(args.runs) :
                for HTO in hto_names :
                    #Select all the cells with the HTO
                    full_HTO = top10k[top10k.obs['Classif_HTO'].str.contains(HTO)]
                    #Take the cells with non-targeting gRNA as negative control
                    Neg = full_HTO[full_HTO.obs['Classif_gRNA'].str.contains('Neg')].to_df().T
                    #Add the label 0 for 'control cell' for each cell in the dataframe
                    Neg.loc['Label'] = pd.Series(np.zeros(len(Neg.columns)), index=Neg.columns)
                    Neg = pd.concat([Neg.loc[['Label']], Neg.drop('Label')])

                    for target in targets : 
                        target_data = full_HTO[full_HTO.obs['Classif_gRNA'].str.contains(target)].to_df().T
                        #Add the label 1 for 'perturbded' for each cell in the dataframe
                        target_data.loc['Label'] = pd.Series(np.ones(len(target_data.columns)), index=target_data.columns)
                        target_data = pd.concat([target_data.loc[['Label']], target_data.drop('Label')])
                        if len(target_data.columns) > 0 and len(Neg.columns) > 0:    
                            #Take a random sample of both datasets to have a matching number of control and perturbed cells
                            Neg_cut = Neg.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=run)
                            target_data_cut = target_data.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=run)
                            #Concatenate the negative control and the perturbed cells counts
                            dataset = pd.concat([Neg_cut, target_data_cut], axis=1)
                            #create a dataframe that contains the expression of each gene in perturbed and control cells for the current run
                            expression_df = pd.DataFrame({'Gene' : Neg.index.to_list()[1:], 
                                                          'Perturbed_expression' : target_data.iloc[1:, :].mean(axis=1), 
                                                          'Control_expression' : Neg.iloc[1:, :].mean(axis=1)})
                            expression_df['log2_ratio'] = np.log2(expression_df['Perturbed_expression'] / expression_df['Control_expression'])
                            #Add the dataframe to the list of dataframes of the current condition
                            condition = f'{target}/{HTO}'
                            expression[condition].append(expression_df)
                            
                            try :
                                #If the best eta was determined beforehand, use it as hyperparameter
                                if args.eta :
                                    new_SSAE(target, dataset, results_dir, HTO, eta=float(acc_df.loc[condition].idxmax().split('_')[-1]))
                                else :
                                    new_SSAE(target, dataset, results_dir, HTO, eta=0.25)

                                #Store the most differentially expressed genes, their weight and their rank in a dataframe
                                scores = pd.read_csv(f'{results_dir}/{condition}/bilevel_proj_l1Inftyball_topGenes_Captum_dl_300.csv', header=0, sep=';')[['Features', 'Mean']]
                                scores['Rank'] = scores.index + 1
                                #Add the dataframe to the list of dataframes corresponding to the current condition
                                allresults[condition].append(scores)
                                #Keep the results for the last run only
                                if run != args.runs-1 :
                                    shutil.rmtree(f"{results_dir}/{condition}")
                            except Exception :
                                print(f"Error for {target}_{HTO} ! Not enough data")
                                shutil.rmtree(f"{results_dir}/{condition}")
                                pass
            
            conditions = product(targets, hto_names)
            for condition in tqdm(conditions) :

                #For each condition, make a dataframe that contains the mean weight, weight std, mean rank, rank std and log2 fold change of each gene
                condition = '/'.join(condition)
                if allresults[condition] and expression[condition]:
                    results1=allresults[condition][0]

                    for idx, results in enumerate(allresults[condition]) :
                        results.index.name = 'Features'
                        if idx != 0 :
                            results = results.reindex(allresults[condition][0].index)                        
                        
                    df = pd.DataFrame({'Gene' : results1['Features'], 
                                    'Mean_Weight' : pd.concat(allresults[condition], axis=1)['Mean'].mean(axis=1),
                                    'Weight_Std' : pd.concat(allresults[condition], axis=1)['Mean'].std(axis=1),
                                    'Mean_Rank' : pd.concat(allresults[condition], axis=1)['Rank'].mean(axis=1),
                                    'Rank_Std' : pd.concat(allresults[condition], axis=1)['Rank'].std(axis=1)})
                    df = df.sort_values(by='Mean_Rank')
                    df.index.name = 'Gene'

                    import ipdb; ipdb.set_trace()
                    df2 = pd.DataFrame({'Gene' : expression[condition][0]['Gene'].to_list(), 
                                        'log2_ratio' : pd.concat(expression[condition], axis=1)['log2_ratio'].mean(axis=1)})
                    df2.index.name='Gene'
                    df2 = df2.reindex(df.index)
                    df['log2_ratio'] = df2['log2_ratio']
                    df.index.name = 'Gene'
                    parts = condition.split('/')
                    df.to_csv(f'{results_dir}/{parts[0]}/{parts[1]}.csv', index = True)   
                else :
                    pass 
        
        else : 
            for HTO in hto_names :
                #Select all the cells with the HTO
                full_HTO = top10k[top10k.obs['Classif_HTO'].str.contains(HTO)]
                #Take the cells with non-targeting gRNA as negative control
                Neg = full_HTO[full_HTO.obs['Classif_gRNA'].str.contains('Neg')].to_df().T
                #Add the label 0 for 'control cell' for each cell in the dataframe
                Neg.loc['Label'] = pd.Series(np.zeros(len(Neg.columns)), index=Neg.columns)
                Neg = pd.concat([Neg.loc[['Label']], Neg.drop('Label')])

                for target in targets : 
                    target_data = full_HTO[full_HTO.obs['Classif_gRNA'].str.contains(target)].to_df().T
                    #Add the label 1 for 'perturbded' for each cell in the dataframe
                    target_data.loc['Label'] = pd.Series(np.ones(len(target_data.columns)), index=target_data.columns)
                    target_data = pd.concat([target_data.loc[['Label']], target_data.drop('Label')])
                    if len(target_data.columns) > 0 and len(Neg.columns) > 0:   
                        #Take a random sample of both datasets to have a matching number of control and perturbed cells 
                        Neg_cut = Neg.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=run)
                        target_data_cut = target_data.sample(n=min(len(target_data.columns), len(Neg.columns)), axis=1, random_state=run)
                        #Concatenate the negative control and the perturbed cells counts
                        dataset = pd.concat([Neg_cut, target_data_cut], axis=1)
                        
                        try :
                            #If the best eta was determined beforehand, use it as hyperparameter
                            if args.eta :
                                new_SSAE(target, dataset, results_dir, HTO, eta=float(acc_df.loc[condition].idxmax().split('_')[-1]))
                            else :
                                new_SSAE(target, dataset, results_dir, HTO, eta=0.25)
                            
                        except Exception :
                            print(f"Error for {target}_{HTO} ! Not enough data")
                            shutil.rmtree(f"{results_dir}/{condition}")
                            pass

        results_files(results_dir, targets, HTO = hto_names, pathways = args.pathways)                                

        end = time.time()

        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Running time - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


if __name__ == "__main__" :
    main()  
