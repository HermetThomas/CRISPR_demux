print('\nLoading packages')
import pandas as pd
import numpy as np
import scanpy as sc; sc.settings.verbosity = 0
from scanpy.external.pp import hashsolo
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange, tqdm
import os
import anndata as ad
import argparse
import time
from scipy.sparse import issparse
from bioservices import KEGG
from CRISPR_functions import *
from autoencoder.Run_SSAE_alldata import run_SSAE

start = time.time()

def main() :

    """
    Load, normalize, merge and demultiplex CRISPR counts matrices
    Classify the cells as perturbed or control with a Sparse Supervised AutoEncoder 

    Outputs :
    - Classification for each cell
    - AutoEncoder scores (Precision, Recall, F1...)
    - Ranking of discriminant features
    - Pathways associated to most discriminant features
    """    

    parser = argparse.ArgumentParser(
        prog = 'demux', 
        formatter_class=argparse.MetavarTypeHelpFormatter,
        description = 'Detects HTO and gRNA present in each cell and creates a classification dataframe')
    
    parser.add_argument('-libs', type = int, help = 'Number of libraries', default = 1)
    parser.add_argument('-counts', type = dir_path, help = 'Path/to/counts/library_1/', required = True)
    parser.add_argument('-grna', type = dir_path, help = 'Path/to/gRNA/library_1/')
    parser.add_argument('-hto', type = dir_path, help = 'Path/to/hto/library_1/')
    parser.add_argument('-plot', action='store_true', help = 'Add -plot to save gRNA and HTO distribution plots', default = False)
    parser.add_argument('-nohto', action='store_true', help = 'Add -nohto i you do not have HTO to demultiplex in yout dataset', default = False)
    parser.add_argument('-pathways', action='store_true', help = 'Add -pathways if you want to find pathways associated to te top genes', default = False)
    parser.add_argument('-sep', type = str, help = 'Separation used to index guides names  e.g :  - / _  : ', default = '-')

    args = parser.parse_args()

    cfolder = args.counts

    if args.grna :
        gfolder = args.grna
        features_file = next((file for file in os.listdir(gfolder) if 'features' in file), None)
        grna_names = list(pd.read_csv(gfolder + features_file, sep = '\t', names = ['Names']).Names)
        grna_names, targets = clean_guides(grna_names)

    if args.hto :
        hfolder = args.hto
        features_file = next((file for file in os.listdir(hfolder) if 'features' in file), None)
        hto_names = list(pd.read_csv(hfolder + features_file, sep = '\t', names = ['Names']).Names)
        hto_names = [hto.split('-')[0] for hto in hto_names]

    if args.libs > 1 :
        counts_matrices = {}
        for i in range(args.libs) :
            i+=1
            print(f"\nLoading counts matrix n°{i}")

            folder = replace_digit(cfolder, i)
            prefix = get_prefix(folder)
            matrix = sc.read_10x_mtx(folder, prefix=prefix, cache_compression='gzip')
            matrix.obs_names = [barcode.split('-')[0] + f"-{i}" for barcode in matrix.obs_names]
            sc.pp.normalize_total(matrix, target_sum=1e6)
            counts_matrices[f"matrix_{i}"] = matrix
        
    elif args.libs == 1 :
        print('\nLoading counts matrix')
        prefix = get_prefix(cfolder)
        counts_adata = sc.read_10x_mtx(cfolder, prefix=prefix, cache_compression='gzip')
        sc.pp.normalize_total(counts_adata, target_sum=1e6)
        counts_adata.obs.index = [barcode.split('-')[0] for barcode in counts_adata.obs.index]


    if args.nohto :

        print('Loading gRNA counts')
        
        if args.libs > 1 :
            i+=1
            for i in range(args.libs) :
                    
                if args.grna :
                    print(f"\nLoading gRNA matrix n°{i}")

                    folder = replace_digit(gfolder, i)
                    matrix_file = next((file for file in os.listdir(folder) if 'matrix' in file), None)
                    barcodes_file = next((file for file in os.listdir(folder) if 'barcodes' in file), None)
                    matrix_name = f"matrix_{i}"
                    matrix = sc.read_mtx(folder + matrix_file).T
                    #CPM normalization on all counts 
                    sc.pp.normalize_total(matrix, target_sum = 1e6)
                    
                    #Load the cell barcodes and add them to the dataset 
                    barcodes = list(pd.read_csv(folder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                    matrix.obs.index = [barcode.split('-')[0] + f"-{i}" for barcode in barcodes]

                    matrix.obs = matrix.obs.set_index(counts_matrices[matrix_name].obs.index)

                    matrix.obs = matrix.to_df().astype(int)
                    matrix.obs.columns = grna_names

                    matrix = matrix[matrix.obs.sum(axis=1) > 0]

                    hashsolo(matrix, grna_names)

                    counts_matrices[matrix_name].obs['Classif_gRNA'] = matrix.obs['Classification']

                else :
                    grna_rows = find_guides(counts_matrices[f"matrix_{i}"])
                    grna_names = [counts_matrices[f"matrix_{i}"].var_names[i] for i in grna_rows]
                    for row, name in zip(grna_rows, grna_names) :
                        counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].astype(int)

                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][:, ~counts_matrices[f"matrix_{i}"].var_names.isin(grna_names)]
                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][counts_matrices[f"matrix_{i}"].obs.sum(axis=1) > 0]

                    hashsolo(counts_matrices[f"matrix_{i}"], grna_names)

                    counts_matrices[f"matrix_{i}"].obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace = True)
            
            counts_adata = ad.concat(list(counts_matrices.values()), label = 'Library')
        
        else :

            if args.grna :
                matrix_file = next((file for file in os.listdir(gfolder) if 'matrix' in file), None)
                barcodes_file = next((file for file in os.listdir(gfolder) if 'barcodes' in file), None)
                grna_adata = sc.read_mtx(gfolder + matrix_file).T
                sc.pp.normalize_total(grna_adata, target_sum = 1e6)

                #Load the cell barcodes and add them to the dataset 
                barcodes = list(pd.read_csv(gfolder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                grna_adata.obs.index = [barcode.split('-')[0] for barcode in barcodes]

                grna_adata.obs = grna_adata.obs.set_index(counts_adata.obs.index)

                grna_adata.obs = grna_adata.to_df()
                grna_adata.obs.columns = grna_names

                grna_adata = grna_adata[grna_adata.obs.sum(axis=1) > 0]

                hashsolo(grna_adata, grna_names)

                counts_adata.obs['Classif_gRNA'] = grna_adata.obs['Classification']  

            else :

                grna_rows = find_guides(counts_adata)
                grna_names = [counts_adata.var_names[i] for i in grna_rows]
                for row, name in zip(grna_rows, grna_names) :
                    counts_adata.obs[name] = counts_adata.X[:, row].astype(int)

                counts_adata = counts_adata[:, ~counts_adata.var_names.isin(grna_names)]
                counts_adata = counts_adata[counts_adata.obs.sum(axis=1) > 0]

                hashsolo(counts_adata, grna_names)

                counts_adata.obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace = True)
        
        results_dir = create_results_folder()
        
        if args.plot :

            distrib_dir = f"{results_dir}/Distribution_plots/"

            #Check if the plots directory already exists
            if not os.path.exists(distrib_dir):
                os.makedirs(distrib_dir)
            else :
                print(f"Directory '{distrib_dir}' already exists.")

            #Put the gRNA classification results in a dataframe for plotting 
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

        counts_adata = counts_adata[counts_adata.obs.Classification.isin(grna_names)]
        counts_adata.obs.index = counts_adata.obs.Barcode
        counts_adata.obs = counts_adata.obs[['Classification']]

        #Select the 10,000 most expressed genes in the counts matrix to reduce the computational cost
        print('\nSelecting 10k most expressed genes')

        gene_counts_array = counts_adata.X.A if issparse(counts_adata.X) else counts_adata.X
        gene_counts = gene_counts_array.sum(axis=0)
        gene_counts_array_1d = gene_counts.A1 if issparse(gene_counts) else gene_counts.flatten()
        gene_expression_df = pd.DataFrame({
            'Gene': counts_adata.var_names,
            'ExpressionSum': gene_counts_array_1d
        })
        
        sorted_genes = gene_expression_df.sort_values(by='ExpressionSum', ascending=False)
        top_10k_genes = sorted_genes.head(10000)['Gene'].tolist()
        top10k =counts_adata[:, counts_adata.var_names.isin(top_10k_genes)].copy()    

        print('\nSeparating the data by gRNA target')

        data_sep = {}

        #Take the cells without perturbation as negative control
        Neg = top10k[top10k.obs['Classification'].str.contains('Neg')].to_df().T
        #Add the label 0 for 'control cell' for each cell in the dataframe
        Neg.loc['Label'] = pd.Series(np.zeros(len(Neg.columns)), index=Neg.columns)
        Neg = pd.concat([Neg.loc[['Label']], Neg.drop('Label')])

        for target in tqdm(targets) :
            target_data = top10k[top10k.obs['Classification'].str.contains(target)].to_df().T
            #Add the label 1 for 'perturbed' for each cell in the dataframe
            target_data.loc['Label'] = pd.Series(np.ones(len(target_data.columns)), index=target_data.columns)
            target_data = pd.concat([target_data.loc[['Label']], target_data.drop('Label')])
            #Concatenate the negative control and the perturbded cells counts
            Neg_cut = Neg.iloc[:, :len(target_data.columns)]
            target_data = pd.concat([Neg_cut, target_data], axis=1)

            if len(set(target_data.loc['Label'])) == 2 : 
                #Add the new dataframe to the dictionary
                data_sep[target] = target_data
        
        print('\nSeparation done')

        #######################
        #   Run AutoEncoder   #
        #######################

        for target_name, target in data_sep :
            print(f"\nProcessing {target_name}")

            run_SSAE(target_name, target, results_dir)

    
        ########################################
        #Analysis of the Classification results#       
        ########################################     
        
        results_files(results_dir, HTO = False, pathways = args.pathways)

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Running time - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) 


    else :
        if args.libs > 1 :
            for i in range(args.libs) :
                i+=1
                if args.hto :
                    print(f"\nLoading HTO matrix n°{i}")

                    folder = replace_digit(hfolder, i)
                    matrix_file = next((file for file in os.listdir(folder) if 'matrix' in file), None)
                    barcodes_file = next((file for file in os.listdir(folder) if 'barcodes' in file), None)
                    matrix_name = f"matrix_{i}"
                    matrix = sc.read_mtx(folder + matrix_file).T
                    #CPM normalization on all counts 
                    sc.pp.normalize_total(matrix, target_sum = 1e6)
                    
                    #Load the cell barcodes and add them to the dataset 
                    barcodes = list(pd.read_csv(folder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                    matrix.obs.index = [barcode.split('-')[0] + f"-{i}" for barcode in barcodes]

                    matrix.obs = matrix.obs.set_index(counts_matrices[matrix_name].obs.index)

                    matrix.obs = matrix.to_df().astype(int)
                    matrix.obs.columns = hto_names

                    matrix = matrix[matrix.obs.sum(axis=1) > 0]

                    hashsolo(matrix, hto_names)

                    counts_matrices[matrix_name].obs['Classif_HTO'] = matrix.obs['Classification']

                else :
                    hto_rows = find_HTOs(counts_matrices[f"matrix_{i}"])
                    hto_names = [counts_matrices[f"matrix_{i}"].var_names[i] for i in hto_rows]
                    for row, name in zip(hto_rows, hto_names) :
                        counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].astype(int)

                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][:, ~counts_matrices[f"matrix_{i}"].var_names.isin(hto_names)]
                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][counts_matrices[f"matrix_{i}"].obs.sum(axis=1) > 0]

                    hashsolo(counts_matrices[f"matrix_{i}"], hto_names)

                    counts_matrices[f"matrix_{i}"].obs.rename(columns={'Classification' : 'Classif_HTO'}, inplace = True)
                
                if args.grna :
                    print(f"\nLoading gRNA matrix n°{i}")

                    folder = replace_digit(gfolder, i)
                    matrix_file = next((file for file in os.listdir(folder) if 'matrix' in file), None)
                    barcodes_file = next((file for file in os.listdir(folder) if 'barcodes' in file), None)
                    matrix_name = f"matrix_{i}"
                    matrix = sc.read_mtx(folder + matrix_file).T
                    #CPM normalization on all counts 
                    sc.pp.normalize_total(matrix, target_sum = 1e6)
                    
                    #Load the cell barcodes and add them to the dataset 
                    barcodes = list(pd.read_csv(folder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                    matrix.obs.index = [barcode.split('-')[0] + f"-{i}" for barcode in barcodes]

                    matrix.obs = matrix.obs.set_index(counts_matrices[matrix_name].obs.index)

                    matrix.obs = matrix.to_df().astype(int)
                    matrix.obs.columns = grna_names

                    matrix = matrix[matrix.obs.sum(axis=1) > 0]

                    hashsolo(matrix, grna_names)

                    counts_matrices[matrix_name].obs['Classif_gRNA'] = matrix.obs['Classification']    

                else :
                    grna_rows = find_guides(counts_matrices[f"matrix_{i}"])
                    grna_names = [counts_matrices[f"matrix_{i}"].var_names[i] for i in grna_rows]
                    for row, name in zip(grna_rows, grna_names) :
                        counts_matrices[f"matrix_{i}"].obs[name] = counts_matrices[f"matrix_{i}"].X[:, row].astype(int)

                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][:, ~counts_matrices[f"matrix_{i}"].var_names.isin(grna_names)]
                    counts_matrices[f"matrix_{i}"] = counts_matrices[f"matrix_{i}"][counts_matrices[f"matrix_{i}"].obs.sum(axis=1) > 0]

                    hashsolo(counts_matrices[f"matrix_{i}"], grna_names)

                    counts_matrices[f"matrix_{i}"].obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace = True)
            
            counts_adata = ad.concat(list(counts_matrices.values()), label = 'Library')

        elif args.libs == 1 :
            if args.hto :
                matrix_file = next((file for file in os.listdir(hfolder) if 'matrix' in file), None)
                barcodes_file = next((file for file in os.listdir(hfolder) if 'barcodes' in file), None)
                hto_adata = sc.read_mtx(hfolder + matrix_file).T
                sc.pp.normalize_total(hto_adata, target_sum = 1e6)

                #Load the cell barcodes and add them to the dataset 
                barcodes = list(pd.read_csv(hfolder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                hto_adata.obs.index = [barcode.split('-')[0] for barcode in barcodes]
                counts_adata = counts_adata[counts_adata.obs.index.isin(hto_adata.obs.index)]
                hto_adata.obs = hto_adata.obs.set_index(counts_adata.obs.index)

                hto_adata.obs = hto_adata.to_df().astype(int)
                hto_adata.obs.columns = hto_names

                hto_adata = hto_adata[hto_adata.obs.sum(axis=1) > 0]

                hashsolo(hto_adata, hto_names)

                counts_adata.obs['Classif_HTO'] = hto_adata.obs['Classification']     

            else :
                hto_rows = find_HTOs(counts_adata)
                hto_names = [counts_adata.var_names[i] for i in hto_rows]
                for row, name in zip(hto_rows, hto_names) :
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

                #Load the cell barcodes and add them to the dataset 
                barcodes = list(pd.read_csv(gfolder + barcodes_file, sep = '\t', names = ['Barcode']).Barcode)
                grna_adata.obs.index = [barcode.split('-')[0] for barcode in barcodes]

                grna_adata = grna_adata[grna_adata.obs.index.isin(counts_adata.obs.index)]
                grna_adata.obs = grna_adata.obs.set_index(counts_adata.obs.index)

                grna_adata.obs = grna_adata.to_df().astype(int)
                grna_adata.obs.columns = grna_names

                grna_adata = grna_adata[grna_adata.obs.sum(axis=1) > 0]

                hashsolo(grna_adata, grna_names)

                counts_adata.obs['Classif_gRNA'] = grna_adata.obs['Classification'] 
            
            else :
                grna_rows = find_guides(counts_adata)
                grna_names = [counts_adata.var_names[i] for i in grna_rows]
                for row, name in zip(grna_rows, grna_names) :
                    counts_adata.obs[name] = counts_adata.X[:, row].astype(int)

                counts_adata = counts_adata[:, ~counts_adata.var_names.isin(grna_names)]
                counts_adata = counts_adata[counts_adata.obs.sum(axis=1) > 0]

                hashsolo(counts_adata, grna_names)

                counts_adata.obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace = True)
    
        results_dir = create_results_folder()

        if args.plot == True :

            distrib_dir = f"{results_dir}/Distribution_plots/"

            #Check if the plots directory already exists
            if not os.path.exists(distrib_dir):
                os.makedirs(distrib_dir)
            else :
                print(f"Directory {distrib_dir} already exists.")

            #Put the HTO classification results in a dataframe for plotting
            
            hashed_data = pd.DataFrame(counts_adata.obs)

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

        counts_adata.obs = counts_adata.obs[['Classif_HTO', 'Classif_gRNA']]

        if 'unmapped' in hto_names :
            hto_names.remove('unmapped')

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

        print('\nSeparating the data by HTO and gRNA target')

        #Separate the cells according to their condition (HTO)
        #Make a subdataset for each HTO
        data_sep = {}

        for HTO in hto_names :
            #Select all the cells with the HTO
            full_HTO = top10k[top10k.obs['Classif_HTO'].str.contains(HTO)]
            #Make a subdataset for each target gene 
            data_sep[HTO] = {}
            #Take the cells without gRNA as negative control
            Neg = full_HTO[full_HTO.obs['Classif_gRNA'].str.contains('Neg')].to_df().T
            #Add the label 0 for 'control cell' for each cell in the dataframe
            Neg.loc['Label'] = pd.Series(np.zeros(len(Neg.columns)), index=Neg.columns)
            Neg = pd.concat([Neg.loc[['Label']], Neg.drop('Label')])

            for target in targets : 
                target_data = full_HTO[full_HTO.obs['Classif_gRNA'].str.contains(target)].to_df().T
                #Add the label 1 for 'perturbded' for each cell in the dataframe
                target_data.loc['Label'] = pd.Series(np.ones(len(target_data.columns)), index=target_data.columns)
                target_data = pd.concat([target_data.loc[['Label']], target_data.drop('Label')])
                #Concatenate the negative control and the perturbded cells counts 
                Neg_cut = Neg.iloc[:, :len(target_data.columns)]
                target_data = pd.concat([Neg_cut, target_data], axis=1) 

                if len(set(target_data.loc['Label'])) == 2 :
                    data_sep[HTO][target] = target_data

        print('Separation done\n')

        #######################
        #   Run AutoEncoder   #
        #######################

        for htoname, HTO in data_sep.items() :
            for guidename, Guide in HTO.items() :
                print(f"\nProcessing  {htoname}_{guidename}")

                run_SSAE(guidename, Guide, results_dir, htoname)

    
        results_files(results_dir, pathways = args.pathways)
        
        end = time.time()

        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Running time - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


if __name__ == "__main__" :
    main()    
