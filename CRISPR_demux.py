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
    genes_names = list(pd.read_csv(cfolder + 'features.tsv.gz', sep = '\t', names = ['feat1', 'feat2', 'feat3']).feat2)
    feats_col = 2

    if args.grna :
        gfolder = args.grna
        grna_names = list(pd.read_csv(gfolder + 'features.tsv.gz', sep = '\t', names = ['Names']).Names)
        grna_names, targets = clean_guides(grna_names, args.sep)

    if args.hto :

        hfolder = args.hto
        hto_names = list(pd.read_csv(hfolder + 'features.tsv.gz', sep = '\t', names = ['Names']).Names)

    if  args.libs > 1 :
        counts_matrices = {}
        for i in range(args.libs) :

            i+=1
            print('\nLoading counts matrix n°'+str(i))

            folder = replace_digit(cfolder, i)
            matrix_name = 'matrix_' + str(i)
            matrix = sc.read_mtx(folder + 'matrix.mtx.gz').T
            #CPM normalization on all counts 
            sc.pp.normalize_total(matrix, target_sum = 1e6)
            
            #Load the cell barcodes and add them to the dataset 
            barcodes = list(pd.read_csv(folder + 'barcodes.tsv.gz', sep = '\t', names = ['Barcode']).Barcode)
            matrix.obs.index = [barcode.split('-')[0] + '-' + str(i) for barcode in barcodes]
            
            matrix.var_names = genes_names

            ID = '-'+str(i)    
            matrix.obs.index += ID

            counts_matrices[matrix_name] = matrix
    
    elif args.libs == 1 :

        counts_adata = sc.read_mtx(cfolder + 'matrix.mtx.gz').T
        #CPM normalization on all counts
        sc.pp.normalize_total(counts_adata, target_sum = 1e6)

        #Load the cell barcodes and add them to the dataset 
        barcodes = list(pd.read_csv(cfolder + 'barcodes.tsv.gz', sep = '\t', names = ['Barcode']).Barcode)
        counts_adata.obs.index = [barcode.split('-')[0] + '-' + str(i) for barcode in barcodes]

        counts_adata.var_names = genes_names


    if args.nohto :

        print('Loading gRNA counts')

        if args.grna :

            if args.libs > 1 :
                for i in range(args.libs) :

                    i+=1
                    print('\nLoading gRNA matrix n°'+str(i))

                    folder = replace_digit(gfolder, i)
                    matrix_name = 'matrix_' + str(i)
                    matrix = sc.read_mtx(folder + 'matrix.mtx.gz').T
                    #CPM normalization on all counts 
                    sc.pp.normalize_total(matrix, target_sum = 1e6)
                    
                    #Load the cell barcodes and add them to the dataset 
                    barcodes = list(pd.read_csv(folder + 'barcodes.tsv.gz', sep = '\t', names = ['Barcode']).Barcode)
                    matrix.obs.index = [barcode.split('-')[0] + '-' + str(i) for barcode in barcodes]
                    ID = '-'+str(i)    
                    matrix.obs.index += ID
                    matrix.obs = matrix.obs.set_index(counts_matrices[matrix_name].obs.index)

                    matrix.obs = matrix.to_df().astype(int)
                    matrix.obs.columns = grna_names

                    matrix = matrix[matrix.obs.sum(axis=1) > 0]

                    hashsolo(matrix, grna_names)

                    counts_matrices[matrix_name].obs['Classif_gRNA'] = matrix.obs['Classification']
                
            elif args.libs == 1 :

                grna_adata = sc.read_mtx(gfolder + 'matrix.mtx.gz').T
                sc.pp.normalize_total(grna_adata, target_sum = 1e6)

                #Load the cell barcodes and add them to the dataset 
                barcodes = list(pd.read_csv(gfolder + 'barcodes.tsv.gz', sep = '\t', names = ['Barcode']).Barcode)
                grna_adata.obs.index = [barcode.split('-')[0] + '-' + str(i) for barcode in barcodes]

                grna_adata.obs = grna_adata.obs.set_index(counts_adata.obs.index)

                grna_adata.obs = grna_adata.to_df()
                grna_adata.obs.columns = hto_names

                grna_adata = grna_adata[grna_adata.obs.sum(axis=1) > 0]

                hashsolo(grna_adata, grna_names)

                counts_adata.obs['Classif_gRNA'] = grna_adata.obs['Classification']
            
        else :

            if args.libs > 1 :

                for i in range(args.libs) :

                    i+=1
                    matrix_name = 'matrix_' + str(i)
                    folder = replace_digit(gfolder, i)
                    #Find the guides columns in the counts matrix and store them in the .obs
                    features = pd.read_csv(folder + 'features.tsv.gz', sep = '\t').iloc[:, feats_col].tolist()
                    grna_rows = find_HTOs(features)
                    grna_names = [genes_names[i] for i in grna_rows]
                    grna_names, targets = clean_guides(grna_names, args.sep)

                    for grna_row, grna_name in zip(grna_rows, grna_names) :
                        counts_matrices[matrix_name].obs[grna_name] = counts_matrices[matrix_name].to_df().T.iloc[grna_row].T.astype(int)

                    counts_matrices[matrix_name] = counts_matrices[matrix_name][:, ~counts_matrices[matrix_name].var_names.isin(grna_names)]

                    counts_matrices[matrix_name] = counts_matrices[matrix_name][counts_matrices[matrix_name].obs.sum(axis=1) > 0]

                    hashsolo(counts_matrices[matrix_name], grna_names)

                    counts_matrices[matrix_name].obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace=True)

            elif args.libs == 1 :

                features = pd.read_csv(cfolder + 'features.tsv.gz', sep = '\t').iloc[:, feats_col].tolist()
                grna_rows = find_guides(features)
                grna_names = [genes_names[i] for i in grna_rows]
                grna_names, targets = clean_guides(grna_names, args.sep)

                for grna_row, grna_name in zip(grna_rows, grna_names) :
                    counts_adata.obs[grna_name] = counts_adata.to_df().T.iloc[grna_row].T.astype(int)

                counts_adata = counts_adata[:, ~counts_adata.var_names.isin(grna_names)]

                counts_adata = counts_adata[counts_adata.obs.sum(axis=1) > 0]

                hashsolo(counts_adata, grna_names)

                counts_adata.obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace=True)


        if args.libs > 1 :

            adatas = list(counts_matrices.values())
            adatas = [adata.var_names_make_unique() for adata in adatas] 
            counts_adata = ad.concat(adatas, label='Library')       

        #Demultiplex the gRNA using the hashsolo function by Scanpy        

        if args.plot == True :

            distrib_dir = 'Distribution_plots/'

            #Check if the plots directory already exists
            if not os.path.exists(distrib_dir):
                os.makedirs(distrib_dir)
            else :
                print(f"Directory '{distrib_dir}' already exists.")

            #Put the gRNA classification results in a dataframe for plotting 
            hashed_grna = pd.DataFrame(adata.obs)

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
            plt.savefig(distrib_dir + 'gRNA_distribution.png')
        
        adata = adata[adata.obs.Classification.isin(grna_names)]
        adata.obs.index = adata.obs.Barcode
        adata.obs = adata.obs[['Classification']]

        adata.var_names = genes_names

        #Select the 10,000 most expressed genes in the counts matrix to reduce the computational cost
        print('\nSelecting 10k most expressed genes')

        gene_counts_array = adata.X.A if issparse(adata.X) else adata.X
        gene_counts = gene_counts_array.sum(axis=0)
        gene_counts_array_1d = gene_counts.A1 if issparse(gene_counts) else gene_counts.flatten()
        gene_expression_df = pd.DataFrame({
            'Gene': adata.var_names,
            'ExpressionSum': gene_counts_array_1d
        })
        
        sorted_genes = gene_expression_df.sort_values(by='ExpressionSum', ascending=False)
        top_10k_genes = sorted_genes.head(10000)['Gene'].tolist()
        top10k = adata[:, adata.var_names.isin(top_10k_genes)].copy()    

        print('\nSeparating the data by gRNA target')

        data_sep = {}

        #Take the cells without gRNA as negative control
        Neg = top10k[top10k.obs['Classification'].str.contains('Neg')].to_df().T
        #Add the label 0 for 'control cell' for each cell in the dataframe
        Neg.loc['Label'] = pd.Series(np.zeros(len(Neg.columns)), index=Neg.columns)
        Neg = pd.concat([Neg.loc[['Label']], Neg.drop('Label')])

        for target in tqdm(targets) :
            target_data = top10k[top10k.obs['Classification'].str.contains(target)].to_df().T
            #Add the label 1 for 'perturbded' for each cell in the dataframe
            target_data.loc['Label'] = pd.Series(np.ones(len(target_data.columns)), index=target_data.columns)
            target_data = pd.concat([target_data.loc[['Label']], target_data.drop('Label')])
            #Concatenate the negative control and the perturbded cells counts
            target_data = pd.concat([Neg, target_data], axis=1)

            #Add the new dataframe to the dictionary
            data_sep[target] = target_data
        
        print('\nSeparation done')

        ####################
        #Run the AutoEncoder
        ####################

        results_dir = create_results_folder()

        for target_name, target in data_sep :
            print(f"\nProcessing {target_name}")

            run_SSAE(target_name, target, results_dir)

    
        ########################################
        #Analysis of the Classification results#       
        ########################################     
        
        #Where the plots will be saved
        figs_dir = 'autoencoder/top_genes_figs'

        #Check if the plots directory already exists
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
        else :
            print(f"Directory {figs_dir} already exists.")

        #Make a dictionary for each guide    
        data_sep = {target : None for target in targets}

        print('\nPlotting the feature ranking for each condition')

        perturbed_cells = pd.DataFrame(columns = ['Condition', 'Perturbed'])


        for target in targets :

            predictions = pd.read_csv(f"{results_dir}/{target}/Labelspred_softmax.csv", sep=';', header=0).Labels
            predictions.columns = ['Name', 'Labels', 'Proba_Class1', 'Proba_Class2']
            predictions = predictions[predictions.Label == 1]
            predictions['label_pred'] = None 

            for pred in range(len(predictions)) :
                if predictions.Proba_Class1.loc[pred] > 0.5 :
                        predictions.label_pred.loc[pred] = 1
                else :
                    predictions.label_pred.loc[pred] = 0
            
            percentage_ones = (sum(predictions.label_pred) / len(predictions)) * 100

            new_row = {'Condition' : target_name, 'Perturbed' : str(percentage_ones)+' % perturbded cells' }
            perturbed_cells.loc[len(perturbed_cells)] = new_row

            genes_info = pd.DataFrame(columns = ['Gene', 'Pathways'])

            #read the results for the current target
            top_features = pd.read_csv(f"{results_dir}/{target_name+'/proj_l11ball_topGenes_Captum_dl_300.csv'}", sep = ';', header = 0)
            top_features = top_features.nlargest(30, 'Mean')

            plt.barh(top_features['Features'], top_features['Mean'], color = 'cornflowerblue')
            plt.xlabel('Mean')
            plt.ylabel('Features')
            plt.invert_yaxis()
            plt.title(f"Most Discriminant genes for {target_name} inhibition")
            plt.savefig(figs_dir + target_name + '.png')

            if args.pathways :

                #####################################################################
                #Fetch a description of the pathways in which each gene is implicated
                #####################################################################

                genes_info.Gene = top_features.nlargest(10, 'Mean').Features

                print(f"Searching for pathways for {target_name} top genes")
                for gene in trange(len(genes_info)) :
                    pathways = get_pathways(genes_info.Gene.loc[gene])

                    pathways_list = []

                    if pathways:
                        for pathway in pathways:
                            pathway_description = get_pathway_info(pathway)
                            pathways_list.append(pathway_description)
                            
                    else:
                        genes_info.Pathways.loc[gene] = 'No pathways found' 

                    genes_info.Pathways.loc[gene] = ' & '.join(pathways_list)
                
            genes_info.to_csv(f"{results_dir}/{target_name}/topGenes_Pathways.csv", index = False, sep = ';')    
            
        perturbed_cells.to_csv(f"{results_dir}/perturbed_cells.csv", index = False)

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Running time - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)) 


####################################################
####################################################
################     gRNA + HTO     ################ 
####################################################
####################################################


    else :

        print('Loading HTO counts')
        
        if args.hto :

            if args.libs > 1 :
                for i in range(args.libs) :
                    
                    i+=1
                    print('\nLoading counts matrix n°'+str(i))

                    folder = replace_digit(hfolder, i)
                    matrix_name = 'matrix_' + str(i)
                    matrix = sc.read_mtx(folder + 'matrix.mtx.gz').T
                    #CPM normalization on all counts 
                    sc.pp.normalize_total(matrix, target_sum = 1e6)
                    
                    #Load the cell barcodes and add them to the dataset 
                    barcodes = list(pd.read_csv(folder + 'barcodes.tsv.gz', sep = '\t', names = ['Barcode']).Barcode)
                    matrix.obs.index = [barcode.split('-')[0] + '-' + str(i) for barcode in barcodes]

                    ID = '-'+str(i)    
                    matrix.obs.index += ID

                    matrix.obs = matrix.obs.set_index(counts_matrices[matrix_name].obs.index)

                    matrix.obs = matrix.to_df().astype(int)
                    matrix.obs.columns = hto_names

                    hashsolo(matrix, hto_names)

                    counts_matrices[matrix_name].obs['Classif_HTO'] = matrix.obs['Classification']

            elif args.libs == 1 :

                hto_adata = sc.read_mtx(cfolder + 'matrix.mtx.gz').T
                sc.pp.normalize_total(hto_adata, target_sum = 1e6)

                #Load the cell barcodes and add them to the dataset 
                barcodes = list(pd.read_csv(hfolder + 'barcodes.tsv.gz', sep = '\t', names = ['Barcode']).Barcode)
                hto_adata.obs.index = [barcode.split('-')[0] + '-' + str(i) for barcode in barcodes]

                hto_adata.obs = hto_adata.obs.set_index(counts_adata.obs.index)

                hto_adata.obs = hto_adata.to_df().astype(int)
                hto_adata.obs.columns = hto_names

                hashsolo(hto_adata, hto_names)

                counts_adata.obs['Classif_HTO'] = hto_adata.obs['Classification']
        
        else :
            if args.libs > 1 :
                for i in range(args.libs) :

                    i+=1
                    matrix_name = 'matrix_' + str(i)
                    folder = replace_digit(cfolder, i)
                    #Find the guides columns in the counts matrix and store them in the .obs
                    features = pd.read_csv(folder + 'features.tsv.gz', sep = '\t').iloc[:, feats_col].tolist()
                    hto_rows = find_HTOs(features)
                    hto_names = [genes_names[i] for i in hto_rows]

                    for hto_row, hto_name in zip(hto_rows, hto_names) :
                        counts_matrices[matrix_name].obs[hto_name] = counts_matrices[matrix_name].to_df().T.iloc[hto_row].T.astype(int)

                    counts_matrices[matrix_name] = counts_matrices[matrix_name][:, ~counts_matrices[matrix_name].var_names.isin(hto_names)]

                    hashsolo(counts_matrices[matrix_name], hto_names)

                    counts_matrices[matrix_name].obs.rename(columns={'Classification' : 'Classif_HTO'}, inplace=True)



            elif args.libs == 1 :

                features = pd.read_csv(cfolder + 'features.tsv.gz', sep = '\t').iloc[:, feats_col].tolist()
                hto_rows = find_HTOs(features)
                hto_names = [genes_names[i] for i in hto_rows]

                for hto_row, hto_name in zip(hto_rows, hto_names) :
                    counts_adata.obs[hto_name] = counts_adata.to_df().T.iloc[hto_row].T.astype(int)

                counts_adata = counts_adata[:, ~counts_adata.var_names.isin(hto_names)]

                hashsolo(counts_adata, hto_names)

                counts_adata.obs.rename(columns={'Classification' : 'Classif_HTO'}, inplace=True)
        
        print('Loading gRNA counts')

        if args.grna :

            if args.libs > 1 :
                for i in range(args.libs) :

                    i+=1
                    print(f"\nLoading gRNA matrix n°{i}")

                    folder = replace_digit(gfolder, i)
                    matrix_name = 'matrix_' + str(i)
                    matrix = sc.read_mtx(folder + 'matrix.mtx.gz').T
                    #CPM normalization on all counts 
                    sc.pp.normalize_total(matrix, target_sum = 1e6)
                    
                    #Load the cell barcodes and add them to the dataset 
                    barcodes = list(pd.read_csv(folder + 'barcodes.tsv.gz', sep = '\t', names = ['Barcode']).Barcode)
                    matrix.obs.index = [barcode.split('-')[0] + '-' + str(i) for barcode in barcodes]

                    ID = '-'+str(i)    
                    matrix.obs.index += ID

                    matrix.obs = matrix.obs.set_index(counts_matrices[matrix_name].obs.index)

                    matrix.obs = matrix.to_df().astype(int)
                    matrix.obs.columns = grna_names

                    matrix = matrix[matrix.obs.sum(axis=1) > 0]

                    hashsolo(matrix, grna_names)

                    counts_matrices[matrix_name].obs['Classif_gRNA'] = matrix.obs['Classification']
                
            elif args.libs == 1 :

                grna_adata = sc.read_mtx(gfolder + 'matrix.mtx.gz').T
                sc.pp.normalize_total(grna_adata, target_sum = 1e6)

                #Load the cell barcodes and add them to the dataset 
                barcodes = list(pd.read_csv(gfolder + 'barcodes.tsv.gz', sep = '\t', names = ['Barcode']).Barcode)
                grna_adata.obs.index = [barcode.split('-')[0] + '-' + str(i) for barcode in barcodes]

                grna_adata.obs = grna_adata.obs.set_index(counts_adata.obs.index)

                grna_adata.obs = grna_adata.to_df()
                grna_adata.obs.columns = hto_names

                grna_adata = grna_adata[grna_adata.obs.sum(axis=1) > 0]

                hashsolo(grna_adata, grna_names)

                counts_adata.obs['Classif_gRNA'] = grna_adata.obs['Classification']
            
        else :

            if args.libs > 1 :

                for i in range(args.libs) :

                    i+=1
                    matrix_name = 'matrix_' + str(i)
                    folder = replace_digit(gfolder, i)
                    #Find the guides columns in the counts matrix and store them in the .obs
                    features = pd.read_csv(folder + 'features.tsv.gz', sep = '\t').iloc[:, feats_col].tolist()
                    grna_rows = find_HTOs(features)
                    grna_names = [genes_names[i] for i in grna_rows]
                    grna_names, targets = clean_guides(grna_names, args.sep)

                    for grna_row, grna_name in zip(grna_rows, grna_names) :
                        counts_matrices[matrix_name].obs[grna_name] = counts_matrices[matrix_name].to_df().T.iloc[grna_row].T.astype(int)

                    counts_matrices[matrix_name] = counts_matrices[matrix_name][:, ~counts_matrices[matrix_name].var_names.isin(grna_names)]

                    counts_matrices[matrix_name] = counts_matrices[matrix_name][counts_matrices[matrix_name].obs.sum(axis=1) > 0]

                    hashsolo(counts_matrices[matrix_name], grna_names)

                    counts_matrices[matrix_name].obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace=True)

            elif args.libs == 1 :

                features = pd.read_csv(cfolder + 'features.tsv.gz', sep = '\t').iloc[:, feats_col].tolist()
                grna_rows = find_guides(features)
                grna_names = [genes_names[i] for i in grna_rows]
                grna_names, targets = clean_guides(grna_names, args.sep)

                for grna_row, grna_name in zip(grna_rows, grna_names) :
                    counts_adata.obs[grna_name] = counts_adata.to_df().T.iloc[grna_row].T.astype(int)

                counts_adata = counts_adata[:, ~counts_adata.var_names.isin(grna_names)]

                counts_adata = counts_adata[counts_adata.obs.sum(axis=1) > 0]

                hashsolo(counts_adata, grna_names)

                counts_adata.obs.rename(columns={'Classification' : 'Classif_gRNA'}, inplace=True)


        if args.libs > 1 :

            adatas = list(counts_matrices.values())
            for i in range(args.libs): 
                adatas[i].var_names_make_unique()
            counts_adata = ad.concat(adatas, label='Library')

        if args.plot == True :

            distrib_dir = 'Distribution_plots/'

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
            elif args.lib == 0 :
                sns.countplot(data=hashed_data,
                        x='Classification', palette='Set2')
            plt.xlabel('HTO')
            plt.ylabel('Count')
            plt.title('Distribution of HTO classification by Hashsolo')
            plt.savefig(distrib_dir + 'HTO_distribution.png') 

            plt.figure(figsize=(25, 9))
            if args.libs > 1 :
                sns.countplot(data=hashed_data,
                        x='Classif_gRNA', hue='Library', palette='Set2')
                plt.legend(title='Libraries')
            elif args.lib == 0 :
                sns.countplot(data=hashed_data,
                        x='Classification', palette='Set2')
            plt.xlabel('Guide')
            plt.ylabel('Count')
            plt.title('Distribution of gRNA classification by Hashsolo')
            plt.savefig(distrib_dir  + 'gRNA_distribution.png')

        counts_adata.obs = counts_adata.obs[['Classif_HTO', 'Classif_gRNA']]

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
        top_10k_genes = sorted_genes.head(10000)['Gene'].tolist()
        top10k = counts_adata[:, counts_adata.var_names.isin(top_10k_genes)].copy()

        print('\nSeparating the data by HTO and gRNA target')

        #Separate the cells according to their condition (HTO)
        #Make a subdataset for each HTO
        data_sep = {key : None for key in hto_names}

        for HTO in hto_names :
            #Select all the cells with the HTO
            full_HTO = top10k[top10k.obs['Classif_HTO'].str.contains(HTO)]
            #Make a subdataset for each target gene 
            data_sep[HTO] = {target : None for target in targets}
            #Take the cells without gRNA as negative control
            data_sep[HTO]['Neg'] = full_HTO[full_HTO.obs['Classif_gRNA'].str.contains('Neg')].to_df().T
            #Add the label 0 for 'control cell' for each cell in the dataframe
            data_sep[HTO]['Neg'].loc['Label'] = pd.Series(np.zeros(len(data_sep[HTO]['Neg'].columns)), index=data_sep[HTO]['Neg'].columns)
            data_sep[HTO]['Neg'] = pd.concat([data_sep[HTO]['Neg'].loc[['Label']], data_sep[HTO]['Neg'].drop('Label')])

            for target in targets : 
                data_sep[HTO][target] = full_HTO[full_HTO.obs['Classif_gRNA'].str.contains(target)].to_df().T
                #Add the label 1 for 'perturbded' for each cell in the dataframe
                data_sep[HTO][target].loc['Label'] = pd.Series(np.ones(len(data_sep[HTO][target].columns)), index=data_sep[HTO][target].columns)
                data_sep[HTO][target] = pd.concat([data_sep[HTO][target].loc[['Label']], data_sep[HTO][target].drop('Label')])
                #Concatenate the negative control and the perturbded cells counts 
                data_sep[HTO][target] = pd.concat([data_sep[HTO]['Neg'], data_sep[HTO][target]], axis=1)   

            #Remove the Negative control from the dictionary
            data_sep[HTO].pop('Neg')

        print('Separation done\n')

        ########################## 
        #First run of AutoEncoder#
        ##########################

        results_dir = create_results_folder()

        for htoname, HTO in data_sep.items() :
            for guidename, Guide in HTO.items() :
                print(f"\nProcessing  {htoname}_{guidename}")

                run_SSAE(guidename, Guide, results_dir, htoname)

    
        ########################################
        #Analysis of the Classification results#       
        ########################################     
        
        #Where the plots will be saved
        figs_dir = 'autoencoder/top_genes_figs/'

        #Check if the plots directory already exists
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
        else :
            print(f"Directory {figs_dir} already exists.")

        #Make a dictionary for each guide    
        data_sep = {target : None for target in targets}

        print('\nPlotting the feature ranking for each condition')

        perturbed_cells = pd.DataFrame(columns = ['Condition', 'Perturbed'])


        for target_name, target in data_sep.items() :

            #Make a dictionary with each HTO condition as a key
            target = {HTO : None for HTO in hto_names}

            #Colors of each HTO subplot (add more if more conditions)
            colors = ['dimgray', 'indianred', 'sienna', 'sandybrown', 'goldenrod', 'darkkhaki', 'olive', 'lightgreen', 'seagreen']

            _, axes = plt.subplots(nrows=1, ncols=4, figsize=(25, 8))

            i = 0

            #Store the gene rankings of all HTO conditions for each target   
            for HTO_name, HTO in target.items() :

                predictions =  pd.read_csv(f"{results_dir}/{HTO_name}_{target_name}/Labelspred_softmax.csv", sep = ';', header = 0)   
                predictions.columns = ['Name', 'Labels', 'Proba_Class1', 'Proba_Class2']
                predictions = predictions[predictions.Labels == 1]
                predictions['label_pred'] = None     
                 
                percentage_ones = sum(predictions.label_pred) / len(predictions) * 100

                new_row = {'Condition' : HTO_name + '_' + target_name, 'Perturbed' : str(percentage_ones) + ' % perturbded cells' }
                perturbed_cells.loc[len(perturbed_cells)] = new_row

                genes_info = pd.DataFrame(columns = ['Gene', 'Pathways'])

                #read the results of each HTO contition for the current target
                top_features = pd.read_csv(f"{results_dir}/{HTO_name}_{target_name}/proj_l11ball_topGenes_Captum_dl_300.csv", sep = ';', header = 0)

                #Select the most discriminant genes in the dataset
                top_features = top_features.nlargest(30, 'Mean')

                if args.pathways :
    
                    ######################################################################
                    #Fetch a description of the pathways in which each gene is implicated#
                    ######################################################################

                    genes_info.Gene = top_features.nlargest(10, 'Mean').Features

                    print(f"Searching pathways for {HTO_name}_{target_name} top genes")

                    for gene in trange(len(genes_info)) :
                        pathways = get_pathways(genes_info.Gene.loc[gene])

                        pathways_list = []

                        if pathways:
                            for pathway in pathways:
                                pathway_description = get_pathway_info(pathway)
                                pathways_list.append(pathway_description)
                                
                        else:
                            genes_info.Pathways.loc[gene] = 'No pathways found' 

                        genes_info.Pathways.loc[gene] = ' & '.join(pathways_list)

                genes_info.to_csv(f"{results_dir}/{HTO_name}_{target_name}/topGenes_Pathways.csv", index = False, sep = ';')

                axes[i].barh(top_features['Features'], top_features['Mean'], color = colors[i])
                axes[i].set_xlabel('Mean')
                axes[i].set_ylabel('Features')
                axes[i].set_title(HTO_name)
                axes[i].invert_yaxis()
                i += 1

            plt.suptitle('Most discriminant genes for each condition', fontsize = 30)
            plt.tight_layout()
            plt.savefig(figs_dir + target_name + '.png')     

        perturbed_cells.to_csv(f"{results_dir}/perturbed_cells.csv", index = False)
        
        end = time.time()

        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Running time - {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


#########################################################################
#########################################################################
########     Definitions of the functions used in the script     ######## 
#########################################################################
#########################################################################

def dir_path(path : str):

        if os.path.isdir(path):
            return path
        else:
            raise NotADirectoryError(path)

def create_results_folder(base_folder="autoencoder/results_stat/"):
    numerical_suffix = max([int(folder[-3:]) for folder in os.listdir(base_folder) if folder.startswith("results_") and folder[-3:].isdigit()], default=0) + 1
    new_folder_name = f"results_{numerical_suffix:03d}"
    new_folder_path = os.path.join(base_folder, new_folder_name)
    os.makedirs(new_folder_path)
    return new_folder_path

def nb_negative_controls() :
    nb = int(input('\nNumber of NEGATIVE CONTROLS among the guides : '))
    return nb

negatives = []

def negative_controls(negatives, nb) :
    control = str(input(f"Name of the negative control n°{nb}"))
    negatives.append(control)
    return negatives 

def replace_digit(path, i):
    # Find the index of 'Lib' in the path
    lib_index = path.find('Lib')

    # Check if 'Lib' is present in the path
    if lib_index != -1:
        # Find the digit next to 'Lib'
        digit_index = lib_index + 3
        if digit_index < len(path) and path[digit_index].isdigit():
            # Replace the digit with the variable i
            modified_path = path[:digit_index] + str(i) + path[digit_index + 1:]
            return modified_path
        else:
            raise ValueError("No digit found next to 'Lib'")
    else:
        raise ValueError("'Lib' not found in the folder name.")

def get_pathways(gene, organism="hsa"):
    try:
        k = KEGG()
        pathways = k.get_pathway_by_gene(gene, organism=organism)
        return list(pathways.keys())[:3]
    except Exception :
        return None
    
def get_pathway_info(pathway_id):
    try:
        k = KEGG()
        pathway_info = k.parse(k.get(pathway_id))
        pathway_name = pathway_info.get("NAME", "No name available")
        return pathway_name[0].split(' - ')[0]
    except Exception :
        return 'No pathway found'
    
def select_names(folder) :
    names = pd.read_csv(folder + 'features.tsv.gz', sep = '\t')
    cols = ['col' + str(i) for i in range(len(names.columns))]
    print('\n', names, '\n')
    col = str(input('\nIndex of the column to select as GENE NAMES (starts from 0) : '))
    col = 'col' + str(col)
    names = list(pd.read_csv(folder + 'features.tsv.gz', sep = '\t', names = cols)[col])
    return names

def select_types(folder) :
    print(pd.read_csv(folder + 'features.tsv.gz', sep = '\t'))
    col = int(input('\nIndex of the column to select as GENE TYPES (starts from 0) : '))
    return col

def negative_controls(guides_list, i) :
    neg = int(input(f"\nIndex of the negative control n°{i} : "))
    guides_list[neg] = 'Neg-sg' + str(i)

def rm_targets(targets) :
    for index, item in enumerate(targets) :
        print(index, item)
    target_to_rm = int(input('\nIndex of Target to remove : ')) 
    targets.remove(targets[target_to_rm])

def find_guides(features) :
    grna_rows = [index for index, value in enumerate(features) if value == 'CRISPR Guide Capture']
    return [row + 1 for row in grna_rows]

def find_HTOs(features) :
    HTO_rows = [index for index, value in enumerate(features) if value == 'Antibody Capture']
    return [row + 1 for row in HTO_rows]

def check_duplicates(input_list):
    seen = {}
    result = []

    for item in input_list:
        if item in seen:
            seen[item] += 1
            new_item = f"{item}-{seen[item] + 1}"
            result.append(new_item)
        else:
            seen[item] = 0
            result.append(item + '-sg1')

    return result

def clean_guides(guides_list, sep) :
    guides_list = [guide.strip() for guide in guides_list]
    guides_list = [guide.split(sep)[0] for guide in guides_list]

    #Remove poly-A tail sequence in the guides names
    guides_list = check_duplicates(guides_list)

    for index, item in enumerate(guides_list) :
        print(index, item)
    nb = int(input('\nNumber of negative controls in the list : '))
    for i in range(nb) :
        i+=1
        negative_controls(guides_list, i)

    targets = [guide.split('-')[0] for guide in guides_list]

    targets = [target for target in targets if 'Neg' not in target]

    targets = list(set(targets))

    #Remove guides you do not want to analyze
    for index, item in enumerate(targets) :
        print(index, item)
    remove = int(input('\nNumber of targets to remove : '))
    if remove > 0 :
        for i in range(remove) :
            rm_targets(targets) 

    return guides_list, targets


if __name__ == "__main__" :
    main()