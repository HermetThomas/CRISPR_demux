import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
import math
import time
from itertools import product
from scipy import interpolate
from bioservices import KEGG


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

def get_prefix(path) :
    parts = next((file for file in os.listdir(path) if 'matrix' in file), None).split('matrix', 1)
    if len(parts) > 1 :
        return parts[0].strip()
    else :
        return ""

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

def find_guides(adata) :
    grna_rows = [index for index, value in enumerate(adata.var.feature_types) if value == 'CRISPR Guide Capture']
    return [row for row in grna_rows]

def find_HTOs(adata) :
    HTO_rows = [index for index, value in enumerate(adata.var.feature_types) if value == 'Antibody Capture']
    return [row for row in HTO_rows]

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
            result.append(item + '-1')

    return result

def clean_guides(guides_list, neg) :
    guides_list = [guide.strip() for guide in guides_list]
    guides_list = [guide.split('-')[0] for guide in guides_list]
    guides_list = [guide.split('_')[0] for guide in guides_list]

    #Remove poly-A tail sequence in the guides names
    guides_list = check_duplicates(guides_list)

    if neg == None :
        for index, item in enumerate(guides_list) :
            print(index, item)
        neg = input('index of the negative controls (separated by a space) :')
        for idx in neg.split() :
            idx = int(idx)
            guides_list[idx] = f"Neg-sg{idx}"
    
    else :
        for neg in neg :
            for i in range(len(guides_list)) :
                if neg in guides_list[i] : 
                    guides_list[i] = f"Neg-sg{i}"

    targets = [guide.split('-')[0] for guide in guides_list]
    targets = [target for target in targets if 'Neg' not in target]
    targets = [target for target in targets if 'unmapped' not in target]
    targets = list(set(targets)) 

    return guides_list, targets


def results_files(results_dir, targets_names, HTO=None, pathways = False) :

    perturbed_cells = pd.DataFrame(columns = ['Condition', 'Precision'])

    accuracies = pd.DataFrame(columns = ['Condition','Accuracy'])

    colors = ['#4e79a7', '#f28e2b', '#59a14f', '#e15759', '#9c755f', '#bab0ac', '#ff9da7', '#9c9ede', '#77aadd', '#99ddff', '#44bb99', '#55cc55', '#bbeebb', '#ffcc66', '#ff9966', '#ff88cc', '#cc99ff', '#778899', '#88aa99', '#ccbbaa']
    
    targets = [target for target in targets_names if target in os.listdir(results_dir)]

    if HTO != None :

        for target in tqdm(targets) :

            htos = [hto for hto in HTO if hto in os.listdir(f'{results_dir}/{target}')]
            _, axes = plt.subplots(nrows=1, ncols=len(htos), figsize=(21, 11))
            target_scores = []

            for HTO_idx,HTO in enumerate(htos):    
                accuracy = pd.read_csv(f'{results_dir}/{target}/{HTO}/proj_l11ball_acctest.csv',sep = ';',index_col=0,header=0).Global.loc['Mean']
                accuracies.loc[len(accuracies)] = {'Condition' : f'{target}_{HTO}', 'Accuracy' : accuracy}
                top_genes = pd.read_csv(f"{results_dir}/{target}/{HTO}/proj_l11ball_topGenes_Captum_dl_300.csv", sep = ';', header = 0)
                top = top_genes.nlargest(30, 'Mean')       
                axes[HTO_idx].barh(top['Features'], top['Mean'], color = colors[HTO_idx])
                axes[HTO_idx].set_xlabel('Mean')
                axes[HTO_idx].set_ylabel('Features')
                axes[HTO_idx].set_title(HTO)
                axes[HTO_idx].invert_yaxis()


                predictions =  pd.read_csv(f"{results_dir}/{target}/{HTO}/Labelspred_softmax.csv", sep = ';', header = 0)   
                predictions.columns = ['Name', 'Label', 'Proba_Class0', 'Proba_Class1']

                pos = predictions[predictions.Proba_Class1 > 0.5]
                true_pos = len(pos[pos.Label == 1])
                false_pos = len(pos[pos.Label == 0])

                neg = predictions[predictions.Proba_Class1 < 0.5]
                true_neg = len(neg[neg.Label == 0])
                false_neg = len(neg[neg.Label == 1])

                classif,classif_fig=plt.subplots()
                classif_fig.bar([true_pos,false_pos,true_neg,false_neg], ['True Positive','False Positive','True Negative','False Negative'], colors=['green','red','green','red'])
                classif.savefig(f"{results_dir}/{target}/{HTO}/classification.png")
                
                precision = pd.read_csv(f'{results_dir}/{target}/{HTO}/proj_l11ball_auctest.csv', header=0,index_col=0,sep=';').Precision.loc['Mean']
                target_scores.append(precision)
                perturbed_cells.loc[len(perturbed_cells)] = {'Condition' : f"{target}_{HTO}", 'Precision' : f"{precision} % precision" }

                if pathways == True :
                    
                    genes_info = pd.DataFrame(columns = ['Gene', 'Pathways'])

                    genes_info.Gene = top.nlargest(10, 'Mean').Features

                    print(f"Searching pathways for {target}_{HTO} top genes")

                    for gene in trange(len(genes_info)) :
                        pathways_info = get_pathways(genes_info.Gene.loc[gene])

                        pathways_list = []

                        if pathways_info:
                            for pathway in pathways_info:
                                pathway_description = get_pathway_info(pathway)
                                pathways_list.append(pathway_description)
                                
                        else:
                            genes_info.Pathways.loc[gene] = 'No pathways found' 

                        genes_info.Pathways.loc[gene] = ' & '.join(pathways_list)

                    genes_info.to_csv(f"{results_dir}/{target}/{HTO}/topGenes_pathways.csv", index = False, sep = ';')
            plt.suptitle(f"Most discriminant Features for {target}", fontsize = 30)
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{target}/top_genes.png")
            
            mean_target = sum(target_scores) / len(target_scores)        
            perturbed_cells.loc[len(perturbed_cells)] = {'Condition' : f"ALL {target}", 'Precision' : f"{mean_target}  % mean precision"} 
            perturbed_cells.loc[len(perturbed_cells)] = {'Condition' : "---------", 'Precision' : "---------------------"}
        perturbed_cells.to_csv(f"{results_dir}/perturbed_cells.csv", index = False, sep = ';')

    else :
        for target in tqdm(targets) :
            
            top_genes = pd.read_csv(f"{results_dir}{target}/proj_l11ball_topGenes_Captum_dl_300.csv", sep = ';', header = 0)
            top = top_genes.nlargest(30, 'Mean')

            plt.figure(figsize=(22, 12))

            plt.barh(top['Features'], top['Mean'], color = "royalblue")
            plt.xlabel("Mean Dicrimination Weight")
            plt.ylabel("Gene")
            plt.title(f"Most Discriminant genes for {target}")
            plt.gca().invert_yaxis()

            if not os.path.exists(f"{results_dir}{target}") :
                os.makedirs(f"{results_dir}{target}")
            
            plt.savefig(f"{results_dir}{target}/top_genes_fig.png")

            if pathways == True :
                genes_info = pd.DataFrame(columns = ['Gene', 'Pathways'])

                genes_info.Gene = top_genes.nlargest(10, 'Mean').Features

                print(f"Searching for pathways for {target} top genes")
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
                
                genes_info.to_csv(f"{results_dir}/{target}/topGenes_Pathways.csv", index = False, sep = ';')

            predictions = pd.read_csv(f"{results_dir}{target}/Labelspred_softmax.csv", sep = ';')
            predictions.columns = ['Name', 'Label', 'Proba_Class0', 'Proba_Class1']

            pos = predictions[predictions.Proba_Class1 > 0.5]
            true_pos = pos[pos.Label == 1]
            false_pos = pos[pos.Label == 0]

            neg = predictions[predictions.Proba_Class1 < 0.5]
            true_neg = len(neg[neg.Label == 0])
            false_neg = len(neg[neg.Label == 1])

            classif,classif_fig=plt.subplots()
            classif_fig.bar([true_pos,false_pos,true_neg,false_neg], ['True Positive','False Positive','True Negative','False Negative'], colors=['green','red','green','red'])
            classif.savefig(f"{results_dir}/{target}/classification.png")


            precision = pd.read_csv(f'{results_dir}/{target}/proj_l11ball_auctest.csv', header=0,index_col=0,sep=';').Precision.loc['Mean']
            perturbed_cells.loc[len(perturbed_cells)] = {'Condition':target, 'Precision':f"{precision} % mean precision"}
        perturbed_cells.to_csv(f"{results_dir}/perturbed_cells.csv", index = False, sep = ';')

def getcloser(x, yfit, xnew):
    idx = (np.abs(xnew - x)).argmin()
    return yfit[idx]

def make_pchip_graph(x, y, npoints=300):
    pchip = interpolate.PchipInterpolator(x, y)
    xnew = np.linspace(min(x), max(x), num=npoints)
    yfit = pchip(xnew)
    plt.plot(xnew, yfit)
    return (xnew, yfit)

def eta_fig(dataframe, results_dir) :

    dataframe.index = [condition.strip() for condition in dataframe.index]
    conditions = list(dataframe.index)
    conditions.remove('count')
    list_eta = [float(eta.split('_')[-1]) for eta in dataframe.columns]

    for condition in conditions :
        accuracy = np.array(list(dataframe.loc[condition]))
        RadiusC = np.array(list_eta)

        radiusToUse= RadiusC
        accToUse=accuracy

        xnew, yfit = make_pchip_graph(radiusToUse,accToUse)

        plt.title("HIF2   ")  # titre général
        plt.xlabel("Parameter $\eta$")                         # abcisses
        plt.ylabel("Accuracy")                      # ordonnées

        a = min(radiusToUse)
        b = max(radiusToUse)  #  
        tol = 0.01  # impact the execution time
        r= 0.5*(3-math.sqrt(5))

        start_time = time.time()

        while (b-a > tol):
            c = a + r*(b-a);
            d = b - r*(b-a);
            if(getcloser(c, yfit,xnew) > getcloser(d, yfit,xnew)):
                b = d
            else:
                a = c

        parameter = getcloser(c,xnew, xnew)

        end_time = time.time()

        # Calculate and print the execution time
        execution_time = (end_time - start_time)*1000

        print(f"Execution time: {execution_time} ms")


        print("Golden Section Optimal parameter", parameter)
        print("Golden section Maximum accuracy ", getcloser(c,yfit,xnew))


        plt.axvline(x=parameter, color='g', linestyle='--', label='Optimal Parameter')


        # Display the legend
        plt.legend()
        parts = condition.split('/')
        # Show the plot
        plt.savefig(f'{results_dir}/{parts[0]}/ETA_curve_{parts[1]}.png')
        plt.close()
