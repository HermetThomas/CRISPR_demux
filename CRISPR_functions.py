import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import os
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


def results_files(results_dir, HTO = True, pathways = False) :

    perturbed_cells = pd.DataFrame(columns = ['Condition', 'Precision'])

    folders = os.listdir(results_dir)

    if HTO == True :
        for target in tqdm(folders) :

            if target in ["perturbed_cells.csv", "Distribution_plots"] :
                pass

            else :
                htos = sorted(os.listdir(f"{results_dir}/{target}"))
                colors = ['dimgray', 'indianred', 'sienna', 'sandybrown', 'goldenrod', 'darkkhaki', 'olive', 'lightgreen', 'seagreen']
                _, axes = plt.subplots(nrows=1, ncols=4, figsize=(21, 11))
                i = 0
                target_scores = []
                for hto in htos :                    

                    if hto == "top_genes.png" :
                        pass
                    else :
                        top_genes = pd.read_csv(f"{results_dir}/{target}/{hto}/proj_l11ball_topGenes_Captum_dl_300.csv", sep = ';', header = 0)
                        top = top_genes.nlargest(30, 'Mean')       
                        axes[i].barh(top['Features'], top['Mean'], color = colors[i])
                        axes[i].set_xlabel('Mean')
                        axes[i].set_ylabel('Features')
                        axes[i].set_title(hto)
                        axes[i].invert_yaxis()
                        i+=1

                        predictions =  pd.read_csv(f"{results_dir}/{target}/{hto}/Labelspred_softmax.csv", sep = ';', header = 0)   
                        predictions.columns = ['Name', 'Label', 'Proba_Class0', 'Proba_Class1']
                        pos = predictions[predictions.Proba_Class1 > 0.5]
                        true_pos = pos[pos.Label == 1]
                        false_pos = pos[pos.Label == 0]
                        precision = len(true_pos) / (len(true_pos) + len(false_pos))
                        target_scores.append(precision)
                        new_row = {'Condition' : f"{target}_{hto}", 'Precision' : f"{precision} % precision" }
                        perturbed_cells.loc[len(perturbed_cells)] = new_row

                        genes_info = pd.DataFrame(columns = ['Gene', 'Pathways'])

                        if pathways == True :

                            genes_info.Gene = top.nlargest(10, 'Mean').Features

                            print(f"Searching pathways for {target}_{hto} top genes")

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
                            genes_info.to_csv(f"{results_dir}/{target}/{hto}/topGenes_pathways.csv", index = False, sep = ';')
                    plt.suptitle(f"Most discriminant Features for {target}", fontsize = 30)
                    plt.tight_layout()
                    plt.savefig(f"{results_dir}/{target}/top_genes.png")
                
                mean_target = sum(target_scores) / len(target_scores)        
                new_row = {'Condition' : f"ALL {target}", 'Precision' : f"{mean_target}  % mean precision"} 
                perturbed_cells.loc[len(perturbed_cells)] = new_row
                perturbed_cells.loc[len(perturbed_cells)] = {'Condition' : "----", 'Precision' : "----"}
        perturbed_cells.to_csv(f"{results_dir}/perturbed_cells.csv", index = False, sep = ';')

    else :
        for target in tqdm(folders) :
            
            if target in ["perturbed_cells.csv", "Distribution_plots"] :
                pass
            else :
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

                classification = pd.read_csv(f"{results_dir}{target}/Labelspred_softmax.csv", sep = ';')
                classification.columns = ['Name', 'Label', 'Proba_Class0', 'Proba_Class1']

                pos = classification[classification.Proba_Class1 > 0.5]
                true_pos = pos[pos.Label == 1]
                false_pos = pos[pos.Label == 0]
                precision = len(true_pos) / (len(true_pos)+len(false_pos))
                perturbed_cells.loc[len(perturbed_cells)] = {'Condition':target, 'Precision':f"{true_pos} % mean precision"}
        perturbed_cells.to_csv(f"{results_dir}/perturbed_cells.csv", index = False, sep = ';')
            
