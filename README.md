# CRISPR_demux

This repository contains a Python script used to exctract discriminating genes between the various conditions in your dataset.

The script uses Hashsolo by Scanpy to load Cellranger counts matrices and perform the demultiplexing of the X HTOs and Y gRNAs in the dataset.
It then separates the dataset into X*Y subsets according to the HTO and gRNA in each cell.
The top discriminating features between each condition and the negative control are found through the use of a supervised AutoEncoder and stored in a dataframe. The biological pathways in which these genes are implicated are fetched in the KEGG database with the BioServices package.

The AutoEncoder used to perform the classification was developed by Barlaud M. and Guyard F. and published in the following papers :

Michel Barlaud, Frédéric Guyard : *Learning sparse deep neural networks using efficient structured projections on convex constraints for green ai.* ICPR 2020 Milan Italy (2020) doi : 10.1109/ICPR48806.2021.9412162

and 

David Chardin, Cyprien Gille, Thierry Pourcher and Michel Barlaud : *Accurate Diagnosis with a confidence score using the latent space of a new Supervised Autoencoder for clinical metabolomic studies.* BMC Informatics 2022 doi: 10.1186/s12859-022-04900-x

# Table of contents 

1. [Repository Content](repository-content)
2. [Installation](#installation)
3. [Input format](#input-format)
4. [Usage](#usage)


## **Repository Content**
|File/Folder|Description|
|:-:|:-:|
|CRISPR_demux.py|Main script to launch|
|CRISPR_functions.py|Definition of functions used in the main script|
|autoencoder/|Contains the AutoEncoder sript and functions it calls|
|requirements.txt|Python packages required to run the script|

## **Installation**

Use conda to create a python environment contain all the required dependencies

Clone the repository on yout device
```{bash}
git clone https://github.com/HermetThomas/CRISPR_demux.git
cd CRISPR_demux
conda env create -f CRISPR_env.yml
conda activate CRISPR_env
```

## **Input format**

If you have multiple libraries, homologs are in the same directory with identical names except their index from 1 to n &rarr; Lib1, Lib2, ... Libn

> [!WARNING]
> Only the digit directly next to 'Lib' will be automatically modified.

Example :
* data/
   * Counts_Lib1/
   * Counts_Lib2/
   * gRNA_UMI_Lib1_06-24/
   * gRNA_UMI_Lib2_06-24/ 
   * HTO_Lib1_03-04-2024_01/
   * HTO_Lib2_03-04-2024_02/

The different libraries need to contain the same files as the following :

* Counts_Lib1/
   * matrix.mtx.gz  
   * barcodes.tsv.gz
   * features.tsv.gz

* Counts_Lib2/
   * matrix.mtx.gz  
   * barcodes.tsv.gz
   * features.tsv.gz
 
The files names can contain a common prefix :

  * Exp1_CRISPR_matrix.mtx.gz
  * Exp1_CRISPR_barcodes.tsv.gz
  * Exp1_CRISPR_features.tsv.gz


### **Matrix.mtx**

Counts matrix &rarr; **X cells * Y genes**

### **barcodes.tsv**

.tsv file containing the cell barcode associated to each cell in the counts matrix
&rarr; **X rows**

### **features.tsv**

.tsv file containing the genes names and types &rarr; **Y rows**

## **Usage**



### **If the HTO counts and gRNA counts are in the main counts matrix**

The HTO and gRNA counts are found by using the gene types in the features.tsv.gz file.

'CRISPR Guide Capture' &rarr; gRNA

'Antibody Capture' &rarr; HTO 

```{bash}
python3 CRISPR_demux.py 
   -libs number_of_libraries
   -counts /path/to/first/counts_library/
```

*Add '-plot' if you want to plot the distribution of gRNAs and HTOs*

*Add '-pathways' if you want to find pathways associated to the most disciminant genes*



### **If the HTO counts or gRNA counts are in a separate counts matrix**

Add the path to the first library of HTO counts / gRNA counts / both

```{bash}
python3 CRISPR_demux.py 
   -counts /path/to/first/counts_library/
   -grna /path/to/first/gRNA_library/
   -hto /path/to/first/HTO_library/
```


### **If you do not have HTOs to demultiplex**

Add '-nohto' to the command line 

```{bash}
python3 CRISR_demux.py
   -counts /path/to/first/counts_library/
   -nohto
```
