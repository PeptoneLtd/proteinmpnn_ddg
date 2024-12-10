# ProteinMPNN-ddG

This repository contains the code and the trained models for the prediction of logits for all positions in a protein, 
as demonstrated in the paper 
[Improving Inverse Folding models at Protein Stability Prediction without additional Training or Data](https://www.mlsb.io/papers_2024/Improving_Inverse_Folding_models_at_Protein_Stability_Prediction_without_additional_Training_or_Data.pdf), NeurIPS 2024, [MLSB](https://www.mlsb.io/).

[![DOI](https://img.shields.io/badge/DOI-10.1101/2024.06.15.599145-blue)](https://doi.org/10.1101/2024.06.15.599145)


Here we propose a novel estimator for the change in stability upon point mutation.

[ProteinMPNN](https://github.com/dauparas/ProteinMPNN) is modified to use full sequence context, and a novel tied decoding scheme is introduced to improve 
computational efficiency and enable saturation mutagenesis studies at scale.

Colab available for ProteinMPNN-ddG as described in the paper to **prioritise stability improving mutations**:   
[![Stability Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PeptoneLtd/proteinmpnn_ddg/blob/main/ProteinMPNN_ddG.ipynb)

Some follow-up work to **prioritise binding affinity improving mutations**, by a similar method (not fully benchmarked):  
[![Binding Affinity Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PeptoneLtd/proteinmpnn_ddg/blob/main/ProteinMPNN_ddG_binding.ipynb)


## Table of Contents

- [ProteinMPNN-ddG](#proteinmpnn-ddg)
  - [Table of Contents](#table-of-contents)
  - [Usage](#usage)
  - [Reproducing results](#reproducing-results)
    - [Notes](#notes)
    - [Reproducing benchmarks](#reproducing-benchmarks)
    - [Predictions to Metrics](#predictions-to-metrics)
    - [PDBs to Predictions](#pdbs-to-predictions)
    - [Reproducing the averaged single backbone related predictions](#reproducing-the-averaged-single-backbone-related-predictions)
    - [Reproducing proteome scale predictions](#reproducing-proteome-scale-predictions)
    - [Reproducing the tied and untied decoding order benchmarks relative to ProteinMPNN](#reproducing-the-tied-and-untied-decoding-order-benchmarks-relative-to-proteinmpnn)
  - [Citations](#citations)
  - [Acknowledgements](#acknowledgements)
  - [Licence](#licence)

## Usage
Run

```bash
git clone https://github.com/PeptoneLtd/proteinmpnn_ddg.git
cd proteinmpnn_ddg
```

We provide a prebuilt [docker image](https://github.com/PeptoneLtd/proteinmpnn_ddg/pkgs/container/proteinmpnn_ddg/232610583?tag=1.0.0_paper)
you can use to predict the logits of each residue in a protein.

Alternatively you can build your own docker image using the `Dockerfile`.

Example

```bash
docker run \
  --gpus '"device=0"' \
  -v $(pwd)/example/:/workspace \
  --workdir /workspace \
  ghcr.io/peptoneltd/proteinmpnn_ddg:1.0.0_base \
  python3 /app/proteinmpnn_ddg/predict.py \
    --pdb_path AF-A0A7L5GP87-F1-model_v4.pdb \
    --chains A \
    --outpath proteinmpnn_predictions.csv \
    --seed 42 \
    --model_name v_48_020
```

Multimers are supported by providing several chains separated by commas (`--chains A,B,C`). Only the first chain specified will be predicted, but it will use the context of all the other chains specified.

This model also runs in CPU-only environments, just remove the `--gpus` flag and use `ghcr.io/peptoneltd/proteinmpnn_ddg:1.0.0_base_cpu`.

## Reproducing results
Relevant data and scripts to the paper are available in `paper/`.
```bash
cd paper
```

The pdbs and predictions for all methods checked in the paper are available in `datasets/`. 

Averaged logit differences for the single residue predictions over the ProteinMPNN training set for both ProteinMPNN and
ESMif are available in `data/` as `coeff_proteinmpnn_ddg_v_48_020.csv` and `coeff_esmif.csv`.

Runtime benchmarks on a
single NVIDIA V100 16 GB GPU machine are in `data/timings_benchmark.csv`. This was produced by `scripts/benchmark_decode_last.py`.

### Notes
Optional dependency [paper] is required to run the scripts necessary for reproducing the paper. This includes a CPU only version of ESMif and various plotting and loading utilities.

ESMif has large model weights, you need to download them from [here](https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt) as `esm_if1_gvp4_t16_142M_UR50.pt` and supply the path in the relevant ESMif scripts.

### Reproducing benchmarks
`Dockerfile` contains two images, you need to build the `paper` target to reproduce results in the paper.
```bash
docker build . --tag proteinmpnn_ddg:paper --target paper
```
or use the prebuilt image: `ghcr.io/peptoneltd/proteinmpnn_ddg:1.0.0_paper`

The installed ESMif is CPU only, but is sufficient to reproduce the results quickly.

### Predictions to Metrics
From the predictions csvs in `datasets/` you can reproduce the results in Tables 3, 4 and 5, benchmarking the models on Tsuboyama, S2648 and S669:
  - Table 3: Accuracy of predictions for various models and datasets
  - Table 4: Ablation results for modifications of PROTEINMPNN
  - Table 5: Results with all mutations involving methionine removed
using `scripts/reproduce_benchmark_tables.py`. This will print out the latex code for the tables in the paper.

### PDBs to Predictions
To reproduce the csvs in `datasets` for ProteinMPNN and ESMif from the pdbs themselves you can run 
```bash
python3 scripts/predict_datasets_proteinmpnn_with_ablations.py \
  --datasets_folder datasets/ \
  --datasets tsuboyama s2648 s669
python3 scripts/predict_datasets_esmif.py \
  --datasets_folder datasets/ \
  --datasets tsuboyama s2648 s669 \
  --esmif_model_path esm_if1_gvp4_t16_142M_UR50.pt
```

### Reproducing the averaged single backbone related predictions

This includes Figures 2 and 3, Table 1, the ESMif correction for methionine coefficient of 4.18 and all correlations relating to $\delta_{X\rightarrow Y}$ in Section 2.2 of the paper.

1. Download and extract the training set of ProteinMPNN from [here](https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz) to get the folder `pdb_2021aug02/`. It's about 17GB tar file, extracted to 72GB.
2. Run `python3 scripts/compute_shifts.py --data_path pdb_2021aug02/ --outpath data/coeff_proteinmpnn_ddg_v_48_020.csv` which will run all the ProteinMPNN related predictions, print the metrics for Table 1: Improved sequence recovery metrics from tailored usage of PROTEINMPNN to stdout, and the spearman correlation between ProteinMPNN single residue context predictions and the background amino acid frequencies which we mention in the main paper text. It will also produce `training_single_structure_per_cluster_23349_structures_5615050_residues.npz` which is a compressed numpy file with all the single residue backbones used to compute metrics on. This also produces `data/coeff_proteinmpnn_ddg_v_48_020.csv` which contains $\delta_{X\rightarrow Y}^{ProteinMPNN}$.
3. Run 
```bash
python3 scripts/compute_shifts_esmif.py \
  --structure_data_path training_single_structure_per_cluster_23349_structures_5615050_residues.npz \
  --esmif_model_path esm_if1_gvp4_t16_142M_UR50.pt \
  --outpath data/coeff_esmif_raw.csv
```   
This produces `data/coeff_esmif_raw.csv` which contains $\delta_{X\rightarrow Y}^{ESMif}$.  
4. Run `python3 scripts/fit_methionine_coefficient_esmif_based_on_proteinmpnn.py --data_folder data/`, to compute the methonine coefficient, (4.18 in the paper) and the various $\delta_{X\rightarrow Y}^{ProteinMPNN}$ and $\delta_{X\rightarrow Y}^{ESMif}$ correlations mentioned.  
5. Run `python3 scripts/backbone_opening_angles.py --structure_data_path training_single_structure_per_cluster_23349_structures_5615050_residues.npz` to produce the Figure 3, the N-CA-C opening angles of the 20 amino acids over the subset of the ProteinMPNN training dataset

The plots in figure 2:  
a. $\delta_{X\rightarrow Y}$ for ProteinMPNN, amino acids ordered by frequency in training set of ProteinMPNN (`data/delta_X_Y.pdf`)  
b. Deviation from antisymmetry of $\delta_{X\rightarrow Y}$ for ProteinMPNN, $|\delta_{X\rightarrow Y}+\delta_{Y\rightarrow X}|$, amino acids ordered by degree of deviation. (`data/asymmetry_heatmap.pdf`)   
  can be reproduced from the `python3 scripts/build_proteinmpnn_delta_X_Y_plots.py --coeff_path data/coeff_proteinmpnn_ddg_v_48_020.csv --outfolder data/`

### Reproducing proteome scale predictions
> Saturation mutagenesis predictions were made for all 23,391 AlphaFold2 predicted structures of the human proteome (UP000005640_9606_HUMAN) in 30 minutes on a single V100 16GB GPU  


This was computed by downloading the human proteome from [here](https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar). Predictions were made using `scripts/predict_proteome.py` on an AWS p3.2xlarge instance which has a single V100 16GB GPU.   

We found using usual PDB parsers were slower than ProteinMPNN-ddG so use a custom parser specialised to strict PDB format, where each line is 80 characters long (using whitespace to pad if neccessary) and numpy operations are used. Predictions were for [AFDB](https://alphafold.ebi.ac.uk/) PDBs which fufilled this criteria so no pre-processing was required. Inputs were padded to minimise recompilation.

The throughput of 9,800 residues per second was calculated from the printed stdout from the script: 'Total time: 1516 seconds, approx 102us per position', this included compilation and file loading time. Through further checks, not in the script, we find we compile to 40 shapes, taking ~197 seconds, (~13\% of the total time) and loading of the PDB files accounts for ~86 seconds (~5\% of the total time). 14,850,403 residues are predicted in that 25 minute period.

Predictions were made using the PDB files downloaded and extracted using the following commands:
```bash
apt-get update && apt-get install -y aria2
aria2c -x 16 https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar
tar -xf UP000005640_9606_HUMAN_v4.tar --wildcards --no-anchored '*.pdb.gz'
gunzip *.pdb.gz
```
### Reproducing the tied and untied decoding order benchmarks relative to ProteinMPNN
The slowdown shown in Figure 1, `timings_benchmark.pdf`, and data underlying it, `timings_benchmark.csv`, can be reproduced via  
`python3 scripts/benchmark_decode_last.py --outfolder data/` 
if you hit OOM on your GPU you may reduce `--n 4096` to a lower power of two.

## Citations <a name="citations"></a>

If you use this work in your research, please cite the the relevant paper:

```bibtex
@inproceedings{proteinmpnn_ddg,
  title     = {Improving Inverse Folding models at Protein Stability Prediction without additional Training or Data},
  author    = {Dutton, Oliver and Bottaro, Sandro and Invernizzi, Michele and Redl, Istvan and Chung, Albert and Fisicaro, Carlo and Airoldi, Fabio and Ruschetta, Stefano and Henderson, Louie and Owens, Benjamin MJ and Foerch, Patrik and Tamiola, Kamil},
  booktitle = {Proceedings of the NeurIPS Workshop on Machine Learning in Structural Biology},
  year      = {2024},
  note      = {Workshop Paper},
  url       = {https://www.mlsb.io/papers_2024/Improving_Inverse_Folding_models_at_Protein_Stability_Prediction_without_additional_Training_or_Data.pdf}
}
```

# Acknowledgements
Thanks to the ColabDesign team for the JAX implementation of ProteinMPNN we use

[Sergey Ovchinnikov](https://github.com/sokrypton)  
[Shihao Feng](https://github.com/JeffSHF)  
[Justas Dauparas](https://github.com/dauparas)  
[Weikun.Wu](https://github.com/guyujun) (from Levinthal.bio)  
[Christopher Frank](https://github.com/chris-kafka)   

Thanks to the entire ESM team for ESMif.

## Licence

This source code is licensed under the MIT license found in the `LICENSE` file in the root directory of this source tree.
