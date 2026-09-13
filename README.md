# HypSkeletonCLR-HPL: Hyperbolic SkeletonCLR with Hierarchy for Skeleton Action Representation Learning

This repository, `HypSkeletonCLR-HPL`, implements HypSkeletonCLR-HPL, a
hierarchy-aware pseudo-label extension of Hyperbolic SkeletonCLR for
skeleton-based action representation learning. The active code focuses on
SkeletonCLR-style pre-training with hyperbolic geometry, clustering,
pseudo-label supervised contrastive losses, hierarchy diagnostics, and embedding
plots for NTU RGB+D skeleton data.

The original CrosSCLR code and experiment configs are still present where useful,
but older experiment files have been moved under `config/legacy`.

The short name follows the thesis contribution naming: `C` denotes balanced
cluster/prototype assignment, `H` denotes the hierarchy-aware prototype
objective, `PL` denotes cluster-derived pseudo-label feedback, and
`HypSkeletonCLR-HPL` denotes the combined variant.

<div align=center>
    <img src="resource/figures/motivation.png", width="600" >
</div>

## Repository Layout

- `main.py` is the main entry point for pre-training, linear evaluation, and
  plotting processors.
- `config/SkeletonCLR` contains the active pre-training configs.
- `config/linear_eval` contains active linear-evaluation configs.
- `config/plotting` contains active embedding-plot configs.
- `config/legacy` contains older CrosSCLR, Euclidean, and plotting configs kept
  for reference and reproduction.
- `feeder` contains NTU dataset loaders and the 300-to-50-frame preprocessing
  script.
- `tools` contains NTU conversion, hyperbolic geometry, hierarchy utilities,
  pseudo-labeling, losses, and embedding diagnostics.
- `torchlight` is the local helper package used by the processors.
- `slurm` contains private cluster helper scripts used by the thesis author.
- `tests` contains unit tests for hierarchy, pseudo-labeling, embedding
  diagnostics, and work-directory behavior.

## Requirements

The project uses Python 3 with PyTorch and the packages listed in
`requirements.txt`. Install a PyTorch build that matches your CUDA setup before
running GPU experiments.

One known setup detail: `torchlight` is a local package and should be installed
editable. The bundled hyperbolic t-SNE package is needed when using `hyp_tsne`
embedding plots.

```bash
conda create -n hypskeletonclr-hpl python=3.11
conda activate hypskeletonclr-hpl

# Install the PyTorch build appropriate for your CUDA/runtime first.
# Example CPU-only fallback:
python -m pip install torch

python -m pip install -r requirements.txt
python -m pip install -e torchlight

# Needed for configs that use hyp_tsne plots.
python -m pip install cython
python -m pip install -e tools/hyperbolic-tsne
```

## Data Preparation

The code expects NTU RGB+D skeleton data. Put the raw `.skeleton` files somewhere
locally, then generate the intermediate 300-frame arrays:

```bash
python tools/ntu_gendata.py --data_path <path-to-nturgbd-skeletons>
```

By default this writes:

```text
data/NTU-RGB-D/
  xsub/train_data.npy
  xsub/train_label.pkl
  xsub/val_data.npy
  xsub/val_label.pkl
  xview/train_data.npy
  xview/train_label.pkl
  xview/val_data.npy
  xview/val_label.pkl
```

Then create the 50-frame normalized position and motion streams. The active
configs expect `*_position.npy` and the original `*_label.pkl` files under the
same `xsub`/`xview` root, so the most convenient setup is to write the
preprocessed arrays beside the files created by `tools/ntu_gendata.py`:

```bash
python feeder/preprocess_ntu.py \
  --dataset_path data/NTU-RGB-D \
  --out_folder data/NTU-RGB-D
```

This gives the active configs one shared dataset root:

```text
data/NTU-RGB-D/
  xsub/train_position.npy
  xsub/train_motion.npy
  xsub/train_label.pkl
  xsub/val_position.npy
  xsub/val_motion.npy
  xsub/val_label.pkl
  xview/train_position.npy
  xview/train_motion.npy
  xview/train_label.pkl
  xview/val_position.npy
  xview/val_motion.npy
  xview/val_label.pkl
```

`tools/ntu_gendata.py` skips known broken NTU60 skeleton files with
`resource/NTU-RGB-D/NTU_RGBD60_samples_with_missing_skeletons.txt` by default.
For NTU RGB+D 120, pass the NTU120 missing-skeleton list explicitly.

Most active configs use relative dataset paths such as `xview/train_position.npy`.
Run with `--dataset_root data/NTU-RGB-D`, or set `dataset_root` in the config.

## Pre-Training

Use the SkeletonCLR configs in `config/SkeletonCLR`.

```bash
python main.py pretrain_skeletonclr \
  --config config/SkeletonCLR/skeletonclr_xview.yaml \
  --dataset_root data/NTU-RGB-D
```

Use `pretrain_skeletonclr_3views` for the three-view configs:

```bash
python main.py pretrain_skeletonclr_3views \
  --config config/SkeletonCLR/skeletonclr_3views_xview.yaml \
  --dataset_root data/NTU-RGB-D
```

Hyperbolic clustering and pseudo-label SupCon variants are configured in files
such as:

```text
config/SkeletonCLR/skeletonclr_xview_hyp_clust.yaml
config/SkeletonCLR/skeletonclr_xview_hyp_pseudo_supcon.yaml
config/SkeletonCLR/skeletonclr_3views_xview_hyp_pseudo_supcon.yaml
```

Training outputs are written under `work_dir`. The processor can also derive
canonical run directories automatically using the work-directory settings in
`processor/work_dir.py`.

## Linear Evaluation

Linear evaluation configs live in `config/linear_eval`.

```bash
python main.py linear_evaluation \
  --config config/linear_eval/linear_eval_skeletonclr_xview.yaml \
  --dataset_root data/NTU-RGB-D \
  --weights <path-to-pretrained-model.pt>
```

The config may already contain a `weights` path. Passing `--weights` on the
command line overrides it.

## Plotting

Embedding diagnostics and hierarchy plots use configs in `config/plotting`.

```bash
python main.py plot_skeletonclr \
  --config config/plotting/plotting_skeletonclr_xview.yaml \
  --dataset_root data/NTU-RGB-D \
  --weights <path-to-pretrained-model.pt>
```

Configs that include `hyp_tsne` require the bundled `tools/hyperbolic-tsne`
package to be installed.

## SLURM

The `slurm` folder contains private helper scripts used by the thesis author for
specific cluster environments. They are kept in this repository for the author's
own workflow and are not intended as a portable public SLURM interface. If you
run the project on a different cluster, treat the scripts as examples and adapt
the paths, modules, partitions, dataset locations, and sync commands to your
environment.

## Tests

Run the unit tests with:

```bash
python -m pytest
```

The tests cover hierarchy utilities, pseudo-labeling behavior, embedding
diagnostics, and work-directory construction.

## Citation

If you use this repository or the thesis-specific hyperbolic SkeletonCLR changes,
please cite the accompanying bachelor thesis:

```bibtex
@bachelorsthesis{schwer2026hypskeletonclr-hpl,
  title  = {Hierarchical Pseudo-Labeling in Hyperbolic Space for Skeleton-Based Human Action Recognition},
  author = {Alexander Schwer},
  year   = {2026},
  note   = {Bachelor thesis}
}
```

## Acknowledgement

- This repository directly builds on Claudia Almeida Jordan's
  [HypSkeletonCLR_SupCon](https://github.com/ClaudiaAJ/HypSkeletonCLR_SupCon)
  repository.
- HypSkeletonCLR_SupCon itself builds on the original
  [CrosSCLR](https://github.com/LinguoLi/CrosSCLR) / SkeletonCLR codebase and
  the CVPR 2021 paper, "3D Human Action Representation Learning via Cross-View
  Consistency Pursuit".
- The original framework is based on the old version of
  [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).
- [NTURGB-D](https://github.com/shahroudy/NTURGB-D)
