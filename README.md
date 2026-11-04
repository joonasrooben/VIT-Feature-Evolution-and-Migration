### Feature Evolution and Migration during Vision Transformer Training
Repository of CIKM´26 paper ["Feature Evolution and Migration during Vision Transformer Training"](https://arxiv.org/abs/2608.20134)

#### [DEMO](https://joonasrooben.github.io/VIT-Feature-Evolution-and-Migration/)

[How to use the DEMO](https://youtu.be/KGjSC27zJf8)

### Requirements

Requirements to run the code are given in the `requirements.txt` file.

### Data and Models

This codebase does not provide the ViT activations and checkpoints due to the storage constraints. (Tough, `sae_track_method.py´ calculates the activations automatically if checkpoints exist and path is given in `generate_acts_vit_fun_tiny.py`)

### Training and evaluating

1) Save the desired VIT checkpoints.
2) Run `sae_track_method.py` with config `sae_exp_config_base.yml` (.yml provides settings for SAE, experiment, similarity etc.) that will train the SAEs and find feature trajectories from `ref_epoch` and `ref_layer` across all the checkpoints.
3) Adapt a script `sae_tracking_exp_large.sh` to calculate all the feature trajectories.
4) To get aggregated feature evolution heatmaps, proceed, by modifying the arguments and then running `evo_aggregate_patterns.py`

### Tutorial on Shapes3D
We provide a tutorial of how to use our methodology playing out the full pipeline in a Jupyter Notebook using [Shapes3D](https://github.com/google-deepmind/3d-shapes) dataset and a smaller adjustable ViT to make it runnable on cpu. The notebook provides reference of how to run similar-scale experiments as provided in the paper.

We also provide an optional alternative to step-by-step mehtod described in the notebook via a compact script to rerun the full IN-1k pipeline using `run_full_pipeline.sbatch` that uses `run_full_pipeline.py` that is explained in the `FULL_PIPELINE_SETUP.MD`.

### Interactive results browser

[Demo](https://joonasrooben.github.io/VIT-Feature-Evolution-and-Migration/) is available for the static feature browser, exporting the Shapes3D notebook results. The browser shows feature evolution maps, top activating images, and clickable cross-SAE matches at each epoch-layer cell.

**Update:** Now the ImageNet-1k is also available for all the layers but limited amount of epochs due to the storage constraints. 

### Acknowledegement

Parts of the code used in our work was adapted from the Github repositories such as: [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main).

### Citation
```bibtex
@inproceedings{J_rve_2026,
   title={Feature Evolution and Migration during Vision Transformer Training},
   url={http://dx.doi.org/10.1145/3799682.3841045},
   DOI={10.1145/3799682.3841045},
   booktitle={Proceedings of the 35th ACM International Conference on Information and Knowledge Management},
   publisher={ACM},
   author={Järve, Joonas and Aysel, Halil Ibrahim and Khajuria, Tarun and Kull, Meelis},
   year={2026},
   month=Nov, pages={1767–1778} }
```
