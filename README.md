### Feature Evolution and Migration during Vision Transformer Training
Repository of CIKM´26 paper ["Feature Evolution and Migration during Vision Transformer Training"](https://arxiv.org/abs/2608.20134)

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

### Acknowledegement

Parts of the code used in our work was adapted from the Github repositories such as: [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main).

### Citation
```bibtex
@article{jarve2026feature,
  title={Feature Evolution and Migration during Vision Transformer Training},
  author={J{\"a}rve, Joonas and Aysel, Halil Ibrahim and Khajuria, Tarun and Kull, Meelis},
  journal={arXiv preprint arXiv:2608.20134},
  year={2026}
}
```
