### Feature Evolution and Migration during Vision Transformer Training
Repository of CIKM´26 paper "Feature Evolution and Migration during Vision Transformer Training"

### Requirements

Requirements to run the code are given in the `requirements.txt` file.

### Data and Models

This codebase does not provide the ViT activations and checkpoints due to the storage constraints. (Tough, `2d_sae_track_independent_shapes3d.py´ calculates the activations automatically if checkpoints exist and path is given in `generate_acts_vit_fun_tiny.py`)

### Training and evaluating


1) Save the desired VIT checkpoints.
2) Run `2d_sae_track_independent_shapes3d.py` with config `sae_exp_config_base.yml` (.yml provides settings for SAE, experiment, similarity etc.) that will train the SAEs and find feature trajectories from `ref_epoch` and `ref_layer` across all the checkpoints.
3) Adapt a script `sae_tracking_exp_large.sh` to calculate all the feature trajectories.
4) To get aggregated feature evolution heatmaps, proceed, by modifying the arguments and then running `evo_aggregate_patterns.py`

### Acknowledegement

Parts of the code used in our work was adapted from the Github repositories such as: [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/main).

### Citation
```
@inproceedings{feature-evolution-migration-2026,
  title  = {Feature Evolution and Migration during Vision Transformer Training},
  year   = {2026},
  note   = {Citation details to be added}
}
```
