# EchoScene

This is the implementation of the submission **EchoScene: Indoor Scene Generation via Echo Diffusion on Scene Graphs**. 

**Notification:** This is a confidential code repository. Do not distribute.


## Setup
### Environment
We have tested it on Ubuntu 20.04 with Python 3.8, PyTorch 1.11.0, CUDA 11.3 and Pytorch3D.

```javascript
conda create -n echoscene python=3.8
conda activate echoscene
pip install -r requirements.txt 
pip install einops omegaconf tensorboardx open3d
```

Install mmcv-det3d:

```javascript
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmdet3d
```

Install CLIP:

```javascript
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```
Setup additional Chamfer Distance calculation for evaluation:
```javascript
cd ./extension
python setup.py install
```
### Dataset
Please follow CommonScenes to prepare the data.
### Models
Please follow CommonScenes to download the VQ-VAE model.
## Training

To train the models, run:

```
cd scripts
python train_3dfront.py --exp /path/to/exp_folder --room_type all --dataset /path/to/dataset --residual True --network_type echoscene --with_SDF True --with_CLIP True --batchSize 64 --workers 8 --loadmodel False --nepoch 10000 --large False --use_scene_rels True
```
`--room_type`: rooms to train, e.g., livingroom, diningroom, bedroom, and all. We train all rooms together in the implementation.

`--network_type`: the network to be trained. `echoscene` is EchoScene, `echolayout` is EchoLayout (single branch).

`--with_SDF`: set to `True` if train EchoScene.

`--batch_size`: the batch size for the layout branch training.

`--large` : default is `False`, `True` means more concrete categories.

## Evaluation

To evaluate the models run:
```
cd scripts
python eval_3dfront.py --exp /path/to/trained_model --dataset /path/to/dataset --epoch 2050 --visualize True --num_samples 1 --room_type all --render_type echoscene --gen_shape True
```
`--exp`: where you store the models.

`--gen_shape`: set `True` if you want to make shape branch work.

### FID/KID
This metric aims to evaluate scene-level fidelity. To evaluate FID/KID, you need to collect ground truth top-down renderings by running `collect_gt_sdf_images.py`.

Make sure you download all the files and preprocess the 3D-FRONT. The renderings of generated scenes can be obtained inside `eval_3dfront.py`.

After obtaining both ground truth images and generated scenes renderings, run `compute_fid_scores_3dfront.py`.
### MMD/COV/1-NN
This metric aims to evaluate object-level fidelity. To evaluate this, you need to store object by object in the generated scenes, which can be done in `eval_3dfront.py`. 

After obtaining object meshes, modify the path in `compute_mmd_cov_1nn.py` and run it to have the results.
## Acknowledgements
**Re-emphasize:** This is a confidential code repository. Do not distribute.
