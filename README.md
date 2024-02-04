# CommonScenes

This is the official implementation of the paper **CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graph Diffusion**. Based on diffusion model, we propose a method to generate entire 3D scene from scene graphs, encompassing its layout and 3D geometries holistically. 


<a href="https://sites.google.com/view/commonscenes">Website</a> | <a href="https://arxiv.org/pdf/2305.16283.pdf">Arxiv</a>

[Guangyao Zhai](https://ymxlzgy.com/), [Evin Pınar Örnek](https://evinpinar.github.io/about/), [Shun-Cheng Wu](https://shunchengwu.github.io/), [Yan Di](https://shangbuhuan13.github.io/), [Federico Tombari](https://federicotombari.github.io/), [Nassir Navab](https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/), and [Benjamin Busam](https://www.cs.cit.tum.de/camp/members/benjamin-busam/)
<br/>
**NeurIPS 2023**


## Setup
### Environment
Download the code and go the folder.
```javascript
git clone https://github.com/ymxlzgy/commonscenes
cd commonscenes
```
We have tested it on Ubuntu 20.04 with Python 3.8, PyTorch 1.11.0, CUDA 11.3 and [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md#3-install-wheels-for-linux).

```javascript
conda create -n commonscenes python=3.8
conda activate commonscenes
pip install -r requirements.txt 
pip install einops omegaconf tensorboardx open3d
pip install -U openmim
mim install mmengine
mim install mmcv
mim install mmdet
mim install mmdet3d
```

To install CLIP, follow this <a href="[https://github.com/TheoDEPRELLE/AtlasNetV2](https://github.com/openai/CLIP)">OpenAI CLIP repo</a>:
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
1. Download the <a href="https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset">3D-FRONT dataset</a> from their official site.

2. Preprocess the dataset following  <a href="https://github.com/nv-tlabs/ATISS#data-preprocessing">ATISS</a>.
3. Download [3D-FUTURE-SDF](https://www.campar.in.tum.de/public_datasets/2023_commonscenes_zhai/3D-FUTURE-SDF.zip). This is processed by ourselves on the 3D-FUTURE meshes using tools in [SDFusion](https://github.com/yccyenchicheng/SDFusion).

4. Follow [this page](./SG-FRONT.md) for downloading SG-FRONT and accessing more information.
5. Create a folder named `FRONT`, and copy all files to it.

The structure should be similar like this:
```
FRONT
|--3D-FRONT
|--3D-FRONT_preprocessed (by ATISS)
|--threed_front.pkl (by ATISS)
|--3D-FRONT-texture
|--3D-FUTURE-model
|--3D-FUTURE-scene
|--3D-FUTURE-SDF
|--All SG-FRONT files (.json and .txt)
```
### Models
**Essential:** Download pretrained VQ-VAE model from [here](https://www.campar.in.tum.de/public_datasets/2023_commonscenes_zhai/vqvae_threedfront_best.pth) to the folder `scripts/checkpoint`.

**Optional:** We provide two trained models of CommonScenes available [here](https://www.campar.in.tum.de/public_datasets/2023_commonscenes_zhai/balancing.zip).
## Training

To train the models, run:

```
cd scripts
python train_3dfront.py --exp /media/ymxlzgy/Data/graphto3d_models/balancing/all --room_type livingroom --dataset /path/to/FRONT --residual True --network_type v2_full --with_SDF True --with_CLIP True --batchSize 4 --workers 4 --loadmodel False --nepoch 10000 --large False
```
`--room_type`: rooms to train, e.g., livingroom, diningroom, bedroom, and all. We train all rooms together in the implementation.

`--network_type`: the network to be trained. `v1_box` is Graph-to-Box, `v1_full` is Graph-to-3D (DeepSDF version), `v2_box` is the layout branch of CommonScenes, and `v2_full` is CommonScenes.
(Note:If you want to train `v1_full`, addtional reconstructed meshes and codes by DeepSDF should also be downloaded from [here](https://www.campar.in.tum.de/public_datasets/2023_commonscenes_zhai/DEEPSDF_reconstruction.zip), and also copy to `FRONT`).

`--with_SDF`: set to `True` if train v2_full.

`--with_CLIP `: set to `True` if train v2_box or v2_full, and not used in other cases.

`--batch_size`: the batch size for the layout branch training. (**Note:** the one for the shape branch is in `v2_full.yaml` and `v2_full_concat.yaml`. The meaning of each batch size can be found in the [Supplementary Material G.1](https://arxiv.org/pdf/2305.16283.pdf).)

`--large` : default is `False`, `True` means more concrete categories.

We provide three examples here: [Graph-to-3D (DeepSDF version)](./scripts/train_Graph-to-3D.sh), [Graph-to-Box](./scripts/train_Graph-to-Box.sh), [CommonScenes](./scripts/train_CommonScenes.sh).
The recommanded GPU is a single A100 for CommonScenes, though 3090 can also train the network with a lower batch size on the shape branch. 
## Evaluation

To evaluate the models run:
```
cd scripts
python eval_3dfront.py --exp /media/ymxlzgy/Data/graphto3d_models/balancing/all --epoch 180 --visualize False --evaluate_diversity False --num_samples 5 --gen_shape False --no_stool True
```
`--exp`: where you store the models.

`--gen_shape`: set `True` if you want to make diffusion-based shape branch work.

`--evaluate_diversity`: set `True` if you want to compute diversity. This takes a while, so it's disabled by default.

`--num_samples`: the number of experiment rounds, when evaluate the diversity.


### FID/KID
This metric aims to evaluate scene-level fidelity. To evaluate FID/KID, you need to collect ground truth top-down renderings by running `collect_gt_sdf_images.py`.

Make sure you download all the files and preprocess the 3D-FRONT. The renderings of generated scenes can be obtained inside `eval_3dfront.py`.

After obtaining both ground truth images and generated scenes renderings, run `compute_fid_scores_3dfront.py`.
### MMD/COV/1-NN
This metric aims to evaluate object-level fidelity. Please follow the implementation in [PointFlow](https://github.com/stevenygd/PointFlow). To evaluate this, you need to store object by object in the generated scenes, which can be done in `eval_3dfront.py`. 

After obtaining object meshes, run `compute_mmd_cov_1nn.py` to have the results.
## Acknowledgements
If you find this work useful in your research, please cite

```
@article{zhai2023commonscenes,
  title={CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graphs},
  author={Zhai, Guangyao and {\"O}rnek, Evin Pinar and Wu, Shun-Cheng and Di, Yan and Tombari, Federico and Navab, Nassir and Busam, Benjamin},
  journal={arXiv preprint arXiv:2305.16283},
  year={2023}
```

This repository is based on <a href="https://github.com/he-dhamo/graphto3d">Graph-to-3D</a> and <a href="https://github.com/yccyenchicheng/SDFusion">SDFusion</a>. We thank the authors for making their code available.
