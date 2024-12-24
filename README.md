<div align="center">

# Dynamic 3D Gaussian Tracking for Graph-Based Neural Dynamics Modeling

[Project Page](https://gs-dynamics.github.io/) | [Video](https://gs-dynamics.github.io/#video) | [Arxiv](https://arxiv.org/abs/2410.18912) | [Interactive Demos](https://huggingface.co/spaces/kaifz/gs-dynamics)

</div>

<div align="center">
  <img src="assets/teaser.png" style="width:80%" />
</div>

---

## Installation

1. Setup conda environment.
```
conda create -n gs-dynamics python=3.10
conda activate gs-dynamics
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
conda install -c dglteam/label/th24_cu124 dgl  # install dgl with cuda 12.4
```

2. Install Gaussian Rasterization module.
```
mkdir third-party
cd third-party
git clone git@github.com:JonathonLuiten/diff-gaussian-rasterization-w-depth.git
cd diff-gaussian-rasterization-w-depth
python setup.py install
cd ..
```

3. Install [Grounded SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything).
```
pip install git+https://github.com/facebookresearch/segment-anything.git
git clone git@github.com:IDEA-Research/GroundingDINO.git
cd GroundingDINO
python setup.py install
cd ../../weights
gdown 1pc195KITtCCLteHkxD7mr1w-cG8XAvTs  # download DINO+SAM weights
gdown 1X-Et3-5TdSWuEUfq9gs_Ng-V-hypmLdB
gdown 1HR3O-rMv9qJoTrG6GNxi-4ZVREQzZ1Rf
```

## Running the Dynamics Model

### Getting started
We prepared an interactive demo **without requiring a robot setup**. Try it now!

1. Download checkpoints [here](https://drive.google.com/drive/folders/1N9AbTgCi9_Wd1gNeNljqml_wcFeIu_6_?usp=sharing). Unzip and put it into the ```log``` folder like this:
```
/xxx/gs-dynamics/log/rope/checkpoints/latest.pth
/xxx/gs-dynamics/log/sloth/checkpoints/latest.pth
...
```

2. Run the demo!
```
cd src
python demo.py
```

### Running in real-world (requires a robot setup)

1. Calibrate the cameras using the ChArUco calibration board.
```
cd src/real_world
python calibrate.py --calibrate  # calibrate realsense cameras
```

2. Put object in workspace, run an interactive interface to simulate interaction with the object with GS-Dynamics.
```
python gs_sim_real_gradio.py --config config/rope.py
```


## Training from Scratch

### Data preparation

Data should be stored in the folder ```{base_path}/data``` (you can define your own base path). Download our data [here](https://drive.google.com/drive/folders/1AZOXdu5MrhvuN28YzdyX8mVxNFawsN5d?usp=sharing) and [here](https://drive.google.com/file/d/1CNwUj_VYI0AIfxSpIU_QEXFtArLrG0uD/view?usp=sharing), unzip, and put it in the data folder like this:
```
{base_path}/data/episodes_rope
{base_path}/data/episodes_dog
...
```

### 3D Tracking

1. Prepare data to obtain the mask, initial point cloud, and metadata for the object.
```
cd src/tracking
python utils/obtain_mask.py --text_prompt "rope" --data_path $data_path # obtain object mask
python utils/merge_masks.py --data_path $data_path # obtain the foreground images of the object
python utils/init_pcd.py --data_path $data_path # obtain the initial point cloud
python utils/metadata.py --data_path $data_path # obtain metadata for training
```

2. Run the optimization.
```
python train_gs.py --sequence $episode --exp_name $exp_name --weight_im $weight_im --weight_rigid $weight_rigid --weight_seg $weight_seg --weight_soft_col_cons $weight_soft_col_cons --weight_bg $weight_bg --weight_iso $weight_iso --weight_rot $weight_rot --num_knn $num_knn --metadata_path $metadata_path --init_pt_cld_path=$init_pt_cld_path --scale_scene_radius=$scale_scene_radius
```

We provide a short description of the data captured from real world and the different configurations for various objects in [assets/datasets.md](assets/datasets.md).


### Training

1. Prepare the data to parse unit actions, saved in ```{base_path}/preprocessed```. We use ```config/rope.yaml``` as an example.
```
cd src
python preprocess.py --config config/rope.yaml  # preprocesses training data.
```

1. Train the dynamics model.
```
python train.py --config config/rope.yaml
```

1. Evaluate model prediction.
```
cd src
python predict.py --config config/rope.yaml --epoch latest  # evaluation
```
## Acknowledgements

We thank the authors of the following projects for making their code open source:

- [Dyn3DGS](https://github.com/JonathonLuiten/Dynamic3DGaussians)
- [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)

## Citation
```
@inproceedings{zhang2024dynamics,
    title={Dynamic 3D Gaussian Tracking for Graph-Based Neural Dynamics Modeling},
    author={Zhang, Mingtong and Zhang, Kaifeng and Li, Yunzhu},
    booktitle={8th Annual Conference on Robot Learning},
    year={2024}
}
```