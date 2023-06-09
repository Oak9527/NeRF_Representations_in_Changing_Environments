# NeRF Representations in Changing Environments


## environment：
OS：ubuntu20.04.5LTS

cuda version：11.3 


1.create a conda environment:
```
conda create -n nerf_changing python=3.8
```
2.activate the conda environment:
```
conda activate nerf_changing
```
3.install torch：
```
pip install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

4.install torch-scatter：
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html
```
5.install tinycudann:
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
6.install apex：
```
pip install apex
```

7.Install core requirements 
```
pip install -r requirements.txt
```
## run
```
python train.py 
   --dataset_name blender 
   --root_dir $BLENDER_DIR 
   --N_importance 64 --img_wh 400 400 --noise_std 0 
   --num_epochs 20 --batch_size 1024 
   --optimizer adam --lr 5e-4 --lr_scheduler cosine 
   --exp_name exp 
   --data_perturb color occ 
   --encode_t 
   --encode_a
   --beta_min 0.1
```
## Citations 
The repo is heavily inspired by [Nerf_pl](https://github.com/kwea123/nerf_pl). 
