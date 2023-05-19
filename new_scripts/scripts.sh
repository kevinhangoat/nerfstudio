#!/bin/bash

source /itet-stor/yuthan/net_scratch/conda/etc/profile.d/conda.sh
conda activate nerfstudio
cd /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data
python ../../nerfstudio/new_scripts/test_original.py \
    --load-config KITTI-360-raw-single/outputs/KITTI-360-raw-single/semantic-nerfacto/2023-04-25_114837/config.yml \
    --base-path KITTI-360-raw-single/outputs/KITTI-360-raw-single/semantic-nerfacto/2023-04-25_114837 \
    --data-name KITTI-360-raw-single \
    --save-name 30k_steps

ns-train depth-nerfacto --viewer.websocket-port 3003 \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs \
    --data KITTI-360-raw-single \
    nerfstudio-data

CUDA_VISIBLE_DEVICES=4 ns-train depth-nerfacto --viewer.websocket-port 3003 \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs \
    --data 11216_3 \
    nerfstudio-data

CUDA_VISIBLE_DEVICES=4 ns-train depth-nerfacto --vis tensorboard \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_blocks/blocks/outputs \
    --data 11216_3_blocks/blocks/1 \
    nerfstudio-data

CUDA_VISIBLE_DEVICES=6 ns-eval \
    --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/KITTI-360-raw-single/outputs/KITTI-360-raw-single/semantic-nerfacto/2023-04-25_130618/config.yml \
    --output-path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/KITTI-360-raw-single/outputs/KITTI-360-raw-single/semantic-nerfacto/2023-04-25_130618/eval.json

ns-train mipnerf \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs \
    --data KITTI-360-raw \
    --load-dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs/KITTI-360-raw/mipnerf/2023-02-06_001456/nerfstudio_models \
    nerfstudio-data

ns-render \
    --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs/11216_3/depth-nerfacto/2023-02-07_171521/config.yml

python ../../nerfstudio/new_scripts/test_eval_semantic.py \
    --base-path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs/Replica_room2/semantic-nerfacto/2023-02-28_001803 \
    --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs/Replica_room2/semantic-nerfacto/2023-02-28_001803/config.yml \
    --save-name 20k_steps

python ../../nerfstudio/new_scripts/test_eval_semantic.py \
    --base-path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs/11216_3_overlap/semantic-nerfacto/2023-02-28_103936 \
    --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs/11216_3_overlap/semantic-nerfacto/2023-02-28_103936/config.yml \
    --save-name 12k_steps
    

ns-train semantic-nerfacto --vis tensorboard \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs \
    --data Replica_room2 \
    semantic-data

ns-train semantic-nerfacto --vis tensorboard \
    --output_dir KITTI-360-raw-single/outputs \
    --data KITTI-360-raw-single \
    semantic-data

ns-train semantic-nerfacto --vis tensorboard \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_blocks/blocks/outputs \
    --data 11216_3_blocks/blocks/2 \
    semantic-data

CUDA_VISIBLE_DEVICES=4 ns-train semantic-nerfacto --vis tensorboard \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs \
    --data 11216_3_overlap \
    semantic-data

python ../../nerfstudio/new_scripts/test_eval.py \  
    --load-config  /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_blocks/blocks/outputs/11216_3_blocks-blocks-3/depth-nerfacto/2023-02-20_214611/config.yml \    
    --data-name 11216_3_blocks/blocks/3 \
    --base-path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_blocks/blocks \
    --save-name 3_default_depth

CUDA_VISIBLE_DEVICES=0 ns-train depth-nerfacto --vis tensorboard --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_blocks/blocks/outputs     --data 11216_3_blocks/blocks/2

CUDA_VISIBLE_DEVICES=0 ns-export pointcloud \
    --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/1/1/semantic-nerfacto/2023-04-15_204051/config.yml \
    --output-dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/1/1/semantic-nerfacto/2023-04-15_204051

CUDA_VISIBLE_DEVICES=4 ns-train mipnerf \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs \
    --data Replica_room2 \
    nerfstudio-data


python ../../nerfstudio/new_scripts/test_spiral.py --load-config  /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_blocks/blocks/outputs/11216_3_blocks-blocks-2/depth-nerfacto/2023-02-21_100232/config.yml --data-name 11216_3_blocks/blocks/2 --base-path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_blocks/blocks --save-name 2_default_depth_spiral

python ../../nerfstudio/new_scripts/test_eval_vanilla.py \
    --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs/Replica_room2/mipnerf/2023-02-22_162832/config.yml \
    --data-name Replica_room2 \
    --save-name default_mipNerf


CUDA_VISIBLE_DEVICES=1 python ../../clean_nerfstudio/nerfstudio/new_scripts/test_translated.py \
    --base-path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs/11216_3_overlap/semantic-nerfacto/2023-03-11_214723 \
    --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/outputs/11216_3_overlap/semantic-nerfacto/2023-03-11_214723/config.yml \
    --save-name lidar_depth

CUDA_VISIBLE_DEVICES=0 python ../../clean_nerfstudio/nerfstudio/new_scripts/test_original.py \
    --base-path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/KITTI-360-raw-single/outputs/KITTI-360-raw-single/semantic-nerfacto/2023-04-25_130618 \
    --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/KITTI-360-raw-single/outputs/KITTI-360-raw-single/semantic-nerfacto/2023-04-25_130618/config.yml \
    --save-name colmap_depth

CUDA_VISIBLE_DEVICES=1 ns-train nerfacto --vis tensorboard \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/0/outputs \
    --data 11216_3_debug/blocks/0 \
    nerfstudio-data

CUDA_VISIBLE_DEVICES=1 ns-train depth-nerfacto --vis tensorboard \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/94102_1/blocks/1/outputs \
    --data 94102_1/blocks/1 \
    nerfstudio-data

CUDA_VISIBLE_DEVICES=0 ns-train semantic-nerfacto --vis tensorboard \
    --output_dir 95115_11 \
    --data 95115_11 \
    semantic-data

python ../../clean_nerfstudio/nerfstudio/new_scripts/test_original.py \
    --base-path 95115_11/95115_11/semantic-nerfacto/2023-04-25_182140 \
    --load-config 95115_11/95115_11/semantic-nerfacto/2023-04-25_182140/config.yml \
    --save-name renders

python ../../clean_nerfstudio/nerfstudio/new_scripts/eval_with_pngs.py \
    --pred_path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/KITTI-360-raw-single/outputs/KITTI-360-raw-single/semantic-nerfacto/2023-04-25_114837/renders/lidar_depth \
    --gt_path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/KITTI-360-raw-single/npy_depth_priors/image_00/data_rect \
    --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --do_kb_crop --garg_crop