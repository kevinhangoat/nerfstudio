#!/bin/bash

source /itet-stor/yuthan/net_scratch/conda/etc/profile.d/conda.sh
conda activate nerfstudio
cd /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data

CUDA_VISIBLE_DEVICES=4 ns-train depth-nerfacto --vis tensorboard \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_blocks/blocks/outputs \
    --data 11216_3_blocks/blocks/1 \
    nerfstudio-data


CUDA_VISIBLE_DEVICES=0,1 ns-train semantic-nerfacto --vis tensorboard \
    --optimizers.fields.optimizer.lr 2e-2 \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/10023_6/blocks/8 \
    --data 11216_3_debug/blocks/1 \
    semantic-data

python ../../nerfstudio/new_scripts/test_eval.py \  
    --load-config  /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_blocks/blocks/outputs/11216_3_blocks-blocks-3/depth-nerfacto/2023-02-20_214611/config.yml \    
    --data-name 11216_3_blocks/blocks/3 \
    --base-path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_blocks/blocks \
    --save-name 3_default_depth


CUDA_VISIBLE_DEVICES=1 ns-train nerfacto --vis tensorboard \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/1/outputs \
    --data 11216_3_debug/blocks/1 \
    nerfstudio-data

CUDA_VISIBLE_DEVICES=1 ns-train depth-nerfacto --vis tensorboard \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/94102_1/blocks/1/outputs \
    --data 94102_1/blocks/1 \
    nerfstudio-data

CUDA_VISIBLE_DEVICES=5 python ../../clean_nerfstudio/nerfstudio/new_scripts/test_original.py \
    --base-path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/0/0/semantic-nerfacto/2023-03-21_170252 \
    --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/0/0/semantic-nerfacto/2023-03-21_170252/config.yml \
    --save-name appearance

CUDA_VISIBLE_DEVICES=0 ns-train semantic-nerfacto --vis tensorboard \
    --max_num_iterations 70000 \
    --output_dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/10023_6/blocks/8 \
    --data /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/10023_6/blocks/8 \
    semantic-data

CUDA_VISIBLE_DEVICES=1 ns-export pointcloud \
    --load-config /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/2/2/semantic-nerfacto/2023-04-19_115204/config.yml \
    --output-dir /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11216_3_debug/blocks/2/2/semantic-nerfacto/2023-04-19_115204

python ../../clean_nerfstudio/nerfstudio/new_scripts/run_nerf_blocks.py --blocks_path /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/11201_48/blocks
python ../../clean_nerfstudio/nerfstudio/new_scripts/run_nerf_export_blocks.py /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio_data/data/10023_6/blocks