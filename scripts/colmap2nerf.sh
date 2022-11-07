#!/bin/bash
#SBATCH  --output=/home/yuthan/kaolin-wisp/scripts/log/singles_mapper_%j.out
#SBATCH  --mem=50G
source /itet-stor/yuthan/net_scratch/conda/etc/profile.d/conda.sh
conda activate wisp
cd /home/yuthan/kaolin-wisp/
# python scripts/colmap2nerf.py \
#     --images /srv/beegfs02/scratch/bdd100k/data/sfm/postcode/11201/daytime/singles/1f49cce2-affba551/dense/0_dense/images \
#     --aabb_scale 16 \
#     --colmap_text /srv/beegfs02/scratch/bdd100k/data/sfm/postcode/11201/daytime/singles/1f49cce2-affba551/dense/0_dense/sparse \
#     --out /srv/beegfs02/scratch/bdd100k/data/sfm/postcode/11201/daytime/singles/1f49cce2-affba551/

python scripts/colmap2nerf.py \
    --images /scratch_net/biwidl208/yuthan/wisp_data/nerf/d389c316-c71f7a5e/images \
    --aabb_scale 16 \
    --colmap_text /scratch_net/biwidl208/yuthan/wisp_data/nerf/d389c316-c71f7a5e/sparse/orientation_aligned \
    --out /scratch_net/biwidl208/yuthan/wisp_data/nerf/d389c316-c71f7a5e/transforms_rot

python scripts/colmap2nerf.py \
    --images /scratch_net/biwidl208/yuthan/wisp_data/nerf/10002_6/images \
    --aabb_scale 16 \
    --colmap_text /scratch_net/biwidl208/yuthan/wisp_data/nerf/10002_6/sparse \
    --out /scratch_net/biwidl208/yuthan/wisp_data/nerf/10002_6

python scripts/colmap2nerf.py \
    --images /scratch_net/biwidl208/yuthan/wisp_data/nerf/10001_52_val/images \
    --aabb_scale 8 \
    --colmap_text /scratch_net/biwidl208/yuthan/wisp_data/nerf/10001_52_val/sparse \
    --out /scratch_net/biwidl208/yuthan/wisp_data/nerf/10001_52_val/transforms

python scripts/colmap2nerf.py \
    --images /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/10001_52_new/images \
    --aabb_scale 16 \
    --colmap_text /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/10001_52_new/sparse \
    --out /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/10001_52_new


python scripts/colmap2nerf.py \
    --images /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/e55efa65-7314c32b/images \
    --aabb_scale 16 \
    --colmap_text /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/e55efa65-7314c32b/sparse \
    --out /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/e55efa65-7314c32b

python scripts/colmap2nerf.py \
    --images /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/d389c316-c71f7a5e/images \
    --aabb_scale 16 \
    --colmap_text /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/d389c316-c71f7a5e/sparse \
    --out /srv/beegfs02/scratch/bdd100k/data/sfm/nerfstudio/data/d389c316-c71f7a5e
    