#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=30
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=sam3_test_%j.out
#SBATCH --error=sam3_test_%j.err

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/marci_sam3_segmentation
export HF_HUB_CACHE=${SAM3_PROJECT_ROOT}/hf_cache

module purge; module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

source ${SAM3_PROJECT_ROOT}/venvs/sam3-env/bin/activate

cd ${SAM3_PROJECT_ROOT}
python sam3_bluebear_test.py