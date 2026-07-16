#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=20
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=sam3_video_%j.out
#SBATCH --error=sam3_video_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.talbot@bham.ac.uk

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/marci_sam3_segmentation/sam3-background-removal
export HF_HUB_CACHE=${SAM3_PROJECT_ROOT}/hf_cache

module purge; module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

cd ${SAM3_PROJECT_ROOT}
source venvs/sam3-env/bin/activate

python sam3_bluebear_video.py