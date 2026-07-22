#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --output=logs/video.out
#SBATCH --error=logs/video.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.talbot@bham.ac.uk
#SBATCH --open-mode=truncate
 
export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/sam3-background-removal
export HF_HUB_CACHE=${SAM3_PROJECT_ROOT}/hf_cache
 
module purge; module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
 
cd ${SAM3_PROJECT_ROOT}
source venvs/sam3-env/bin/activate
 
python src/video.py