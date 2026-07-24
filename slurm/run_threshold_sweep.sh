#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=60
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=threshold_sweep_%j.out
#SBATCH --error=threshold_sweep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=m.talbot@bham.ac.uk

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/sam3-background-removal
export HF_HUB_CACHE=${SAM3_PROJECT_ROOT}/hf_cache

module purge; module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

cd ${SAM3_PROJECT_ROOT}
source venvs/sam3-env/bin/activate

# Usage: sbatch run_threshold_sweep.sh IMG_5097 "violin"
python threshold_sweep.py "$1" "$2" --thresholds 0.1 0.2 0.3 0.4 --mask_thresholds 0.1 0.2 0.3 0.4