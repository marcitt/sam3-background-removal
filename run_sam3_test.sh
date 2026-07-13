#!/bin/bash
#SBATCH --account=dilucam-arme
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=45
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=sam3_test_%j.out
#SBATCH --error=sam3_test_%j.err

export SAM3_PROJECT_ROOT=/rds/projects/d/dilucam-arme/marci_sam3_segmentation/sam3-background-removal
export HF_HUB_CACHE=${SAM3_PROJECT_ROOT}/hf_cache

module purge; module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0

cd ${SAM3_PROJECT_ROOT}

# Only create the venv if it doesn't already exist
if [ ! -d "venvs/sam3-env" ]; then
    echo "Creating venv..."
    python -m venv venvs/sam3-env
    source venvs/sam3-env/bin/activate
    pip install -r requirements.txt
else
    echo "Venv already exists, activating..."
    source venvs/sam3-env/bin/activate
fi

python sam3_bluebear_test.py