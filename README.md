# sam3-background-removal

This repo uses huggingface transformers and Sam3Processor, Sam3Model
- model = Sam3Model using facebook/sam3
- processor = Sam3Processor usng facebook/sam3

Use of SAM3 requires access request:
https://huggingface.co/facebook/sam3

```
cd /rds/projects/d/dilucam-arme/marci_sam3_segmentation
git clone git@github.com:marcitt/sam3-background-removal.git
```

```
module purge; module load bluebear
module load bear-apps/2023a
module load Python/3.11.3-GCCcore-12.3.0
```

environment setup: 
```
cd sam3-background-removal
python -m venv venvs/sam3-env
source venvs/sam3-env/bin/activate
pip install -r requirements.txt
```

This is used for authentication (required for gated models): 
```
hf auth login
```
See full details here: https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication

## Model Structure
https://huggingface.co/facebook/sam3/tree/main

```
facebook/sam3/
- config.json (model architecture details)
- model.safetensors
- preprocessor_config.json
- tokenizer.json 
- tokenizer_config.json
```