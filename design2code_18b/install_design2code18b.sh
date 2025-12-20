#!/bin/bash
# Install dependencies for Design2Code-18B-v0

echo "Installing dependencies for Design2Code-18B-v0..."

# Core SAT dependencies
pip install SwissArmyTransformer
pip install wandb
pip install timm
pip install einops
pip install xformers
pip install deepspeed
pip install accelerate

# CogVLM dependencies
pip install spacy
pip install seaborn
pip install loguru
pip install jsonlines
pip install streamlit

# Download spacy model
python -m spacy download en_core_web_sm

# Optional but recommended
pip install bitsandbytes

echo ""
echo "Dependencies installed!"
echo ""
echo "Next steps:"
echo "1. Download the model (if not already done):"
echo "   huggingface-cli download SALT-NLP/Design2Code-18B-v0 --local-dir /root/models/design2code-18b-v0"
echo ""
echo "2. Run inference:"
echo "   cd /root/Design2code"
echo "   python run_design2code_18b.py \\"
echo "       --model-path /root/models/design2code-18b-v0 \\"
echo "       --cogvlm-path /root/Design2Code_official/CogVLM \\"
echo "       --use-dataset \\"
echo "       --samples 50 \\"
echo "       --output-dir results_Design2Code18B/predictions"
