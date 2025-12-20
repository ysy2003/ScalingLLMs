#!/bin/bash
# Full Design2Code-18B Pipeline
# 1. Run inference on clean images (generates HTML)
# 2. Render HTML predictions to PNG
# 3. Run robustness testing (perturbed images -> inference -> render -> evaluate)

set -e

# Configuration
PYTHON="/root/anaconda3/envs/myenv/bin/python"
SAMPLES=${1:-50}
STRENGTH=${2:-0.05}
OUTPUT_BASE="/root/Design2code/results_Design2Code18B"
ROBUSTNESS_DIR="/root/Design2code/robustness_results/design2code18b_${STRENGTH}"

echo "============================================================"
echo "DESIGN2CODE-18B FULL PIPELINE"
echo "============================================================"
echo "Samples: $SAMPLES"
echo "Perturbation strength: $STRENGTH"
echo "Output base: $OUTPUT_BASE"
echo "Robustness results: $ROBUSTNESS_DIR"
echo "============================================================"

# Step 1: Run inference on clean images (if not already done)
echo ""
echo "[STEP 1] Running inference on clean images..."
PRED_DIR="$OUTPUT_BASE/predictions"
PRED_COUNT=$(ls -1 "$PRED_DIR"/*.html 2>/dev/null | wc -l || echo 0)

if [ "$PRED_COUNT" -ge "$SAMPLES" ]; then
    echo "  Found $PRED_COUNT predictions. Skipping inference."
else
    echo "  Running Design2Code-18B inference..."
    cd /root/models--design2code-18b-v0
    $PYTHON /root/Design2code/run_design2code_18b.py \
        --model-path /root/models--design2code-18b-v0/design2code-18b-v0 \
        --cogvlm-path /root/Design2Code_official/CogVLM \
        --use-dataset \
        --samples $SAMPLES \
        --output-dir "$PRED_DIR"
fi

# Step 2: Render predictions to PNG
echo ""
echo "[STEP 2] Rendering predictions to PNG..."
cd /root/Design2code
$PYTHON render_design2code18b.py \
    --predictions-dir "$PRED_DIR" \
    --output-dir "$OUTPUT_BASE/rendered" \
    --samples $SAMPLES

# Step 3: Run robustness testing
echo ""
echo "[STEP 3] Running robustness testing..."
$PYTHON test_robustness_design2code18b.py \
    --strength $STRENGTH \
    --samples $SAMPLES \
    --clean-predictions-dir "$PRED_DIR" \
    --clean-rendered-dir "$OUTPUT_BASE/rendered" \
    --output-dir "$ROBUSTNESS_DIR"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo "Clean predictions: $PRED_DIR"
echo "Clean rendered: $OUTPUT_BASE/rendered"
echo "Robustness results: $ROBUSTNESS_DIR"
