#!/bin/bash

# Base model path
BASE_MODEL_PATH="sparta_alignment/base_model"

# Data directory
DATA_DIR="sparta_alignment/data/culture"

# Output directory
OUTPUT_DIR="sparta_alignment/data/culture/evaluation_results"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# List of adapter models to evaluate
ADAPTER_MODELS=(
    "sparta_alignment/init_model/code_alpaca"
    "sparta_alignment/init_model/cot"
    "sparta_alignment/init_model/flan_v2"
    "sparta_alignment/init_model/gemini_alpaca"
    "sparta_alignment/init_model/lima"
    "sparta_alignment/init_model/oasst1"
    "sparta_alignment/init_model/open_orca"
    "sparta_alignment/init_model/science"
    "sparta_alignment/init_model/sharegpt"
    "sparta_alignment/init_model/wizardlm"
)

# Function to evaluate a single model
evaluate_model() {
    adapter_path=$1
    gpu_id=$2
    
    echo "====================================================="
    echo "Evaluating adapter: $adapter_path on GPU $gpu_id"
    echo "====================================================="
    
    # Extract model name from path
    model_name=$(basename "$adapter_path")
    
    # Run evaluation
    CUDA_VISIBLE_DEVICES=$gpu_id python sparta_alignment/data/culture/evaluation.py \
        --base_model_path $BASE_MODEL_PATH \
        --adapter_path $adapter_path \
        --task test \
        --data_dir $DATA_DIR \
        --output_dir "$OUTPUT_DIR/$model_name" > "$OUTPUT_DIR/${model_name}_log.txt" 2>&1
    
    echo "Evaluation complete for $model_name"
    echo "Results saved to $OUTPUT_DIR/$model_name"
    echo "====================================================="
}

# Run evaluations in parallel
for i in "${!ADAPTER_MODELS[@]}"; do
    # Determine GPU ID (assuming you have 10 GPUs numbered 0-9)
    gpu_id=$i
    
    # Run evaluation in background
    evaluate_model "${ADAPTER_MODELS[$i]}" $gpu_id &
    
    # Optional: add a small delay to prevent all processes from starting at exactly the same time
    sleep 2
done

# Wait for all background processes to finish
wait

echo "All evaluations complete!"