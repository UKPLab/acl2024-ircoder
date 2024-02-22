modelname="$1"

for split in "func_name"
do
    python /code-harness/main.py \
        --model $modelname \
        --max_length_generation 1024 \
        --tasks "perturbed-humaneval-$split-num_seeds_5" \
        --temperature 0.0 \
        --n_samples 1 \
        --precision "fp16" \
        --allow_code_execution \
        --continuous_batching_size 64 \
        --swap_space 128 \
        --save_references_path "/Outputs/$modelname/recode/$split/references.json" \
        --save_generations_path "/Outputs/$modelname/recode/$split/generations.json" \
        --metric_output_path "/Outputs/$modelname/recode/$split/metrics.json" 
done