modelname="$1"

for lang in "python" "go" "ruby"
do
    python /code-harness/main.py \
        --model $modelname \
        --max_length_generation 512 \
        --tasks "codexglue_code_to_text-$lang" \
        --temperature 0.0 \
        --n_samples 1 \
        --precision "fp16" \
        --allow_code_execution \
        --continuous_batching_size 64 \
        --swap_space 128 \
        --save_references_path "/Outputs/$modelname/code-text/$lang/references.json" \
        --save_generations_path "/Outputs/$modelname/code-text/$lang/generations.json" \
        --metric_output_path "/Outputs/$modelname/code-text/$lang/metrics.json" 
done