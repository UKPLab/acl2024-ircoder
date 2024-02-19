modelname="$1"

for lang in "cpp" "go" "python" "rust"
do
    python /vllm-code-harness/main.py \
        --model $modelname \
        --max_length_generation 1024 \
        --prompt "ircoder" \
        --tasks "humanevalfixdocs-$lang" \
        --temperature 0.2 \
        --n_samples 20 \
        --precision "fp16" \
        --allow_code_execution \
        --continuous_batching_size 32 \
        --swap_space 128 \
        --save_references_path "/Outputs/$modelname/humanevalfixtests/$lang/references.json" \
        --save_generations_path "/Outputs/$modelname/humanevalfixtests/$lang/generations.json" \
        --metric_output_path "/Outputs/$modelname/humanevalfixtests/$lang/metrics.json" 
done