modelname="$1"

for lang in "d" "cpp" "go" "py" "rb" "rs" "swift"
do
    for temperature in 0.2 0.8
    do
        python /vllm-code-harness/main.py \
            --model $modelname \
            --max_length_generation 1024 \
            --tasks "multiple-$lang" \
            --temperature $temperature \
            --n_samples 50 \
            --precision "fp16" \
            --allow_code_execution \
            --continuous_batching_size 32 \
            --swap_space 128 \
            --save_references_path "/Outputs/$modelname/multipl-e/$lang/$temperature/references.json" \
            --save_generations_path "/Outputs/$modelname/multipl-e/$lang/$temperature/generations.json" \
            --metric_output_path "/Outputs/$modelname/multipl-e/$lang/$temperature/metrics.json" 
    done
done