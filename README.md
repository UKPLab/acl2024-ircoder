# IRCoder

The code accompanying ACL 24 submission IRCoder.

We provide scripts to reproduce the creation of the size-optimized and performance-optimized IR from source files. We also provide the code to reproduce the evaluations we run in the paper.

## Dataset Creation

We provide multiprocessing-based sample scripts to convert the source files to IR. The scripts are located in the `IR_Compilation_Sample_Scripts` directory. The scripts are named `Compile_*.py` where `*` is the name of the language split. The scripts are designed to be run in the following way:

```bash
python Compile_*.py \
    --num_workers 8 \
    --subset 1400000
```

The `--num_workers` flag specifies the number of processes to use for the conversion. The `--subset` flag specifies the number of files to convert. The scripts will convert the files in the `data` directory and save the IR in the `IR` directory.

## IR Based Continued Pre-Training

We provide the code to run the continued pre-training of the IR-based models. The script is located in the `Training_Scripts` directory. It requires a HuggingFace dataset and a HuggingFace model to run. Create a pairwise dataset of source code and IR and upload it to the HuggingFace datasets.

The script is named `continued_pretrain.py`. The script is designed to be run as follows:

```bash
accelerate launch --num_processes=4 --main_process_port=29699 Training_Scripts/continued_pretrain.py \
    --dataset_name "YOUR_DATASET_NAME" \
    --token "YOUR_HF_TOKEN" \
    --wandb_token "YOUR_WANDB_TOKEN" \
    --project_name "YOUR_PROJECT_NAME" \
    --run_name "YOUR_RUN_NAME" \
    --do_eval True \
    --do_train True \
    --trust_remote_code True \
    --low_cpu_mem_usage True \
    --gradient_accumulation_step 2 \
    --optim "adamw_apex_fused" \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --tf32 True \
    --logging_steps 100 \
    --logging_strategy "steps" \
    --eval_steps 1000 \
    --evaluation_strategy "steps" \
    --lr_scheduler_type "cosine" \
    --max_train_samples 256000 \
    --max_eval_samples 8192 \
    --model_name_or_path "bigcode/starcoderbase-1b" \
    --num_train_epochs 1.0 \
    --output_dir "YOUR_OUTPUT_DIR" \
    --overwrite_output_dir True \
    --per_device_eval_batch_size 8 \
    --per_device_train_batch_size 8 \
    --preprocessing_num_workers 12 \
    --report_to "wandb" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --seed 484 \
    --validation_split_percentage 10 \
    --warmup_ratio 0.05 \
    --dataloader_drop_last True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --ddp_find_unused_parameters False \
    --llm_int8_threshold 6.0 \
    --lora_alpha 128 \
    --lora_r 256 \
    --lora_dropout 0.05 \
    --deepspeed "YOUR_DEEPSPEED_CONFIG"
```

We provide the deepspeed configuration file we used in the `Misc` directory under the name `ds_config.json`. The script will run the continued pre-training and save the model in the `output_dir` directory.

## Zero-shot Tasks Evaluation

We provide the code to run the zero-shot tasks evaluation. The script is located in the `Evaluation_Scripts` directory. The tasks include CodeXGLUE code to text, Multipl-E, HumanEvalPack-FixDocs and ReCode.

## Instruction Tuning

We provide the code to run the instruction tuning. The script is located in the `Training_Scripts` directory. The script is named `instruct_tune.py`. The script is designed to be run as follows:

```bash
python Training_Scripts/instruct_tune.py \
    --model_name_or_path "deepseek-ai/deepseek-coder-5.7bmqa-base" \
    --token "YOUR_HF_TOKEN" \
    --wandb_token "YOUR_WANDB_TOKEN" \
    --hf_data_path "YOUR_DATASET_NAME" \
    --project_name "YOUR_PROJECT_NAME" \
    --run_name "YOUR_RUN_NAME" \
    --output_dir "YOUR_OUTPUT_DIR" \
    --do_train True \
    --trust_remote_code True \
    --low_cpu_mem_usage True \
    --num_train_epochs 4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --optim "adamw_apex_fused" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --dataloader_drop_last True \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --ddp_find_unused_parameters False \
    --llm_int8_threshold 6.0 \
    --lora_alpha 16 \
    --lora_r 32 \
    --lora_dropout 0.1 \
    --tf32 True
```

## Commit Chronicle Training and Evaluation

We provide the code to run the commit chronicle training and evaluation. It requires the runner to make the dataset available on HuggingFace datasets, split by language. The script is located in the `Training_Scripts` directory. The script is named `commitchronicle_train.py`. The script is designed to be run as follows:

```bash
for language in "Ruby" "Objective-C" "Swift" "Rust" "Go" "C" "C++" "Python"
do
    python /Training_Scripts/commitchronicle_train.py \
        --model_name_or_path "iNeil77/codellama-7b-hf-irv-400" \
        --token "YOUR_HF_TOKEN" \
        --wandb_token "TOUR_WANDB_TOKEN" \
        --hf_data_path "YOUR_DATASET_PATH" \
        --language $language \
        --project_name "LLVM_Align" \
        --run_name "YOUR_RUN_NAME_$language" \
        --output_dir "YOUR_OUTPUT_DIR/$language" \
        --do_train True \
        --do_predict True \
        --trust_remote_code True \
        --low_cpu_mem_usage True \
        --num_train_epochs 2 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "epoch" \
        --save_strategy "epoch" \
        --optim "adamw_apex_fused" \
        --learning_rate 3e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 50 \
        --dataloader_drop_last True \
        --dataloader_num_workers 4 \
        --dataloader_pin_memory True \
        --dataloader_persistent_workers True \
        --ddp_find_unused_parameters False \
        --llm_int8_threshold 6.0 \
        --lora_alpha 32 \
        --lora_r 64 \
        --lora_dropout 0.1 \
        --tf32 True \
        --model_max_length 768 \
        --max_train_samples 30720 \
        --max_eval_samples 2560 \
        --max_predict_samples 2560
done
```
