<div align="center">

# IRCoder: Intermediate Representations Make Language Models Robust Multilingual Code Generators

[![arXiv](https://img.shields.io/badge/arXiv-2403.03894-b31b1b.svg)](https://arxiv.org/abs/2403.03894)
[![The Vault on HuggingFace datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-SLTrans-yellow?style=flat)](https://huggingface.co/datasets/UKPLab/SLTrans)

</div>

> **Abstract:**
>
> Code understanding and generation have fast become some of the most popular applications of language models (LMs). Nonetheless, research on multilingual aspects of Code-LMs (i.e., LMs for code generation) such as cross-lingual transfer between different programming languages, language-specific data augmentation and post-hoc LM adaptation, and exploitation of data sources other than source code, has been much scarcer than for their natural language counterparts. In particular, most mainstream Code-LMs have been pre-trained on source code files alone. In this work, we investigate the prospect of leveraging readily available compiler intermediate representations---shared across programming languages---to improve the multilingual capabilities of Code-LMs and facilitate cross-lingual transfer.
>
> To this end, we first compile SLTrans, a parallel dataset consisting of nearly 4M self-contained source code files coupled with respective intermediate representations. Next, starting from various base Code-LMs (ranging in size from 1.1B to 7.3B parameters), we carry out continued causal language modeling training on SLTrans, forcing the Code-LMs to (1) learn the IR language and (2) align the IR constructs with respective constructs of various programming languages. Our resulting models, dubbed IRCoder, display sizeable and consistent gains across a wide variety of code generation tasks and metrics, including prompt robustness, multilingual code completion, code understanding, and instruction following.
>
Contact person: [Indraneil Paul](mailto:indraneil.paul@tu-darmstadt.de)

[UKP Lab](https://www.ukp.tu-darmstadt.de/) | [TU Darmstadt](https://www.tu-darmstadt.de/
)

This repo contains the code accompanying ACL 24 submission IRCoder. We provide scripts to reproduce the creation of the size-optimized and performance-optimized IR from source files. We also provide the code to reproduce the evaluations we run in the paper.

## Setup and Workflow

Clone the repository and follow the workflow to reproduce the experiments. For training we provide a dockerfile to setup the training environment complete with compiled Apex optimizations and flash attention. The dockerfile is located in the `Misc` directory as `Training.Dockerfile`. Similarly, we provide a dockerfile to setup the evaluation environment. The dockerfile is located in the `Misc` directory as `Evaluation.Dockerfile`. For the stages involving training, one can setup the training environment using the following commands:
>
```bash
docker build -t ircoder-training:latest - < Misc/Training.Dockerfile
docker run -it --gpus all --name ircoder-training ircoder-training:latest
```
>
For the stages involving evaluation, one can setup the evaluation environment using the following commands:
>
```bash
docker build -t ircoder-evaluation:latest - < Misc/Evaluation.Dockerfile
docker run -it --gpus all --name ircoder-evaluation ircoder-evaluation:latest
```
>
Alternatively, one could install the dependencies in the `requirements.txt` file and get started in their CLI, though this is not recommended due to the need to execute model generated code.
>
1. Create the pair-wise dataset of source code and IR.
2. Run the continued pre-training of the base models.
3. Instruciton tuning of the models.
4. Run the zero-shot tasks evaluation.
5. Run the commit chronicle training and evaluation.
>
## Dataset Creation

We provide multiprocessing-based sample scripts to convert the source files to IR. Make sure to have the source files in a directory segregated by language. So the 1121st of the Python language should be in the `Python/Source_1121/Source.py` file. The scripts are located in the `IR_Compilation_Sample_Scripts` directory. The scripts are named `Compile_*.py` where `*` is the name of the language split. Make sure to modify the root paths in the scripts according to your setup. The scripts are designed to be run in the following way:
>
```bash
python Compile_*.py \
    --num_workers 8 \
    --subset 1400000
```
>
The `--num_workers` flag specifies the number of processes to use for the conversion. The `--subset` flag specifies the number of files to convert. Setting the `--subset` flag to `25000`, for example, will convert all the source files in the `Source_0` to `Source_24999` directory. The scripts will convert the files in the `data` directory and save the IR in the `IR` directory. Create a pairwise dataset from this and upload to HuggingFace datasets, with one field containing both the contents in the format mentioned in the paper.
>
Due to the presence of near-duplicate files in programming language source file corpora, we provide a script to remove the near-duplicates. This can be run as follows:
>
```bash
python Misc/minhash_deduplication.py --dataset "YOUR_DATASET_NAME" \
    --split "train" \
    --column "content" \
    --min-ngram-size 8 \
    --threshold 0.5 \
    --output "YOUR_OUTPUT_DIR" \
    --map-parallelism 40 \
    --filter-parallelism 40
```
>
For more agressive deduplication raise the `min-ngram-size` and lower the `threshold`. The script will remove the near-duplicates and save the dataset in the `output` directory.
>
## IR Based Continued Pre-Training

We provide the code to run the continued pre-training of the IR-based models. The script is located in the `Training_Scripts` directory. It requires a HuggingFace dataset and a HuggingFace model to run. Create a pairwise dataset of source code and IR (as shown above) and upload it to the HuggingFace datasets.
>
The script is named `continued_pretrain.py`. The script is designed to be run as follows:
>
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
>
We provide the deepspeed configuration file we used in the `Misc` directory under the name `ds_config.json`. The script will run the continued pre-training and save the model in the `output_dir` directory.
>
## Instruction Tuning

We provide the code to run the instruction tuning. The script is located in the `Training_Scripts` directory. The script is named `instruct_tune.py`. The script is designed to be run as follows:
>
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
>
## Zero-shot Tasks Evaluation

We take inspiration from the [vllm-code-harness](https://github.com/iNeil77/vllm-code-harness) library to run the zero-shot tasks evaluation. This allows us to speed up evaluations thus allowing for the extensive experiments in the paper.
>
We provide the code to run the zero-shot tasks evaluation. The script is located in the `Evaluation_Scripts` directory. The tasks include CodeXGLUE code to text, Multipl-E, HumanEvalPack-FixDocs and ReCode. We scripts are named `codexglue_code_to_text.sh`, `multipl_e.sh`, `huma_eval_pack_fixdocs.sh` and `recode.sh` respectively. The scripts already have the hyperparameters used in the paper and are designed to be run directly. For example, to run the CodeXGLUE code to text task, run the following command:
>
```bash
./Evaluation_Scripts/codexglue_code_to_text.sh
```
>
## Commit Chronicle Training and Evaluation

We provide the code to run the commit chronicle training and evaluation. It requires the runner to make the dataset available on HuggingFace datasets, split by language. The script is located in the `Training_Scripts` directory. The script is named `commitchronicle_train.py`. The script is designed to be run as follows:
>
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
>
## Experimental Disclaimer

This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

## Citation

```bib
@article{paul2024ircoder,
  title = {IRCoder: Intermediate Representations Make Language Models Robust Multilingual Code Generators},
  author = {Paul, Indraneil and Luo, Jun and Glava\v{s}, Goran and Gurevych, Iryna},
  year = 2024,
  month = mar,
  journal = {arXiv preprint},
  url = {https://arxiv.org/abs/2403.03894v1},
  eprint = {2403.03894},
  archiveprefix = {arXiv},
  primaryclass = {cs.AI},
}
```
