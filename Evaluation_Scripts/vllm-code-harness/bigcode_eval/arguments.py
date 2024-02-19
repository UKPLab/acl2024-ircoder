from dataclasses import (
    dataclass, 
    field
)
from typing import Optional
from bigcode_eval.tasks import ALL_TASKS
import fnmatch

def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    if len(task_names) > 1:
        raise ValueError(f"This repo only supports one task at a time but received {len(task_names)} tasks")
    return list(task_names)[0]


@dataclass
class ModelArguments:
    model: str = field(
        metadata={"help":"Model to evaluate, provide a repo name in Hugging Face hub or a local path"}
    )
    use_auth_token: Optional[bool] = field(
        default=True,
        metadata={"help":"Use the token generated when running `huggingface-cli login` (necessary for private model)."}
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={"help":"Use a model with custom code, this requires executing code by the author of the model."}
    )
    precision: Optional[str] = field(
        default="fp32",
        metadata={"help":"Model precision, from: fp32, fp16 or bf16"}
    )
    left_padding: Optional[bool] = field(
        default=False,
        metadata={"help":"Force left padding, needed for models like chatglm3-6b"}
    )

@dataclass
class WorkflowArguments:
    tasks: str = field(
        metadata={"help":f"Evaluation tasks from {ALL_TASKS}"},
    )
    instruction_tokens: Optional[str] = field(
        default=None,
        metadata={"help":"A series of instruction tokens used for instruction-tuning benchamrks" 
                  + "separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>"}
    )
    metric_output_path: Optional[str] = field(
        default="/tmp/evaluation_results.json",
        metadata={"help":"Path to save the results"}
    )
    save_generations: Optional[bool] = field(
        default=True,
        metadata={"help":"Whether to save code generations"}
    )
    save_generations_path: Optional[str] = field(
        default="/tmp/generations.json",
        metadata={"help":"Path for saving the code generations"}
    )
    save_references: Optional[bool] = field(
        default=True,
        metadata={"help":"Whether to save reference solutions/tests"}
    )
    save_references_path: Optional[str] = field(
        default="/tmp/references.json",
        metadata={"help":"Path for saving the references solutions/tests"}
    )
    prompt: Optional[str] = field(
        default="prompt",
        metadata={"help":"Prompt type to use for generation in HumanEvalPack tasks"}
    )
    prefix: Optional[str] = field(
        default="",
        metadata={"help":"Prefix to add to the prompt. For example InCoder needs prefix='<| file ext=.py |>\n'"}
    )
    seed: Optional[int] = field(
        default=0, 
        metadata={"help":"Random seed used for evaluation."}
    )
    limit: Optional[int] = field(
        default=None,
        metadata={"help":"Number of samples to solve and evaluate from the benchmark"}
    )
    limit_start: Optional[int] = field(
        default=0,
        metadata={"help":"Optional offset to start from when limiting the number of samples"}
    )
    postprocess: Optional[bool] = field(
        default=True,
        metadata={"help":"Postprocess model outputs before execution, always on except during generation tests"}
    )
    allow_code_execution: Optional[bool] = field(
        default=True,
        metadata={"help":"Allow code evaluation to execute external/untrusted Python code on your machine"}
    )
    generation_only: Optional[bool] = field(
        default=False,
        metadata={"help":"Do code generation but no evaluation"}
    )
    load_generations_path: Optional[str] = field(
        default=None,
        metadata={"help":"Path of file with previously generated solutions, if provided generation is"
                  + "skipped and only evaluation is done"}
    )
    load_data_path: Optional[str] = field(
        default=None,
        metadata={"help":"Path of additional data to load for the tasks"}
    )

@dataclass
class VLLMArguments:
    gpu_memory_utilization: Optional[float] = field(
        default=0.95,
        metadata={"help":"Proportion of GPU memory to reserve for vllm"}
    )
    swap_space: Optional[int] = field(
        default=64,
        metadata={"help":"RAM memory to reserve for excess GPU pages"}
    )
    continuous_batching_size: Optional[int] = field(
        default=None,
        metadata={"help":"The number of dataset samples to be sent at a time for vllm to apply continuous batching. "
            + "If None (default), all the prompts are sent to the LLM Engine together. Make sure to "
            + "modify the CPU swap_space as you modify this parameter or you may get OOM errors."}
    )

@dataclass
class GenerationArguments:
    temperature: Optional[float] = field(
        default=0.2, 
        metadata={"help":"Sampling temperature used for generation. " 
            + "Temperatures lower than 1e-5 will leads to switching to greedy mode."}
    )
    top_k: Optional[int] = field(
        default=-1, 
        metadata={"help":"Top-k parameter used for generation. Disabled (-1) by default. "
                  + "Set to an integer of at least 1 to enable."}
    )
    top_p: Optional[float] = field(
        default=0.95, 
        metadata={"help":"Top-p parameter used for nucleus sampling."}
    )
    n_samples: Optional[int] = field(
        default=1,
        metadata={"help":"Number of completions to generate for each sample."}
    )
    repetition_penalty: Optional[float] = field(
        default=1.0, 
        metadata={"help":"Float that penalizes new tokens based on whether "
            + "they appear in the prompt and the generated text so far. Values > 1 "
            + "encourage the model to use new tokens, while values < 1 encourage "
            + "the model to repeat tokens."}
    )
    frequency_penalty: Optional[float] = field(
        default=0.0, 
        metadata={"help":"Float that penalizes new tokens based on their "
            + "frequency in the generated text so far. Values > 0 encourage the "
            + "model to use new tokens, while values < 0 encourage the model to "
            + "repeat tokens."}
    )
    presence_penalty: Optional[float] = field(
        default=0.0, 
        metadata={"help":"Float that penalizes new tokens based on whether they "
            + "appear in the generated text so far. Values > 0 encourage the model "
            + "to use new tokens, while values < 0 encourage the model to repeat"
            + "tokens."}
    )
    max_length_generation: Optional[int] = field(
        default=512,
        metadata={"help":"Maximum length of generated sequence (prompt+generation)"}
    )
