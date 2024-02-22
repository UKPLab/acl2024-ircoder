from transformers import AutoTokenizer


SAVE_MODULES_MAP = {
    "bigcode/starcoderbase-1b": [
        "wte",
        "lm_head"
    ],
    "bigcode/astraios-1b-fft": [
        "wte",
        "lm_head"
    ],
    "bigcode/starcoderbase-3b": [
        "transformer.wte",
        "lm_head"
    ],
    "bigcode/starcoderbase-7b": [
        "transformer.wte",
        "lm_head"
    ],
    "Salesforce/codegen-350M-multi": [
        "wte",
        "lm_head"
    ],
    "Salesforce/codegen-2B-multi": [
        "wte",
        "lm_head"
    ],
    "deepseek-ai/deepseek-coder-1.3b-base": [
        "embed_tokens",
        "lm_head"
    ],
    "deepseek-ai/deepseek-coder-5.7bmqa-base": [
        "embed_tokens",
        "lm_head"
    ],
    "codellama/CodeLlama-7b-hf": [
        "embed_tokens",
        "lm_head"      
    ]
}


LORA_COMPONENTS_MAP = {
    "bigcode/starcoderbase-1b": [
        #"wte",
        #"lm_head",
        "c_attn",
        "c_proj",
        "q_attn",
        #"c_fc"
    ],
    "iNeil77/starcoderbase-1b-irv-400": [
        #"wte",
        #"lm_head",
        "c_attn",
        "c_proj",
        "q_attn",
        #"c_fc"
    ],
    "bigcode/astraios-1b-fft": [
        #"wte",
        #"lm_head",
        "c_attn",
        "c_proj",
        "q_attn",
        #"c_fc"
    ],
    "bigcode/starcoderbase-3b": [
        "c_attn",
        "c_proj",
        "q_attn",
        #"c_fc"
    ],
    "iNeil77/starcoderbase-3b-irv-800": [
        "c_attn",
        "c_proj",
        "q_attn",
        #"c_fc"
    ],
    "bigcode/starcoderbase-7b": [
        "c_attn",
        "c_proj",
        "q_attn",
        #"c_fc"
    ],
    "Salesforce/codegen-350M-multi": [
        "qkv_proj",
        "out_proj"
    ],
    "Salesforce/codegen-2B-multi": [
        "qkv_proj",
        "out_proj"
    ],
    "deepseek-ai/deepseek-coder-1.3b-base": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        #"up_proj",
        #"down_proj"
    ],
    "iNeil77/deepseek-coder-1.3b-base-irf": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        #"up_proj",
        #"down_proj"
    ],
    "deepseek-ai/deepseek-coder-5.7bmqa-base": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        #"up_proj",
        #"down_proj"
    ],
    "codellama/CodeLlama-7b-hf": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        #"up_proj",
        #"down_proj"      
    ],
    "iNeil77/codellama-7b-hf-irv-400": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        #"up_proj",
        #"down_proj"      
    ]
}


TOKENIZER_MAP = {
    "bigcode/starcoderbase-1b": AutoTokenizer.from_pretrained(
        "bigcode/starcoderbase-1b",
        padding_side="left", 
        truncation_side="right", 
        model_max_length=4096, 
        pad_token="<|pad|>", 
        additional_special_tokens=[
            '<source_to_llvm>', 
            '<llvm_to_source>'
        ]
    ),
    "bigcode/astraios-1b-fft": AutoTokenizer.from_pretrained(
        "bigcode/astraios-1b-fft",
        padding_side="left", 
        truncation_side="right", 
        model_max_length=4096, 
        pad_token="<|pad|>", 
        additional_special_tokens=[
            '<source_to_llvm>', 
            '<llvm_to_source>'
        ]
    ),
    "bigcode/starcoderbase-3b": AutoTokenizer.from_pretrained(
        "bigcode/starcoderbase-3b",
        padding_side="left", 
        truncation_side="right", 
        model_max_length=4096, 
        pad_token="<|pad|>", 
        additional_special_tokens=[
            '<source_to_llvm>', 
            '<llvm_to_source>'
        ]
    ),
    "bigcode/starcoderbase-7b": AutoTokenizer.from_pretrained(
        "bigcode/starcoderbase-7b",
        padding_side="left", 
        truncation_side="right", 
        model_max_length=4096, 
        pad_token="<|pad|>", 
        additional_special_tokens=[
            '<source_to_llvm>', 
            '<llvm_to_source>'
        ]
    ),
    "Salesforce/codegen-350M-multi": AutoTokenizer.from_pretrained(
        "Salesforce/Codegen-350M-multi",
        padding_side="left", 
        truncation_side="right", 
        model_max_length=2048, 
        pad_token="<|pad|>", 
        additional_special_tokens=[
            '<source_to_llvm>', 
            '<llvm_to_source>',
            '<filename>',
            '<gh_stars>',
            '<issue_start>',
            '<issue_comment>',
            '<issue_closed>',
            '<jupyter_start>',
            '<jupyter_text>',
            '<jupyter_code>',
            '<jupyter_output>',
            '<empty_output>',
            '<commit_before>',
            '<commit_msg>',
            '<commit_after>',
            '<reponame>',
        ]
    ),
    "Salesforce/codegen-2B-multi": AutoTokenizer.from_pretrained(
        "Salesforce/Codegen-2B-multi",
        padding_side="left", 
        truncation_side="right", 
        model_max_length=2048, 
        pad_token="<|pad|>", 
        additional_special_tokens=[
            '<source_to_llvm>', 
            '<llvm_to_source>',
            '<filename>',
            '<gh_stars>',
            '<issue_start>',
            '<issue_comment>',
            '<issue_closed>',
            '<jupyter_start>',
            '<jupyter_text>',
            '<jupyter_code>',
            '<jupyter_output>',
            '<empty_output>',
            '<commit_before>',
            '<commit_msg>',
            '<commit_after>',
            '<reponame>',
        ]
    ),
    "deepseek-ai/deepseek-coder-5.7bmqa-base": AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-5.7bmqa-base",
        padding_side="left", 
        truncation_side="right", 
        model_max_length=4096, 
        pad_token="<|pad|>", 
        additional_special_tokens=[
            '<source_to_llvm>', 
            '<llvm_to_source>',
            '<filename>',
            '<gh_stars>',
            '<issue_start>',
            '<issue_comment>',
            '<issue_closed>',
            '<jupyter_start>',
            '<jupyter_text>',
            '<jupyter_code>',
            '<jupyter_output>',
            '<empty_output>',
            '<commit_before>',
            '<commit_msg>',
            '<commit_after>',
            '<reponame>',
        ]
    ),
    "deepseek-ai/deepseek-coder-1.3b-base": AutoTokenizer.from_pretrained(
        "deepseek-ai/deepseek-coder-1.3b-base",
        padding_side="left", 
        truncation_side="right", 
        model_max_length=4096, 
        pad_token="<|pad|>", 
        additional_special_tokens=[
            '<source_to_llvm>', 
            '<llvm_to_source>',
            '<filename>',
            '<gh_stars>',
            '<issue_start>',
            '<issue_comment>',
            '<issue_closed>',
            '<jupyter_start>',
            '<jupyter_text>',
            '<jupyter_code>',
            '<jupyter_output>',
            '<empty_output>',
            '<commit_before>',
            '<commit_msg>',
            '<commit_after>',
            '<reponame>',
        ]
    ),
    "codellama/CodeLlama-7b-hf": AutoTokenizer.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        padding_side="left", 
        truncation_side="right", 
        model_max_length=4096, 
        pad_token="<|pad|>", 
        additional_special_tokens=[
            '<source_to_llvm>', 
            '<llvm_to_source>',
            '<filename>',
            '<gh_stars>',
            '<issue_start>',
            '<issue_comment>',
            '<issue_closed>',
            '<jupyter_start>',
            '<jupyter_text>',
            '<jupyter_code>',
            '<jupyter_output>',
            '<empty_output>',
            '<commit_before>',
            '<commit_msg>',
            '<commit_after>',
            '<reponame>',
        ]
    )
}

