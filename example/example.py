import os
import argparse

import sglang as sgl
from transformers import AutoTokenizer


def main(args):
    sgl.set_default_backend("vllm")
    model_path = args.model_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = sgl.Engine(
        model_path=model_path,
        tp_size=2,
        log_level="info",
        disable_overlap_schedule=True
    )

    prompts = []
    prompts_path = args.prompts_dir
    for p in os.listdir(prompts_path):
        with open(os.path.join(prompts_path, p), "r") as f:
            prompts.append(f.read())
    def construct_prompt(user_query):
        system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}\n<|im_end|>\n<|im_start|>assistant\n"
        return formatted_prompt
    prompts = [construct_prompt(p) for p in prompts]
    
    sampling_params = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 8000,
        "skip_special_tokens": False,
        "stop_token_ids": [151670, 151674]
    }
    outputs = []
    for prompt in prompts:
        outputs.append(llm.generate(prompt, sampling_params))
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
        
    for i in range(len(outputs)):
        print(f"Ratio: {len(tokenizer.encode(outputs[i]['text'])) / sampling_params['max_new_tokens']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to the Multiverse model")
    parser.add_argument("--prompts_dir", type=str, default=None, help="Path to the prompt directory")
    args = parser.parse_args()
    main(args)
