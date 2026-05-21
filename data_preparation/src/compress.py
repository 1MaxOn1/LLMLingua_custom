import argparse
import copy
import json
import os
import time

import tiktoken
from tqdm import tqdm

parser = argparse.ArgumentParser(description="compress any prompt.")

parser.add_argument("--compressor", help="compress method", default="gpt4")
parser.add_argument("--model_name", help="llm used to compress", default="gpt-4-32k")

parser.add_argument(
    "--load_origin_from", help="dataset used to compress", required=True
)
parser.add_argument(
    "--load_key", help="the key to load the text to compress", default="prompt"
)
parser.add_argument(
    "--save_key",
    help="the key to save the compressed text",
    default="compressed_prompt",
)

parser.add_argument("--save_path", help="path to save results", required=True)

parser.add_argument(
    "--load_prompt_from", help="", default="compression_instructions.json"
)
parser.add_argument("--prompt_id", type=int, default=4)
parser.add_argument("--n_max_new_token", type=int, default=4000)

parser.add_argument("--chunk_size", type=int, default=-1)

parser.add_argument(
    "--compression_rate", help="compression rate", type=float, default=0.5
)
parser.add_argument(
    "--n_target_token", help="number of target tokens", type=int, default=-1
)

args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

data = json.load(open(args.load_origin_from))
print(f"num data: {len(data)}")

if args.compressor == "gpt4":
    from GPT4_compressor import PromptCompressor

    prompts = json.load(open(args.load_prompt_from))
    system_prompt = prompts[str(args.prompt_id)]["system_prompt"]
    user_prompt = prompts[str(args.prompt_id)]["user_prompt"]
    compressor = PromptCompressor(
        model_name=args.model_name, system_prompt=system_prompt, user_prompt=user_prompt
    )
elif args.compressor == "llama3":
    from llama3_compressor import PromptCompressor

    prompts = json.load(open(args.load_prompt_from))
    system_prompt = prompts[str(args.prompt_id)]["system_prompt"]
    user_prompt = prompts[str(args.prompt_id)]["user_prompt"]
    compressor = PromptCompressor(
        model_name=args.model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        top_p=0.95,
        n_max_token=4096,
    )
elif args.compressor == "qwen":
    from qwen_compressor import PromptCompressor

    prompts = json.load(open(args.load_prompt_from))
    system_prompt = prompts[str(args.prompt_id)]["system_prompt"]
    user_prompt = prompts[str(args.prompt_id)]["user_prompt"]
    compressor = PromptCompressor(
        model_name=args.model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        top_p=0.95,
        n_max_token=4096,
    )
elif args.compressor == "llmlingua" or args.compressor == "longllmlingua":
    from llmlingua import PromptCompressor

    compressor = PromptCompressor()
elif args.compressor == "sc":
    from select_context import SelectiveContext

    compressor = SelectiveContext(model_type="NousResearch/Llama-2-7b-hf", lang="en")
else:
    raise NotImplementedError()

results = {}
results_list = []
total_time = 0

if os.path.exists(args.save_path):
    results = json.load(open(args.save_path))

tokenizer = tiktoken.encoding_for_model("gpt-4")

def chunk_origin(origin_text):
    origin_list = []
    origin_token_ids = tokenizer.encode(origin_text)
    end_token_ids = set(tokenizer.encode(".") + tokenizer.encode("\n"))
    n = len(origin_token_ids)
    st = 0
    while st < n:
        if st + args.chunk_size > n - 1:
            chunk = tokenizer.decode(origin_token_ids[st:n])
            origin_list.append(chunk)
            break
        else:
            ed = st + args.chunk_size
            for j in range(0, ed - st):
                if origin_token_ids[ed - j] in end_token_ids:
                    ed = ed - j
                    break
            chunk = tokenizer.decode(origin_token_ids[st : ed + 1])
            origin_list.append(chunk)
            st = ed + 1
    return origin_list

for sample in tqdm(data):
    idx = int(sample["idx"])
    origin = copy.deepcopy(sample[args.load_key])
    if origin is None:
        continue
    if idx in results or str(idx) in results:
        print(f"{idx}-th sample is processed")
        continue

    t = time.time()

    if args.compressor == "llmlingua" or args.compressor == "longllmlingua":
        comp_dict = compressor.compress_prompt(
            origin, ratio=args.compression_rate, target_token=args.n_target_token
        )
        comp = comp_dict["compressed_prompt"]
        comp_list = []
    elif args.compressor in ["llama3", "qwen"]:
        if args.chunk_size > 0:
            if isinstance(origin, str):
                origin_chunks = chunk_origin(origin)
            elif isinstance(origin, list):
                origin_chunks = []
                for doc in origin:
                    origin_chunks.extend(chunk_origin(doc))
            else:
                origin_chunks = [origin]
        else:
            origin_chunks = [origin] if isinstance(origin, str) else origin[:]

        comp_list = []
        for chunk in origin_chunks:
            comp_part = compressor.compress(chunk, args.n_max_new_token)
            comp_list.append(comp_part)
        comp = "".join(comp_list)
    else:
        if isinstance(origin, list):
            if args.chunk_size > 0:
                chunk_list = []
                for j, document in enumerate(origin):
                    ori_list = chunk_origin(document)
                    chunk_list.extend(ori_list)
                origin = chunk_list
        else:
            origin = [origin]
            if args.chunk_size > 0:
                origin = chunk_origin(origin[0])

        comp_list = []
        for j, chunk in enumerate(origin):
            if args.compressor == "gpt4":
                comp_part = compressor.compress(chunk, args.n_max_new_token)
            elif args.compressor == "sc":
                if args.n_target_token > 0:
                    reduce_ratio = 1 - min(
                        (args.n_target_token // len(origin))
                        / len(tokenizer.encode(chunk)),
                        1.0,
                    )
                else:
                    reduce_ratio = 1.0 - args.compression_rate
                comp_part, reduced = compressor(
                    chunk, reduce_ratio=reduce_ratio, reduce_level="token"
                )
                comp_part = comp_part.replace("<s>", "").replace("</s>", "")
            comp_list.append(comp_part)
        comp = "".join(comp_list)

    total_time += time.time() - t
    new_sample = copy.deepcopy(sample)
    new_sample[args.save_key] = comp
    if "comp_list" in locals() and len(comp_list) > 0:
        origin_to_save = origin_chunks if args.compressor in ["llama3", "qwen"] and "origin_chunks" in locals() else origin
        new_sample["prompt_list"] = origin_to_save
        new_sample["compressed_prompt_list"] = comp_list

    results[idx] = new_sample
    json.dump(
        results,
        open(args.save_path, "w", encoding="utf8"),
        indent=4,
        ensure_ascii=False,
    )

print(args.save_path, total_time)
json.dump(
    results, open(args.save_path, "w", encoding="utf8"), indent=4, ensure_ascii=False
)