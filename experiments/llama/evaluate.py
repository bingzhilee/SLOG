import os
import sys

import fire
import gradio as gr
import torch
import json
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from datasets import load_dataset
from datasets import load_metric
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from metrics.ExactMatchAcc import ExactMatchAcc
from utils.prompter import Prompter
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "cogs",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    datapath: str = None,
    max_new_tokens=256,
    pred_output_path: str = "",
):
    # detect if the directory of pred_output_path does not exist
    if pred_output_path != "":
        pred_output_dir = os.path.dirname(pred_output_path)
        if not os.path.exists(pred_output_dir):
            os.makedirs(pred_output_dir)

    # write somthing to pred_output_path
    # if pred_output_path != "":
    #     with open(pred_output_path, "w") as f:
    #         f.writelines("hello world")

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model, padding_side="left",)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(data_path):
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)

        # # Sort the list of data samples based on their length
        # data.sort(key=lambda x: len(x["input"]))
        #
        # # Divide the sorted list of data samples into buckets of equal size
        # batch_size = 32
        # bucket_size = batch_size * 100
        # buckets = [data[i:i + bucket_size] for i in range(0, len(data), bucket_size)]

        pred_output = []
        # Record the time cost for each step of the inference process

        # Use bucket batching to speed up the inference process


        eval_dataloader = DataLoader(data["train"], batch_size=32)
        # start_time = time.time()
        for batch in tqdm(eval_dataloader):
            # get the longest output length in the batch
            max_output_len = max([len(x) for x in batch["output"]])
            pred_output += evaluate_batch(batch)



        with open(pred_output_path, "w") as f:
            f.writelines(pred_output)

    def evaluate_batch(batch,
                       temperature=1.0,
                       top_p=0.75,
                       top_k=40,
                       num_beams=2,
                       max_new_tokens=max_new_tokens,
                       **kwargs,
    ):
        batch_size = len(batch["instruction"])
        gen_types = batch["gen_type"]
        prompts = [prompter.generate_prompt(batch["instruction"][i],
                                            batch["input"][i])
                                            for i in range(batch_size)]
        gold_outputs = batch["output"]
        encodings = tokenizer(prompts, return_tensors="pt", padding=True).to('cuda')
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_outputs = model.generate(
                **encodings,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            )
        pred = tokenizer.batch_decode(generation_outputs)

        processed_pred = prompter.get_batch_response(pred)

        acc.add_batch(pred=processed_pred, gold=gold_outputs, gen_types=gen_types)
        output = []
        for i in range(batch_size):
            input = batch["input"][i]
            pred = processed_pred[i]
            gold = gold_outputs[i]
            gen_type = batch["gen_type"][i]
            line = "\t".join([input, gold, pred, gen_type])+ "\n"
            output.append(line)

        return output

    acc = ExactMatchAcc()
    evaluate(data_path=datapath)
    print(acc.compute_metric())
    metric_path = os.path.join(os.path.dirname(pred_output_path), "metrics.json")
    with open(metric_path, "w") as f:
        json.dump(acc.compute_metric(), f)

    """
    # testing code for readme
    for instruction in [
        "Tell me about alpacas.",
        "Tell me about the president of Mexico in 2019.",
        "Tell me about the king of France in 2019.",
        "List all Canadian provinces in alphabetical order.",
        "Write a Python program that prints the first 10 Fibonacci numbers.",
        "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
        "Tell me five words that rhyme with 'shock'.",
        "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
        "Count up from 1 to 500.",
    ]:
        print("Instruction:", instruction)
        print("Response:", evaluate(instruction))
        print()
    """


if __name__ == "__main__":
    fire.Fire(main)
