import argparse
import logging
import time

import torch
import tqdm

from fastchat.api.generate import Creator

from . import utils


def benchmark(model, data, batch_size=16, strategy="stream", max_length=512, **kwargs):
    logging.warning(f"Batch size: {batch_size}")
    logging.warning(f"Strategy: {strategy}")

    num_tokens = 0
    num_finished = 0
    inputs_len, outputs_len, sentences_len = [], [], []
    T1 = time.time()
    for i in tqdm.tqdm(range(0, len(data), batch_size)):
        batch = data[i : i + batch_size]
        inputs = batch["input"]
        out = model(
            inputs,
            strategy=strategy,
            temperature=0.7,
            max_length=max_length,
            **kwargs,
        )
        for item in out:
            num_tokens += item["num_output_tokens"]
            num_finished += item["is_finished"]
            inputs_len.append(item["num_input_tokens"])
            outputs_len.append(item["num_output_tokens"])
            sentences_len.append(item["num_total_tokens"])

    T2 = time.time()

    interval_s = T2 - T1
    throughput_token = num_tokens / interval_s
    throughput_sample = len(data) / interval_s
    unfinish_ratio = (len(data) - num_finished) / len(data)

    print(f"Strategy: {strategy}, Batch size: {batch_size}, Max length: {max_length}")
    if strategy == "group":
        print(f"mini batch size: {kwargs['mini_batch_size']}")
    print(f"Total samples: {len(data)}")
    print(f"Time: {interval_s:.2f} s")
    print(
        f"Throughput: {throughput_token:.2f} tokens/s, {1/throughput_token*1000:.2f} ms/token"
    )
    print(
        f"Throughput: {throughput_sample:.2f} samples/s, {1/throughput_sample:.2f} s/sample"
    )
    print(f"Unfinished: {unfinish_ratio*100:.2f} %, {len(data) - num_finished} samples")

    utils.describe(outputs_len, name="output")
    # utils.describe(inputs_len, name="input")
    # utils.describe(sentences_len, name="sentence")


@torch.inference_mode()
def benchmark_model(model, bs, max_length, num_iter=100):
    vocab_size = model.tokenizer.vocab_size
    dummy = torch.randint(0, vocab_size, (bs, max_length)).cuda()
    t = 0
    model.model(input_ids=dummy)
    for _ in range(num_iter):
        T0 = utils.timeit()
        model.model(dummy)
        t += utils.timeit(T0)
    print(f"Batch size: {bs}, Max length: {max_length}, Time: {t/num_iter:.5f} s")


@torch.inference_mode()
def generate_subset(model, data, batch_size=16):
    out_json = []
    for i in tqdm.tqdm(range(0, len(data), batch_size)):
        batch = data[i : i + batch_size]
        out = model(batch["input"], strategy="stream", temperature=0.7, max_length=1024)
        for i, item in enumerate(out):
            result = dict(
                id=batch["id"][i],
                input=batch["input"][i],
                output_model=item["output"],
                output_gt=batch["output"][i],
            )
            result["len_output_model"] = item["num_output_tokens"]
            result["len_output_gt"] = len(model.tokenizer(batch["output"][i]).input_ids)
            out_json.append(result)
    return out_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, default="playground/data/alpaca-data-conversation.json"
    )
    parser.add_argument("--num-data", type=int, default=128)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--model", type=str, default="/data/scratch/vicuna-7b")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--conv-template", type=str, default="vicuna_v1.1")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data = utils.EvalDataset(args.data_path, conv_template=args.conv_template)
    data.sample(args.num_data, seed=args.seed)
    model = Creator(args.model, debug=args.debug, conv_template=args.conv_template)

    # ===
    # benchmark
    # ===
    # --- dummy ---
    # for length in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #     benchmark_model(model, 1, length)
    # for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #     benchmark_model(model, bs, 1)
    # for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #     benchmark_model(model, bs, 64)

    # --- batch size ---
    # result = benchmark(model, data, batch_size=1)
    # result = benchmark(model, data, batch_size=2)
    # result = benchmark(model, data, batch_size=4)
    # result = benchmark(model, data, batch_size=8)
    # result = benchmark(model, data, batch_size=16)
    # result = benchmark(model, data, batch_size=32)
    # result = benchmark(model, data, batch_size=64)

    # --- max length ---
    # result = benchmark(model, data, strategy="batch", max_length=64)
    # result = benchmark(model, data, strategy="batch", max_length=128)
    # result = benchmark(model, data, strategy="batch", max_length=256)
    # result = benchmark(model, data, strategy="batch", max_length=512)

    # --- group strategy ---
    # result = benchmark(model, data, batch_size=256, strategy="group", max_length=512, mini_batch_size=16)
    # result = benchmark(model, data, batch_size=16, strategy="group", max_length=512, mini_batch_size=8)

    # ===
    # Generate a subset
    # ===
    output_path = args.data_path.replace(
        ".json", f"-{args.num_data}-seed{args.seed}.json"
    )
    out_json = generate_subset(model, data, batch_size=16)
    utils.jdump(out_json, output_path)
    logging.warning(f"Output saved to {output_path}")
