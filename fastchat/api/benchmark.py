import argparse
import logging
import time

import tqdm

from fastchat.api.generate import Creator

from . import utils


def benchmark(model, data, batch_size=1, strategy="stream", max_length=512, **kwargs):
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
            num_tokens += item["num_input_tokens"]
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
    utils.describe(inputs_len, name="input")
    utils.describe(sentences_len, name="sentence")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, default="playground/data/alpaca-data-conversation.json"
    )
    parser.add_argument("--num-data", type=int, default=32)
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

    # --- stream ---
    # result = benchmark(model, data, batch_size=1, strategy="stream", max_length=512)
    # result = benchmark(model, data, batch_size=1, strategy="batch", max_length=512)

    # --- batch size ---
    result = benchmark(model, data, batch_size=4, strategy="batch", max_length=512)
    # result = benchmark(model, data, batch_size=8, strategy="batch", max_length=512)
    # result = benchmark(model, data, batch_size=16, strategy="batch", max_length=512)
    # result = benchmark(model, data, batch_size=32, strategy="batch", max_length=512)
    # !! OOM !!
    # result = benchmark(model, data, batch_size=64, strategy="batch", max_length=512)

    # --- max length ---
    # result = benchmark(model, data, batch_size=8, strategy="batch", max_length=256)
    # result = benchmark(model, data, batch_size=8, strategy="batch", max_length=128)
    # result = benchmark(model, data, batch_size=8, strategy="batch", max_length=64)

    # --- group strategy ---
    # result = benchmark(model, data, batch_size=256, strategy="group", max_length=512, mini_batch_size=16)
    # result = benchmark(model, data, batch_size=16, strategy="group", max_length=512, mini_batch_size=8)
