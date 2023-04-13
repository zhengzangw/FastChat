import logging
import copy
import json
import time
import tqdm
import argparse

import numpy as np

from fastchat.train.train import LazySupervisedDataset, _add_speaker_and_signal
from fastchat.api.generate import Creator
from fastchat import conversation


class EvalDataset:
    def __init__(self, data_path: str):
        super().__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.list_data_dict_ori = list_data_dict
        self.list_data_dict = list_data_dict
        self.conv = conversation.conv_templates["v1"].copy()

    def sample(self, num=100, seed=1):
        np.random.seed(seed)
        self.list_data_dict = np.random.choice(self.list_data_dict_ori, num)
        logging.warning(f"Sampling {num} data points")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        data_ = copy.deepcopy(data.list_data_dict[idx])
        if isinstance(idx, int):
            data_ = [data_["conversations"]]
        else:
            data_ = [d["conversations"] for d in data_]

        inputs = []
        outputs = []
        for d in data_:
            conv = self.conv.copy()
            conv.append_message(conv.roles[0], d[0]["value"])
            conv.append_message(conv.roles[1], None)
            inputs.append(conv.get_prompt())
            outputs.append(d[1]["value"])

        if isinstance(idx, int):
            return dict(input=inputs[0], output=outputs[0])
        else:
            return dict(input=inputs, output=outputs)


def describe(input_len, name=""):
    print(f"Statistics of {name}:")
    print(f"\tMean: {np.mean(input_len):.2f}, Std: {np.std(input_len):.2f}")
    print(f"\tquartiles: {np.quantile(input_len, [0, 0.25, 0.5, 0.75, 1])}")


def benchmark(model, data, batch_size=1, strategy="stream", max_length=512):
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
            debug=True,
        )
        num_tokens += sum([o["num_input_tokens"] for o in out])
        num_finished += sum([o["num_finished"] for o in out])
        inputs_len.extend([o["num_input_tokens"] for o in out])
        outputs_len.extend([o["num_output_tokens"] for o in out])
        sentences_len.extend([o["num_total_tokens"] for o in out])

    T2 = time.time()

    interval_s = T2 - T1
    throughput_token = num_tokens / interval_s
    throughput_sample = len(data) / interval_s
    unfinish_ratio = (len(data) - num_finished) / len(data)

    print(f"Strategy: {strategy}, Batch size: {batch_size}, Max length: {max_length}")
    print(f"Total samples: {len(data)}")
    print(f"Time: {interval_s:.2f} s")
    print(
        f"Throughput: {throughput_token:.2f} tokens/s, {1/throughput_token*1000:.2f} ms/token"
    )
    print(
        f"Throughput: {throughput_sample:.2f} samples/s, {1/throughput_sample:.2f} s/sample"
    )
    print(f"Unfinished: {unfinish_ratio*100:.2f} %, {len(data) - num_finished} samples")

    describe(outputs_len, name="output")
    describe(inputs_len, name="input")
    describe(sentences_len, name="sentence")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=str, default="playground/data/alpaca-data-conversation.json"
    )
    parser.add_argument("--num-data", type=int, default=32)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--model", type=str, default="/data/scratch/vicuna-7b")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    data = EvalDataset(args.data_path)
    data.sample(args.num_data, seed=args.seed)
    model = Creator(args.model, debug=False)

    # ===
    # benchmark
    # ===

    # --- stream ---
    # result = benchmark(model, data, batch_size=1, strategy="stream", max_length=512)
    result = benchmark(model, data, batch_size=1, strategy="batch", max_length=512)

    # --- batch size ---
    # result = benchmark(model, data, batch_size=4, strategy="batch", max_length=512)
    # result = benchmark(model, data, batch_size=8, strategy="batch", max_length=512)
    # result = benchmark(model, data, batch_size=16, strategy="batch", max_length=512)
    # result = benchmark(model, data, batch_size=32, strategy="batch", max_length=512)
    # !! OOM !!
    # result = benchmark(model, data, batch_size=64, strategy="batch", max_length=512)

    # --- max length ---
    # result = benchmark(model, data, batch_size=8, strategy="batch", max_length=256)
    # result = benchmark(model, data, batch_size=8, strategy="batch", max_length=128)
    # result = benchmark(model, data, batch_size=8, strategy="batch", max_length=64)
