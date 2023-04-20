import copy
import io
import json
import logging
import os
import time

import numpy as np
import torch
from flexgen.compression import CompressionConfig
from flexgen.pytorch_backend import TorchDevice, fix_recursive_import

from fastchat import conversation


class EvalDataset:
    def __init__(self, data_path: str, conv_template: str = "v1"):
        super().__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.list_data_dict_ori = list_data_dict
        self.list_data_dict = list_data_dict
        self.conv = conversation.conv_templates[conv_template].copy()

    def sample(self, num=100, seed=1):
        np.random.seed(seed)
        self.list_data_dict = np.random.choice(self.list_data_dict_ori, num)
        logging.warning(f"Sampling {num} data points")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        data_json = copy.deepcopy(self.list_data_dict[idx])
        if isinstance(idx, int):
            data_ = [data_json["conversations"]]
            ids = [data_json["id"]]
        else:
            data_ = [d["conversations"] for d in data_json]
            ids = [d["id"] for d in data_json]

        inputs = []
        outputs = []
        for d in data_:
            conv = self.conv.copy()
            conv.append_message(conv.roles[0], d[0]["value"])
            conv.append_message(conv.roles[1], None)
            inputs.append(conv.get_prompt())
            outputs.append(d[1]["value"])

        if isinstance(idx, int):
            return dict(input=inputs[0], output=outputs[0], id=ids[0])
        else:
            return dict(input=inputs, output=outputs, id=ids)


def timeit(T0=None):
    torch.cuda.synchronize()
    T1 = time.time()
    if T0 is not None:
        T1 = T1 - T0
    return T1


def describe(input_len, name=""):
    print(f"Statistics of {name}:")
    print(f"\tMean: {np.mean(input_len):.2f}, Std: {np.std(input_len):.2f}")
    # print(f"\tquartiles: {np.quantile(input_len, [0, 0.25, 0.5, 0.75, 1])}")


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default, ensure_ascii=False)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


# ===
# Cache
# ===


fix_recursive_import()
dev = TorchDevice("cuda:0", 0, 0).compressed_device


def compress(x):
    config = CompressionConfig(num_bits=4, group_size=32, group_dim=0, symmetric=False)
    packed = dev.compress(x, config)
    del x
    return packed


def decompress(packed):
    x = dev.decompress(packed)
    del packed
    return x


class CACHE:
    def __init__(
        self,
        model,
        batch_size,
        length_predict_strategy="ground_truth",
        kv_cache_strategy="recomputation",
        max_new_tokens=1024,
    ) -> None:
        self.length_predict_strategy = length_predict_strategy
        self.kv_cache_strategy = kv_cache_strategy
        self.max_new_tokens = max_new_tokens

        # kv shape
        self.num_layers = model.config.num_hidden_layers
        self.num_kv = 2
        self.batch_size = batch_size
        self.num_heads = model.config.num_attention_heads
        self.dim = model.config.hidden_size

        # kv cache
        self.pre_computed = True
        self.batch_caches = []
        self.token_caches = []
        self.atten_mask_caches = []

        if length_predict_strategy == "ground_truth":
            self.pre_computed = False
            gt_json = jload(
                "playground/data/alpaca-data-conversation-10000-seed44.json"
            )
            gt_dict = {d["id"]: d["len_output_model"] for d in gt_json}
            self.gt_dict = gt_dict

    def predict_length(self, inputs, ids=None):
        if self.length_predict_strategy == "ground_truth":
            length = [self.gt_dict[id] for id in ids]
            return length
        elif self.length_predict_strategy == "max_length":
            return [self.max_new_tokens] * len(inputs)
        else:
            raise NotImplementedError


# ===
# Scheduler
# ===


def schedule(
    lengths,
    mini_batch_size=1,
    strategy="block",
):
    lengths_with_id = [(i, l) for i, l in enumerate(lengths)]
    if strategy == "block":
        sorted_lengths_with_id = sorted(
            lengths_with_id, key=lambda x: x[1], reverse=False
        )
        batches = []
        for i in range(0, len(lengths), mini_batch_size):
            batch = sorted_lengths_with_id[i : i + mini_batch_size]
            batch_ids = [x[0] for x in batch]
            max_len = batch[-1][1] + 10
            batches.append((batch_ids, max_len))
        return batches
    else:
        raise NotImplementedError
