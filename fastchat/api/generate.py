import logging
import tqdm
import time

import torch
from flexgen.pytorch_backend import TorchDevice, fix_recursive_import
from flexgen.compression import CompressionConfig

from fastchat.serve.inference import load_model
from fastchat.conversation import conv_templates

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
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
    def __init__(self, model, batch_size) -> None:
        self.num_layers = model.config.num_hidden_layers
        self.num_kv = 2
        self.batch_size = batch_size
        self.num_heads = model.config.num_attention_heads
        self.dim = model.config.hidden_size

        self.batch_caches = []
        self.token_caches = []
        self.atten_mask_caches = []

    def append(self, kv_caches, token=None, attention_mask=None):
        kv_cache_processed = []
        for c in kv_caches:
            k = compress(c[0])
            v = compress(c[1])
            kv_cache_processed.append((k, v))
        self.batch_caches.append(kv_cache_processed)
        self.token_caches.append(token)
        self.atten_mask_caches.append(attention_mask)

    def get(self, i):
        kv_cache_processed = self.batch_caches[i]
        kv_cache = []
        for j, (k, v) in enumerate(kv_cache_processed):
            kv_cache_processed[j] = None
            k = decompress(k)
            v = decompress(v)
            kv_cache.append((k, v))
        return kv_cache, self.token_caches[i], self.atten_mask_caches[i]


class Creator:
    def __init__(
        self,
        model_name,
        conv_template="v1",
        device="cuda",
        num_gpus=1,
        load_8bit=False,
        debug=False,
    ):
        self.model, self.tokenizer = load_model(
            model_name, device, num_gpus, load_8bit, debug
        )
        self.conv = conv_templates[conv_template].copy()
        self.tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
        self.debug = debug
        self.device = device

    def print_response(self, out):
        for line in out:
            print(line["sentence"])
            print()

    def generate_stream(self, prompt, **kwargs):
        assert len(prompt) == 1, "Batch size must be 1"

        # default values
        temperature = kwargs.get("temperature", 1.0)
        max_new_tokens = kwargs.get("max_length", 256)
        stop_str = "###"
        tokenizer, model, device = self.tokenizer, self.model, self.device

        # preparation
        input_ids = tokenizer(prompt).input_ids
        num_finished = 0
        output_ids = list(input_ids[0])
        l_prompt = len(prompt[0])

        # stream generation
        for i in range(max_new_tokens):
            torch.cuda.synchronize()
            T0 = time.time()
            if i == 0:
                out = model(torch.as_tensor(input_ids, device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device=device
                )
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            torch.cuda.synchronize()
            T1 = time.time()
            print(f"Time {i}: {T1 - T0:.3f} s")

            last_token_logits = logits[0][-1]

            # greedy or sampling strategy
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            # decode output
            output_ids.append(token)
            output = tokenizer.decode(output_ids, skip_special_tokens=True)

            # ending detection
            pos = output.rfind(stop_str, l_prompt)
            if token == tokenizer.eos_token_id or pos != -1:
                if pos != -1:
                    output = output[:pos]
                num_finished += 1
                break

        del past_key_values

        # tokens
        response = output[l_prompt:]
        num_input_tokens = len(input_ids[0])
        num_total_tokens = len(output_ids)
        num_output_tokens = num_total_tokens - num_input_tokens

        return [
            dict(
                input=prompt[0],
                output=response,
                sentence=output,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                num_total_tokens=num_total_tokens,
                num_finished=num_finished,
            )
        ]

    def generate_batch(self, prompt, **kwargs):
        # default values
        temperature = kwargs.get("temperature", 1.0)
        max_new_tokens = kwargs.get("max_length", 256)
        stop_str = "###"
        tokenizer, model, device = self.tokenizer, self.model, self.device

        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        generation_output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            return_dict_in_generate=True,
        )

        output_ids = generation_output.sequences
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        results = []
        for i in range(len(outputs)):
            # ending detection
            num_finished = 0
            l_prompt = len(prompt[i])
            pos = outputs[i].find(stop_str, l_prompt)
            pos_eos = -1
            for j in range(len(output_ids[i])):
                if output_ids[i][j] == tokenizer.eos_token_id:
                    pos_eos = j
                    break
            if pos_eos != -1:
                outputs[i] = outputs[i][:pos_eos]
                num_finished += 1
            elif pos != -1:
                outputs[i] = outputs[i][:pos]
                num_finished += 1

            # tokens
            output = outputs[i][len(prompt[i]) :]
            num_input_tokens = len(input_ids[i])
            num_output_tokens = len(tokenizer(output).input_ids)
            num_total_tokens = num_input_tokens + num_output_tokens

            # return
            result = dict(
                input=prompt[i],
                output=output,
                sentence=outputs[i],
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                num_total_tokens=num_total_tokens,
                num_finished=num_finished,
            )
            results.append(result)
        return results

    def batch_ending_detect(self, outputs, prompt, input_ids, tokenizer, stop_str="###"):
        results = []
        for i in range(len(outputs)):
            # ending detection
            num_finished = 0
            l_prompt = len(prompt[i])
            pos = outputs[i].find(stop_str, l_prompt)
            if pos != -1:
                outputs[i] = outputs[i][:pos]
                num_finished += 1

            # tokens
            output = outputs[i][len(prompt[i]) :]
            num_input_tokens = len(input_ids[i])
            num_output_tokens = len(tokenizer(output).input_ids)
            num_total_tokens = num_input_tokens + num_output_tokens

            # return
            result = dict(
                input=prompt[i],
                output=output,
                sentence=outputs[i],
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                num_total_tokens=num_total_tokens,
                num_finished=num_finished,
            )
            results.append(result)
        return results

    def generate_group(self, prompt, **kwargs):
        # default values
        temperature = kwargs.get("temperature", 1.0)
        max_new_tokens = kwargs.get("max_length", 256)
        mini_batch_size = kwargs.get("mini_batch_size", 32)
        stop_str = "###"
        tokenizer, model, device = self.tokenizer, self.model, self.device

        # kv cache computation & length prediction
        cache = CACHE(model, mini_batch_size)
        input_ids_list = []
        for i in tqdm.tqdm(range(0, len(prompt), mini_batch_size)):
            # preparation
            inputs = tokenizer(
                prompt[i : i + mini_batch_size], return_tensors="pt", padding=True
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)

            # kv cache computation
            # torch.cuda.synchronize()
            T0 = time.time()
            out = model(
                input_ids=input_ids, use_cache=True, attention_mask=attention_mask
            )
            # torch.cuda.synchronize()
            T1 = time.time()
            print(f"kv cache compute: {T1 - T0:.3f} s")

            # offload kv cache
            kv_cache_gpu = out.past_key_values
            # > kv cache: (#layers, 2, #batch, #heads, #tokens, #dim)
            # > 7B: (32, 2, bs, 32, length, 128)
            token = self.sample(out["logits"][:, -1], temperature)
            cache.append(kv_cache_gpu, token, attention_mask)

            # torch.cuda.synchronize()
            T2 = time.time()
            print(f"kv cache store: {T2 - T1:.3f} s")
            del out

            input_ids_list.append(input_ids)

        # rescheduling
        pass

        # batch generation
        outputs = []
        for i in tqdm.tqdm(range(len(prompt)//mini_batch_size)):
            T1 = time.time()
            past_key_values, token, attention_mask = cache.get(i)
            T2 = time.time()
            print(f"kv cache get: {T2 - T1:.3f} s")
            output_ids = input_ids_list[i]
            for j in range(max_new_tokens):
                extend_mask = torch.ones(len(token), 1, dtype=attention_mask.dtype).to(device)
                attention_mask = torch.cat((attention_mask, extend_mask), dim=1)
                out = model(
                    input_ids=token,
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )

                logits = out.logits
                past_key_values = out.past_key_values
                last_token_logits = logits[:,-1]
                token = self.sample(last_token_logits, temperature)
                output_ids = torch.cat((output_ids, token), dim=1)
            T3 = time.time()
            print(f"output gen: {T3 - T2:.3f} s")
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs.extend(output)

        input_ids_list_flat = [item for sublist in input_ids_list for item in sublist]
        results = self.batch_ending_detect(outputs, prompt, input_ids_list_flat, tokenizer, stop_str)
        return results

    def sample(self, last_token_logits, temperature):
        if temperature < 1e-4:
            token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
        return token

    @torch.inference_mode()
    def __call__(self, prompt, strategy="stream", **kwargs):
        # ===
        # batch size = 1, ending detection
        # ===
        if strategy == "stream":
            out = self.generate_stream(prompt, **kwargs)
        # ===
        # batch size = B
        # ===
        elif strategy == "batch":
            out = self.generate_batch(prompt, **kwargs)
        elif strategy == "group":
            out = self.generate_group(prompt, **kwargs)
        else:
            raise NotImplementedError

        if self.debug:
            self.print_response(out)

        return out
