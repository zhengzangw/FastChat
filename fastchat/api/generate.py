import logging
import tqdm

import torch

from fastchat.serve.inference import load_model
from fastchat.conversation import conv_templates
from . import utils


class Creator:
    def __init__(
        self,
        model_name,
        conv_template="vicuna_v1.1",
        device="cuda",
        num_gpus=1,
        load_8bit=False,
        debug=False,
    ):
        self.model, self.tokenizer = load_model(
            model_name, device, num_gpus, load_8bit, debug
        )
        self.conv = conv_templates[conv_template].copy()
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"
        self.debug = debug
        self.device = device

    def print_response(self, out):
        for line in out:
            print(line["sentence"])
            print()

    def sample(self, last_token_logits, temperature):
        if temperature < 1e-4:
            token = torch.argmax(last_token_logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
        return token

    def generate(self, prompt, past_info=None, stream=True, **kwargs):
        # default values
        temperature = kwargs.get("temperature", 1.0)
        max_new_tokens = kwargs.get("max_length", 256)
        tokenizer, model, device = self.tokenizer, self.model, self.device

        # preparation
        if past_info is None:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(device)
            l_input_ids = len(input_ids[0])
            output_ids = input_ids
            attention_mask = inputs.attention_mask.to(device)
        else:
            breakpoint()
        ending = [-1] * len(prompt)

        for i in range(max_new_tokens):
            if self.debug:
                T0 = utils.timeit()
            # generation
            if i == 0 and past_info is None:
                out = model(input_ids, use_cache=True, attention_mask=attention_mask)
            else:
                out = model(
                    input_ids=token,
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )
            if self.debug:
                print("Time: ", utils.timeit(T0))

            # sample
            last_token_logits = out.logits[:, -1]
            token = self.sample(last_token_logits, temperature)
            output_ids = torch.cat((output_ids, token), dim=1)

            # update attn & kv cache
            past_key_values = out.past_key_values
            attn_dtype = attention_mask.dtype
            extend_mask = torch.ones(len(token), 1, dtype=attn_dtype).to(device)
            attention_mask = torch.cat((attention_mask, extend_mask), dim=1)

            # ending detection
            num_ended = 0
            for j in range(len(prompt)):
                if ending[j] == -1 and token[j] == tokenizer.eos_token_id:
                    ending[j] = i
                if ending[j] != -1:
                    num_ended += 1
            if num_ended == len(prompt) and stream:
                break

        # collect results
        results = []
        for i in range(len(output_ids)):
            if ending[i] != -1:
                output_ = output_ids[i][:l_input_ids + ending[i]]
                is_finished = True
            else:
                output_ = output_ids[i]
                is_finished = False
            sentence = self.tokenizer.decode(output_, skip_special_tokens=True)
            output = sentence[len(prompt[i]):]
            
            num_input_tokens = len(input_ids[i])
            num_output_tokens = len(tokenizer(output).input_ids)
            num_total_tokens = num_input_tokens + num_output_tokens

            # return
            result = dict(
                input=prompt[i],
                output=output,
                sentence=sentence,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                num_total_tokens=num_total_tokens,
                is_finished=is_finished,
            )
            results.append(result)
        return results

    def batch_ending_detect(
        self, outputs, prompt, input_ids, tokenizer, stop_str="###"
    ):
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
        for i in tqdm.tqdm(range(len(prompt) // mini_batch_size)):
            past_key_values, token, attention_mask = cache.get(i)
            print(f"kv cache get: {T2 - T1:.3f} s")
            output_ids = input_ids_list[i]
            for j in range(max_new_tokens):
                extend_mask = torch.ones(len(token), 1, dtype=attention_mask.dtype).to(
                    device
                )
                attention_mask = torch.cat((attention_mask, extend_mask), dim=1)
                out = model(
                    input_ids=token,
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                )

                logits = out.logits
                past_key_values = out.past_key_values
                last_token_logits = logits[:, -1]
                token = self.sample(last_token_logits, temperature)
                output_ids = torch.cat((output_ids, token), dim=1)
            print(f"output gen: {T3 - T2:.3f} s")
            output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            outputs.extend(output)

        input_ids_list_flat = [item for sublist in input_ids_list for item in sublist]
        results = self.batch_ending_detect(
            outputs, prompt, input_ids_list_flat, tokenizer, stop_str
        )
        return results

    @torch.inference_mode()
    def __call__(self, prompt, strategy="stream", **kwargs):
        # ===
        # batch size = 1, ending detection
        # ===
        if strategy == "stream":
            out = self.generate(prompt, **kwargs)
        # ===
        # batch size = B
        # ===
        elif strategy == "batch":
            out = self.generate(prompt, stream=False, **kwargs)
        elif strategy == "group":
            out = self.generate_group(prompt, **kwargs)
        else:
            raise NotImplementedError

        if self.debug:
            self.print_response(out)

        return out
