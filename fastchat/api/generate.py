from fastchat.serve.inference import load_model
from fastchat.conversation import conv_templates

import torch
import logging

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"


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
        else:
            raise NotImplementedError

        if self.debug:
            self.print_response(out)

        return out
