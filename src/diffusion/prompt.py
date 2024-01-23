import math
import re
from typing import List

import torch

from src.utils.file_ops import get_absolute_path


def read_prompt_from_file(file_path, lpw):
    file_path = get_absolute_path(file_path)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Removing leading and trailing empty lines and spaces
    start = 0
    end = len(lines) - 1

    while start < len(lines) and lines[start].strip() == '':
        start += 1

    while end >= 0 and lines[end].strip() == '':
        end -= 1

    prompt = ''.join(lines[start:end + 1])

    prompt = read_prompt_from_str(prompt, lpw)

    return prompt


def read_prompt_from_str(prompt: str, lpw: bool) -> str:
    # remove lora description
    prompt = remove_a1111_prompt_lora_description(prompt)
    # convert a1111 format to compel
    if not lpw:
        prompt_weight = parse_prompt_attention(prompt)
        prompt = convert_a1111_prompt_weighting_to_compel_v2(prompt_weight, True)
    return prompt


def remove_a1111_prompt_weight(prompt: str) -> str:
    """
    Remove parentheses, colons and numbers from A1111 formatted prompts.
    :param prompt: str, input prompt in A1111 format
    :return: prompt after removing numbers, brackets and colons
    """
    cleaned_prompt = re.sub(r'[\(\):\d\.]', '', prompt)
    return cleaned_prompt


def remove_paired_brackets_in_string(prompt: str) -> str:
    """
    Only handle brackets with more than two levels of nesting, retain one level of brackets,
    and add plus signs after the right bracket, the number of which is the number of nesting levels minus one.
    :param prompt:
    :return:
    """
    result = ""
    nested_count = 0
    max_nested_count = 0

    for char in prompt:
        if char == '(':
            if nested_count == 0:
                result += '('
            nested_count += 1
        elif char == ')':
            if nested_count == 1:
                result += ')'
                result += '+' * (max_nested_count - 1)
            nested_count -= 1
        else:
            result += char
        max_nested_count = max(max_nested_count, nested_count)

    return result


def remove_a1111_prompt_lora_description(prompt: str) -> str:
    """
    Remove pairs of angle brackets < > and the characters between them from a string
    :param prompt:
    :return:
    """
    result = ""
    stack = []

    for char in prompt:
        if char == '<':
            stack.append(len(result))
        elif char == '>':
            if stack:
                start = stack.pop()
                result = result[:start]
            else:
                result += char
        else:
            if not stack:
                result += char
    result = merge_whitespace(result)
    return result


def merge_whitespace(sentence: str) -> str:
    """
    Replace multiple consecutive whitespace characters with a single space using regular expression
    :param sentence:
    :return:
    """
    cleaned_sentence = re.sub(r'\s+', ' ', sentence)
    return cleaned_sentence


def convert_a1111_prompt_weighting_to_compel(prompt: str, keep_float_weight: bool = False) -> str:
    """
    Convert a1111 prompt weighting format (such as '(long hair: 1.2)')to compel format (such as 'long hair++').
    In compel format, + corresponds to the value 1.1, ++ corresponds to 1.1^2, and - corresponds to 0.9 and -- corresponds to 0.9^2.
    :param prompt: str, a1111 format
    :param keep_float_weight: bool, whether to keep the floating point number representation of weights.
    :return: prompt weighting in compel format
    """

    def calculate_weight(match):
        prompt_text = match.group(1).strip()
        weight = float(match.group(2))
        if keep_float_weight:
            return f"({prompt_text}){weight}"

        # Calculate the compel format corresponding to the weight.
        symbols = ''
        if 0.9 <= weight < 1.0:
            symbols += '-'
            return f"({prompt_text}){symbols}"
        elif 1.0 <= weight <= 1.1:
            symbols += '+'
            return f"({prompt_text}){symbols}"
        elif weight > 1.1:
            while weight >= 1.1:
                symbols += '+'
                weight /= 1.1
            return f"({prompt_text}){symbols}"
        elif weight < 0.9:
            while weight <= 0.9:
                symbols += '-'
                weight /= 0.9
            return f"({prompt_text}){symbols}"
        else:
            return f"({prompt_text})"

    pattern = r'\((.*?)\:\s*([\d\.]+)\)'
    output = re.sub(pattern, calculate_weight, prompt)
    return output


def get_filename_from_prompt(prompt, length=20):
    """
    This function takes a string, converts it to an underscore-separated string,
    and truncates it to the specified length (default is 20 characters).
    """
    # Using regular expression to replace punctuation and spaces with underscores
    converted_string = re.sub(r'[\W\s]+', '_', prompt)
    # Remove numbers
    converted_string = re.sub(r'\d+', '', converted_string)
    # Simplify multiple consecutive underscores to a single underscore
    converted_string = re.sub(r'_{2,}', '_', converted_string)

    # Truncate the string to the specified length
    truncated_string = converted_string[:length]

    return truncated_string


def parse_prompt_attention(text):
    """
    Derived from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/prompt_parser.py
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
    """

    re_attention = re.compile(r"""
    \\\(|
    \\\)|
    \\\[|
    \\]|
    \\\\|
    \\|
    \(|
    \[|
    :\s*([+-]?[.\d]+)\s*\)|
    \)|
    ]|
    [^\\()\[\]:]+|
    :
    """, re.X)

    re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def convert_a1111_prompt_weighting_to_compel_v2(prompt_weight: List[List], keep_float_weight: bool = False) -> str:
    num_parts = len(prompt_weight)
    res = ""
    for i in range(num_parts):
        prompt = prompt_weight[i][0]
        weight = round(prompt_weight[i][1], 2)
        if keep_float_weight:
            if weight == 1.0:
                res += prompt
            else:
                res += f"({prompt}){weight}"
        else:
            if weight == 1.0:
                res += prompt
            else:
                plus_and_minus = convert_float_value_to_plus_and_minus(value=weight)
                res += f"({prompt}){plus_and_minus}"
    return res


def convert_float_value_to_plus_and_minus(value: float) -> str:
    if value == 1.0:
        return ''
    elif value < 1.0:
        n = 0
        while True:
            if 0.9 ** n < value:
                return '-' * n
            n += 1
    else:
        m = 0
        while True:
            if 1.1 ** m > value:
                return '+' * m
            m += 1


###########################################################################################################################

# The following code is derived from https://gist.github.com/takuma104/43552b8ec70b63323c57dc9c6fcb9b90


class CLIPTextCustomEmbedder:
    def __init__(self,
                 tokenizer,
                 text_encoder,
                 device,
                 clip_stop_at_last_layers=1):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.token_mults = {}
        self.device = device
        self.clip_stop_at_last_layers = clip_stop_at_last_layers

    def tokenize_line(self, line):
        def get_target_prompt_token_count(token_count):
            return math.ceil(max(token_count, 1) / 75) * 75

        id_end = self.tokenizer.eos_token_id
        parsed = parse_prompt_attention(line)
        tokenized = self.tokenizer(
            [text for text, _ in parsed], truncation=False,
            add_special_tokens=False)["input_ids"]

        fixes = []
        remade_tokens = []
        multipliers = []

        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]
                remade_tokens.append(token)
                multipliers.append(weight)
                i += 1

        token_count = len(remade_tokens)
        prompt_target_length = get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens)
        remade_tokens = remade_tokens + [id_end] * tokens_to_add
        multipliers = multipliers + [1.0] * tokens_to_add
        return remade_tokens, fixes, multipliers, token_count

    def process_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        remade_batch_tokens = []
        cache = {}
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, fixes, multipliers = cache[line]
            else:
                remade_tokens, fixes, multipliers, _ = self.tokenize_line(line)
                cache[line] = (remade_tokens, fixes, multipliers)

            remade_batch_tokens.append(remade_tokens)
            batch_multipliers.append(multipliers)

        return batch_multipliers, remade_batch_tokens

    def __call__(self, text):
        batch_multipliers, remade_batch_tokens = self.process_text(text)

        z = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[75:] for x in remade_batch_tokens]
            rem_multipliers = [x[75:] for x in batch_multipliers]

            tokens = []
            multipliers = []
            for j in range(len(remade_batch_tokens)):
                if len(remade_batch_tokens[j]) > 0:
                    tokens.append(remade_batch_tokens[j][:75])
                    multipliers.append(batch_multipliers[j][:75])
                else:
                    tokens.append([self.tokenizer.eos_token_id] * 75)
                    multipliers.append([1.0] * 75)

            z1 = self.process_tokens(tokens, multipliers)
            z = z1 if z is None else torch.cat((z, z1), dim=-2)

            remade_batch_tokens = rem_tokens
            batch_multipliers = rem_multipliers
            i += 1

        return z

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        remade_batch_tokens = [[self.tokenizer.bos_token_id] + x[:75] +
                               [self.tokenizer.eos_token_id] for x in remade_batch_tokens]
        batch_multipliers = [[1.0] + x[:75] + [1.0] for x in batch_multipliers]

        tokens = torch.asarray(remade_batch_tokens).to(self.device)
        # print(tokens.shape)
        # print(tokens)
        outputs = self.text_encoder(
            input_ids=tokens, output_hidden_states=True)

        if self.clip_stop_at_last_layers > 1:
            z = self.text_encoder.text_model.final_layer_norm(
                outputs.hidden_states[-self.clip_stop_at_last_layers])
        else:
            z = outputs.last_hidden_state

        # restoring original mean is likely not correct, but it seems to work well
        # to prevent artifacts that happen otherwise
        batch_multipliers_of_same_length = [
            x + [1.0] * (75 - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(
            batch_multipliers_of_same_length).to(self.device)
        # print(batch_multipliers.shape)
        # print(batch_multipliers)

        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape +
                                       (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z

    def get_text_tokens(self, text):
        batch_multipliers, remade_batch_tokens = self.process_text(text)
        return [[self.tokenizer.bos_token_id] + remade_batch_tokens[0]], \
            [[1.0] + batch_multipliers[0]]


def text_embeddings_equal_len(text_embedder, prompt, negative_prompt):
    cond_embeddings = text_embedder(prompt)
    uncond_embeddings = text_embedder(negative_prompt)

    cond_len = cond_embeddings.shape[1]
    uncond_len = uncond_embeddings.shape[1]
    if cond_len == uncond_len:
        return cond_embeddings, uncond_embeddings
    else:
        if cond_len > uncond_len:
            n = (cond_len - uncond_len) // 77
            return cond_embeddings, torch.cat([uncond_embeddings] + [text_embedder("")] * n, dim=1)
        else:
            n = (uncond_len - cond_len) // 77
            return torch.cat([cond_embeddings] + [text_embedder("")] * n, dim=1), uncond_embeddings


def text_embeddings(pipe, prompt, negative_prompt, clip_stop_at_last_layers=1):
    text_embedder = CLIPTextCustomEmbedder(tokenizer=pipe.tokenizer,
                                           text_encoder=pipe.text_encoder,
                                           device=pipe.text_encoder.device,
                                           clip_stop_at_last_layers=clip_stop_at_last_layers)
    cond_embeddings, uncond_embeddings = text_embeddings_equal_len(text_embedder, prompt, negative_prompt)
    return cond_embeddings, uncond_embeddings
