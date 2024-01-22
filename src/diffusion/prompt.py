import re
from typing import List

import torch

from src.utils.file_ops import get_absolute_path


def read_prompt_from_file(file_path):
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

    prompt = read_prompt_from_str(prompt)

    return prompt


def read_prompt_from_str(prompt: str) -> str:
    # remove lora description
    prompt = remove_a1111_prompt_lora_description(prompt)
    # convert a1111 format to compel
    # prompt_weight = a1111_parse_prompt_attention(prompt)
    # prompt = convert_a1111_prompt_weighting_to_compel_v2(prompt_weight, True)
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


def a1111_parse_prompt_attention(text):
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


################################################################################################################
# The following code is derived from https://github.com/damian0815/compel/issues/59#issuecomment-1696031827


def concat_tensor(t):
    t_list = torch.split(t, 1, dim=0)
    t = torch.cat(t_list, dim=1)
    return t


def merge_embeds(prompt_chunks, compel):
    num_chunks = len(prompt_chunks)
    if num_chunks != 0:
        power_prompt = 1 / (num_chunks * (num_chunks + 1) // 2)
        prompt_embs = compel(prompt_chunks)
        t_list = list(torch.split(prompt_embs, 1, dim=0))
        for i in range(num_chunks):
            t_list[-(i + 1)] = t_list[-(i + 1)] * ((i + 1) * power_prompt)
        prompt_emb = torch.stack(t_list, dim=0).sum(dim=0)
    else:
        prompt_emb = compel('')
    return prompt_emb


def detokenize(chunk, actual_prompt):
    chunk[-1] = chunk[-1].replace('</w>', '')
    chanked_prompt = ''.join(chunk).strip()
    while '</w>' in chanked_prompt:
        if actual_prompt[chanked_prompt.find('</w>')] == ' ':
            chanked_prompt = chanked_prompt.replace('</w>', ' ', 1)
        else:
            chanked_prompt = chanked_prompt.replace('</w>', '', 1)
    actual_prompt = actual_prompt.replace(chanked_prompt, '')
    return chanked_prompt.strip(), actual_prompt.strip()


def tokenize_line(line, tokenizer):  # split into chunks
    actual_prompt = line.lower().strip()
    actual_tokens = tokenizer.tokenize(actual_prompt)
    max_tokens = tokenizer.model_max_length - 2
    comma_token = tokenizer.tokenize(',')[0]

    chunks = []
    chunk = []
    for item in actual_tokens:
        chunk.append(item)
        if len(chunk) == max_tokens:
            if chunk[-1] != comma_token:
                for i in range(max_tokens - 1, -1, -1):
                    if chunk[i] == comma_token:
                        actual_chunk, actual_prompt = detokenize(chunk[:i + 1], actual_prompt)
                        chunks.append(actual_chunk)
                        chunk = chunk[i + 1:]
                        break
                else:
                    actual_chunk, actual_prompt = detokenize(chunk, actual_prompt)
                    chunks.append(actual_chunk)
                    chunk = []
            else:
                actual_chunk, actual_prompt = detokenize(chunk, actual_prompt)
                chunks.append(actual_chunk)
                chunk = []
    if chunk:
        actual_chunk, _ = detokenize(chunk, actual_prompt)
        chunks.append(actual_chunk)

    return chunks


def get_embed_new(prompt, pipeline, compel):
    attention = a1111_parse_prompt_attention(prompt)
    global_attention_chunks = []

    for att in attention:
        for chunk in att[0].split(','):
            temp_prompt_chunks = tokenize_line(chunk, pipeline.tokenizer)
            for small_chunk in temp_prompt_chunks:
                temp_dict = {
                    "weight": round(att[1], 2),
                    "length": len(pipeline.tokenizer.tokenize(f'{small_chunk},')),
                    "prompt": f'{small_chunk},'
                }
                global_attention_chunks.append(temp_dict)

    max_tokens = pipeline.tokenizer.model_max_length - 2
    global_prompt_chunks = []
    current_list = []
    current_length = 0
    for item in global_attention_chunks:
        if current_length + item['length'] > max_tokens:
            global_prompt_chunks.append(current_list)
            current_list = [[item['prompt'], item['weight']]]
            current_length = item['length']
        else:
            if not current_list:
                current_list.append([item['prompt'], item['weight']])
            else:
                if item['weight'] != current_list[-1][1]:
                    current_list.append([item['prompt'], item['weight']])
                else:
                    current_list[-1][0] += f" {item['prompt']}"
            current_length += item['length']
    if current_list:
        global_prompt_chunks.append(current_list)

    return merge_embeds(
        [convert_a1111_prompt_weighting_to_compel_v2(i, keep_float_weight=True) for i in global_prompt_chunks], compel)
