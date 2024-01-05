import re

from tools.data.file_ops import get_absolute_path


def read_prompt(file_path, neg=False):
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
    prompt = remove_a1111_prompt_lora_description(prompt)
    prompt = prompt_weighting_from_a1111_to_compel(prompt)
    prompt = remove_paired_brackets_in_string(prompt)

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


def prompt_weighting_from_a1111_to_compel(prompt: str) -> str:
    """
    Convert a1111 prompt weighting format (such as '(long hair: 1.2)')to compel format (such as 'long hair++').
    In compel format, + corresponds to the value 1.1, ++ corresponds to 1.1^2, and - corresponds to 0.9 and -- corresponds to 0.9^2.
    :param prompt: str, a1111 format
    :return: prompt weighting in compel format
    """

    def calculate_weight(match):
        prompt_text = match.group(1).strip()
        weight = float(match.group(2))

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


def convert_prompt_to_filename(prompt, length=20):
    """
    This function takes a string, converts it to an underscore-separated string,
    and truncates it to the specified length (default is 20 characters).
    """
    # Using regular expression to replace punctuation and spaces with underscores
    converted_string = re.sub(r'[\W\s]+', '_', prompt)

    # Truncate the string to the specified length
    truncated_string = converted_string[:length]

    return truncated_string
