import re

from tools.data.file_ops import get_absolute_path


def read_prompt(file_path):
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

    return prompt


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
