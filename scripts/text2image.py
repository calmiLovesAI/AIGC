from src.diffusion.stable_diffusion_v1_5 import get_stable_diffusion_v1_5_output
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


if __name__ == '__main__':
    prompt = read_prompt('experiments/prompt.txt')
    print(f"The prompt is {prompt}")
    get_stable_diffusion_v1_5_output(prompt)