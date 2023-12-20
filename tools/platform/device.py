import torch


def get_device(type="gpu"):
    """
    Specify the device on which the current code is running.
    :param type: str, "cpu" or "gpu"
    :return:
    """
    match type:
        case "cpu":
            return torch.device("cpu")
        case "gpu":
            return get_max_memory_gpu()
        case _:
            return get_max_memory_gpu()


def get_max_memory_gpu():
    """
    Get the GPU with the largest memory capacity.
    :return: torch.device
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No cuda available.")
    else:
        num_gpus = torch.cuda.device_count()
        if num_gpus == 1:
            return torch.device("cuda:0")
        gpu_memory = [torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)]
        max_memory_index = gpu_memory.index(max(gpu_memory))
        return torch.device(f"cuda:{max_memory_index}")


def set_seed_based_on_device(device_type):
    if device_type == "gpu":
        # If the device type is 'gpu', create a random number generator based on the GPU device and set the seed
        generator = torch.Generator(device='cuda')
        generator.manual_seed(
            torch.cuda.current_device() or 0)  # Using the ID of the current GPU device or default to 0
    elif device_type == "cpu":
        # If the device type is 'cpu', create a random number generator based on the CPU and set the seed
        generator = torch.Generator(device=device_type)
        generator.manual_seed(0)  # Set the seed to 0
    else:
        raise ValueError("Unsupported device type. Please use 'gpu' or 'cpu'.")

    return generator
