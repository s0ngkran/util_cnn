import torch
def get_gpu_memory_info():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    reserved_memory = torch.cuda.memory_reserved(0)
    allocated_memory = torch.cuda.memory_allocated(0)
    free_memory = reserved_memory - allocated_memory

    total_memory_GB = total_memory / (1024 ** 3)
    reserved_memory_GB = reserved_memory / (1024 ** 3)
    allocated_memory_GB = allocated_memory / (1024 ** 3)
    free_memory_GB = free_memory / (1024 ** 3)

    # print(f'Total GPU memory: {total_memory_GB:.2f} GB')
    # print(f'Reserved GPU memory: {reserved_memory_GB:.2f} GB')
    # print(f'Allocated GPU memory: {allocated_memory_GB:.2f} GB')
    # print(f'Free GPU memory: {free_memory_GB:.2f} GB')
    return allocated_memory_GB