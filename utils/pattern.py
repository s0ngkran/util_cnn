import torch


def generate_unique_filter_sin(random_num, img_size):
    x = torch.linspace(0, 3.14 * 2, steps=img_size).repeat(img_size, 1)
    y = torch.linspace(0, 3.14 * 2, steps=img_size).repeat(img_size, 1).T
    unique_filter = torch.sin(x) * torch.cos(y)
    return unique_filter


def generate_unique_filter_linear(i, img_size):
    pattern_list = [(0, 1), (0.5, 0.5), (1, 0)]
    scale_x, scale_y = pattern_list[i]

    x = torch.linspace(0, 1, steps=img_size).repeat(img_size, 1) * scale_x
    y = torch.linspace(0, 1, steps=img_size).repeat(img_size, 1).T * scale_y

    unique_filter = x + y

    return unique_filter
