import torch

def get_dist(p1, p2):
    p1_tensor = torch.tensor(p1, dtype=torch.float32)
    p2_tensor = torch.tensor(p2, dtype=torch.float32)
    vector = p2_tensor - p1_tensor
    dist = torch.linalg.norm(vector)
    return dist

def test():
    p1 = (0,1)
    p2 = (0,2)
    dist = get_dist(p1, p2)
    assert dist == 1
    print('passed')

if __name__ == '__main__':
    test()