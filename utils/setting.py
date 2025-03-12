import torch
from dataclasses import dataclass
from typing import List 
import ast

@dataclass
class Setting:
    name: str
    batch: int
    tr_va: str
    lr: float
    nw: int
    device: str
    model: str
    continue_training: bool
    checkpoint: str
    continue_ep: str
    log_dir: str
    min_stop: int

    def __init__(self, data: str):
        parsed_data = self.parse_data(data)
        self.name = parsed_data.get('name')
        self.batch = parsed_data.get('batch')
        self.tr_va = parsed_data.get('tr:va')
        self.lr = parsed_data.get('lr')
        self.nw = parsed_data.get('nw')
        self.device = parsed_data.get('device')
        self.model = Setting.Model(parsed_data.get('model'))
        self.continue_training = parsed_data.get('continue')
        self.checkpoint = parsed_data.get('checkpoint')
        self.continue_ep = parsed_data.get('continue_ep')
        self.log_dir = parsed_data.get('log_dir')
        self.min_stop = parsed_data.get('min_stop')
    
    @staticmethod
    def parse_data(data: str) -> dict:
        lines = data.split('\n')[1:]  # Skip the first line
        parsed_data = {}
        
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
                if value.lower() == 'none':
                    value = None
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'true':
                    value = True
                elif value.isdigit():
                    value = int(value)
                elif key == 'lr':
                    value = float(value)
                parsed_data[key] = value
        
        return parsed_data

    @dataclass
    class Model:
        model_type: str
        n_joint: int
        n_link: int
        n_stage: int
        sig_point: List[float]
        sig_link: List[float]

        def __init__(self, model_str: str):
            model_data = self.parse_model(model_str)
            self.model_type = model_data.get('model_type')
            self.n_joint = model_data.get('n_joint')
            self.n_link = model_data.get('n_link')
            self.n_stage = model_data.get('n_stage')
            self.sig_point = model_data.get('sig_point')
            self.sig_link = model_data.get('sig_link')

        @staticmethod
        def parse_model(model_str: str) -> dict:
            parts, parts2 = model_str.split('sig_point=')
            parts = parts.split()
            sp, sl = parts2.split('sig_link=')
            def to_list(s):
                s = s.replace('[', '').replace(']', '')
                out = s.split(',')
                return [(float(n) if n != 'None' else 0) for n in out]
            model_data = {
                'model_type': parts[0],
                'n_joint': int(parts[2].split('=')[1]),
                'n_link': int(parts[3].split('=')[1]),
                'n_stage': int(parts[4].split('=')[1]),
                'sig_point': to_list(sp),
                'sig_link': to_list(sl),
            }
            return model_data

def read_setting(checkpoint):
    setting = Setting(checkpoint['setting'])
    print(setting.model.sig_point)
    print(setting.model.sig_link)
    for i in setting.model.sig_link:
        print(i)

def test(path = 'save/test.best', device= 'cpu'):
    checkpoint = torch.load(path, map_location=torch.device(device))
    setting = read_setting(checkpoint)

if __name__ == '__main__':
    test()
