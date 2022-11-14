import numpy as np
import yaml
from collections import namedtuple
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger
from time import time
from glob import glob
import os
from constants import PATH
    
def timer(func):
    def wrapper(*args,**kw):
        start = time()
        res = func(*args,**kw)
        duration = time() - start
        print(f"run {func.__name__} in {duration:.1f} seconds")
        return res
    return wrapper
    
def load_yaml_to_dict(path):
    with open(path) as f:
        x = yaml.safe_load(f)
    res = {}
    for i in x:
        res[i] = x[i]['value']
    return res
    
def load_yaml(path):
    res = load_yaml_to_dict(path)
    config = dict_to_namedtuple(res)
    print(config)
    return config

def dict_to_namedtuple(dic):
    return namedtuple('Config', dic.keys())(**dic)
    
if __name__ == '__main__':
    pass
