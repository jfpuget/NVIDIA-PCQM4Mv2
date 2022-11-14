from datasets import get_ds
from constants import PATH
from glob import glob
import sys

def eval(commit):
    f'{PATH}/rnn/{commit}/valid'

if __name__ == '__main__':
    from utils import load_yaml
    config = load_yaml('../yaml/gnn.yaml')
    print(config)
    print(type(config.use_kfold))
