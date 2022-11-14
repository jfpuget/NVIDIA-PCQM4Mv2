import argparse

parser = argparse.ArgumentParser(description='Traing gnn')
parser.add_argument('--gpu','-g',dest='gpu',default=0)
parser.add_argument('--seed','-s',dest='seed',default=0)
parser.add_argument('--fold','-f',dest='fold',default=0)

args = parser.parse_args()
print(args)
