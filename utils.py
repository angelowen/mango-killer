import argparse


def parse_args():
    desc = 'PyTorch example code for Kaggle competition -- Plant Seedlings Classification.\n' \
           'See https://www.kaggle.com/c/plant-seedlings-classification'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--path', help='path to dataset')
    parser.add_argument('-w', '--weight', help='path to model weights')
    parser.add_argument('-c', '--cuda_devices', type=int, help='path to model weights')
    return parser.parse_args()
