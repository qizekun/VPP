import torch
import torch.nn as nn
import argparse
from clip import clip
from coco_queries_val import queries



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", type=str)
    args = parser.parse_args()

    clip_model = clip.load(args.clip)


if __name__ == "__main__":
    main()

