"""Console script for project Livia Project."""
import os
from cli import parse_args
from utils import seed_everything
import numpy as np

def main():
    """
    Console script for project Livia Project.
    """
    args = parse_args()
    if not args.eval:
        from trainer import LTrainer
        LTrainer(args)

if __name__ == '__main__':
    main()
