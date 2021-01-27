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
        if args.type == 'maml':
            from trainer_maml import La_MAML_Trainer
            La_MAML_Trainer(args)
        else:
            from trainer import LTrainer
            LTrainer(args)

        if args.use_correct_smooth:
            from correct_smooth import correct_smooth
            correct_smooth(args)

if __name__ == '__main__':
    main()
