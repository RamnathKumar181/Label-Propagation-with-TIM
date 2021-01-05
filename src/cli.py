"""Console script for project livia."""
import os
from argparse import ArgumentParser

def parse_args():
    """
    Parse arguments
    """
    parser = ArgumentParser(
        description = "Basic Interface for the Livia "
    )
    parser.add_argument("--dataset",
                        required=True,
                        type=str,
                        choices=['arxiv', 'products'],
                        help="Dataset to be used. Options include %(choices)")
    parser.add_argument("--exp_name",
                        required=False,
                        type=str,
                        default='exp',
                        help="Name of experiment (Default = 'exp')")
    parser.add_argument("--model",
                        required=True,
                        type=str,
                        choices=['mlp'],
                        help="Model to be used. Options include %(choices)")
    parser.add_argument("--lr",
                        required=False,
                        type=float,
                        default=0.01,
                        help="Learning rate (Default = 0.01)")
    parser.add_argument("--lmda",
                        required=False,
                        type=float,
                        default=1,
                        help="Lambda used in tim. (Default = 1)")
    parser.add_argument("--p",
                        required=False,
                        type=float,
                        default=1,
                        help="Entropy importance. (Default = 1)")
    parser.add_argument("--epochs",
                        required=False,
                        type=int,
                        default=300,
                        help="Number of epochs (Default = 300)")
    parser.add_argument("--batch_size",
                        required=False,
                        type=int,
                        default=32,
                        help="Batch size (Default = 32)")
    parser.add_argument("--runs",
                        required=False,
                        type=int,
                        default=10,
                        help="Number of runs (Default = 10)")
    parser.add_argument("--split_ratio",
                        required=False,
                        type=float,
                        default=0.2,
                        help="Train-test split ratio (Default = 0.2)")
    parser.add_argument("--eval",
                        required=False,
                        default=False,
                        dest="eval",
                        action="store_true",
                        help="Set to true if you want to evaluate your model")
    parser.add_argument("--correct_smooth",
                        required=False,
                        default=False,
                        dest="correct_smooth",
                        action="store_true",
                        help="Set to true if you want to use correct_smooth")
    parser.add_argument("--use_embeddings",
                        required=False,
                        default=False,
                        dest="use_embeddings",
                        action="store_true",
                        help="Set to true if you want to use embeddings")
    parser.add_argument("--criterion",
                        required=True,
                        type=str,
                        choices=['nll', 'ce'],
                        help="Dataset to be used. Options include %(choices)")
    parser.add_argument("--tim",
                        required=False,
                        default=False,
                        dest="tim",
                        action="store_true",
                        help="Set to true if you want to use tim learning in your model")
    parser.add_argument('--model_path',
                        type=str,
                        default='',
                        help="Pretrained model path (Default = '')")
    args = parser.parse_args()
    return args
