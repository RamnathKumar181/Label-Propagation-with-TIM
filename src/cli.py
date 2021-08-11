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
    parser.add_argument("--type",
                        required=False,
                        type=str,
                        choices=['maml', 'vanilla'],
                        default='vanilla',
                        help="Type of training to be used. Options include %(choices)")
    parser.add_argument("--dataset",
                        required=False,
                        type=str,
                        choices=['arxiv', 'products'],
                        default='arxiv',
                        help="Dataset to be used. Options include %(choices)")
    parser.add_argument("--exp_name",
                        required=False,
                        type=str,
                        default='exp',
                        help="Name of experiment (Default = 'exp')")
    parser.add_argument("--model",
                        required=False,
                        type=str,
                        choices=['mlp'],
                        default='mlp',
                        help="Model to be used. Options include %(choices)")
    parser.add_argument("--lr",
                        required=False,
                        type=float,
                        default=0.01,
                        help="Learning rate (Default = 0.01)")
    parser.add_argument("--lr_tim",
                        required=False,
                        type=float,
                        default=0.0001,
                        help="Learning rate (Default = 0.0005)")
    parser.add_argument("--lmda",
                        required=False,
                        type=float,
                        default=1,
                        help="Lambda used in tim. (Default = 1)")
    parser.add_argument("--alpha",
                        required=False,
                        type=float,
                        default=1,
                        help="Entropy importance. (Default = 1)")
    parser.add_argument("--beta",
                        required=False,
                        type=float,
                        default=1,
                        help="Entropy importance. (Default = 1)")
    parser.add_argument("--gamma",
                        required=False,
                        type=float,
                        default=1,
                        help="KL Divergence importance. (Default = 1)")
    parser.add_argument("--epochs",
                        required=False,
                        type=int,
                        default=300,
                        help="Number of epochs (Default = 300)")
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
    parser.add_argument("--use_correct_smooth",
                        required=False,
                        default=False,
                        dest="use_correct_smooth",
                        action="store_true",
                        help="Set to true if you want to use correct_smooth")
    parser.add_argument("--use_embeddings",
                        required=False,
                        default=False,
                        dest="use_embeddings",
                        action="store_true",
                        help="Set to true if you want to use embeddings")
    parser.add_argument("--criterion",
                        required=False,
                        type=str,
                        choices=['nll', 'ce'],
                        default='ce',
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
