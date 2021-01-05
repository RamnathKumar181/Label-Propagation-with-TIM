from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
from utils import preprocess
import torch

def get_dataloader(args, device):
    dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}',transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()

    x = data.x


    split_idx = dataset.get_idx_split()
    preprocess_data = PygNodePropPredDataset(name=f'ogbn-{args.dataset}')[0]
    if args.dataset == 'arxiv':
        embeddings = torch.cat([preprocess(preprocess_data, 'diffusion', post_fix=args.dataset),
                                preprocess(preprocess_data, 'spectral', post_fix=args.dataset)], dim=-1)
    elif args.dataset == 'products':
        embeddings = preprocess(preprocess_data, 'spectral', post_fix=args.dataset)

    if args.use_embeddings:
        x = torch.cat([x, embeddings], dim=-1)

    if args.dataset == 'arxiv':
        x = (x-x.mean(0))/x.std(0)
    x = x.to(device)
    y_true = data.y.to(device)
    train_idx = split_idx['train'].to(device)

    return x, y_true, train_idx, dataset.num_classes, split_idx
