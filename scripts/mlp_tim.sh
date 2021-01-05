export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5
cd ../src && python main.py --dataset arxiv --model mlp --use_embeddings --exp_name mlp_tim --criterion ce  --tim --lmda 1 --alpha 0.1 --beta 0
cd ../scripts
