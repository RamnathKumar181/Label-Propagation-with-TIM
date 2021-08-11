# Livia_TIM

This project involves the extension of TIM for graph tasks. Instead of using the standard TIM algorithm from the paper [Transductive Information Maximization For Few-Shot Learning](https://arxiv.org/abs/2008.11297) or the [TIM Implementation](https://github.com/mboudiaf/TIM), we adapt the loss to include KL-divergence instead of uniform distribution. The prior for the KL divergence is calculated using the validation dataset.
Furthemore, we also reproduce the work from [Correct & Smooth paper](https://arxiv.org/abs/2010.13993) to serve as our backbone for our experimentation.
The objective of this research to study the effect of TIM loss on few-shot graph models.

## Datasets
We run our experiments on 2 graph datasets - ogbn-products and ogbn-arxiv, both of which are public datasets.

## Results
Some of the runs from this experiment have been logged [here](https://wandb.ai/ramnath_181/TIM?workspace=user-ramnath_181). What we understood from our experiments was that, the inclusion of the TIM loss to the model actually decreases the performance of the model, rather than improving it. This is extremely counter-intuitive since the addition of external information from the validation set, does not improve the validation accuracy either. This was so, even after replacing the entropy loss with KL divergence loss, which does not assume that the prior distribution is uniform. Although we were not able to push for a paper with this project, I believe we have shown that further study of this setting is warranted, in order to explain this peculiar behavior of TIM loss in this setting.

## Acknowledgements
I would like to thank Prof. Samira Kahou for giving me the opportunity to work on this project.

# Paper Citation

If you use the TIM loss in your research and found it helpful, please cite the following paper:

```
@article{boudiaf2020transductive,
  title={Transductive information maximization for few-shot learning},
  author={Boudiaf, Malik and Masud, Ziko Imtiaz and Rony, J{\'e}r{\^o}me and Dolz, Jos{\'e} and Piantanida, Pablo and Ayed, Ismail Ben},
  journal={arXiv preprint arXiv:2008.11297},
  year={2020}
}
```

Similarly, if you use the Correct & Smooth model and found them helpful, please cite the following paper:

```
@article{huang2020combining,
  title={Combining label propagation and simple models out-performs graph neural networks},
  author={Huang, Qian and He, Horace and Singh, Abhay and Lim, Ser-Nam and Benson, Austin R},
  journal={arXiv preprint arXiv:2010.13993},
  year={2020}
}
```
