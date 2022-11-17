# NVIDIA-PCQM4Mv2
Code of the NVIDIA winning solution to the 2nd OGB-LSC at the NeurIPS 2022 challenge with dataset PCQM4Mv2

In order to reproduce our solution one has to execute code in directories. Each directory has instructions in its README.md file.

- Create training folds by running the code in the data directory.
- Train models by running code in these directories (order is not important): cnn, modelcular_transformer, pd_dgn, TransformerM. Each of these downloads a variant of the competition dataset.
- Run the ensembling notebook in ensemble directory.
