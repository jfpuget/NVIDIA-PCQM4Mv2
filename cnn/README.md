Train using a image representation from molecules graphs.

Images are created using rdkit. Run `create-images.ipynb`

- Input: smiles string. Put competition data in `../dataset` directory.
- Output: 384x384 rgb image. Images in `../images` directory.

```mol = Chem.MolFromSmiles(smiles)
Draw.MolToFile(mol, f'../images/{id}.png', size=(384, 384), kekulize=True, wedgeBonds=True)
```

To run the `train-resnet.ipynb` notebook set the specific FOLD variable in cell 3 with a value from 0 to 3 to run each fold respectively.

The training run has 4 stages. At each stage the image size, number of epochs and learning rate change.
- Stage 1: LR: 4e-4, Image size: 224, epochs: 12
- Stage 2: LR: 2e-4, Image size: 224, epochs: 12
- Stage 3: LR: 5e-5, Image size: 288, epochs: 6
- Stage 4: LR: 2.5e-5, Image size: 352, epochs: 12

The average MAE is: 0.0983




- Enviroment Requirements:
All the code was run in a nvidia/pytorch:22.04-py3 container. We installed these packages in the container before running the code:
```
torch==1.12.1
torch-scatter==2.0.9
torch-sparse==0.6.15
torch-geometric==2.1.0
ogb==1.3.5
rdkit==2022.09.1
opencv-python==4.5.5.64
numpy==1.22.4
pandas==1.4.3
tqdm==4.64.0
matplotlib==3.5.2
```
