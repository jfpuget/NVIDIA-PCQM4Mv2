Train using a image representation from molecules graphs.

Images created using rdkit.

- Input: smiles string.
- Output: 384x384 rgb image

```mol = Chem.MolFromSmiles(smiles)
Draw.MolToFile(mol, f'../images/{id}.png', size=(384, 384), kekulize=True, wedgeBonds=True)
```

To run the notebook set the specific FOLD variable in cell 3 with a value from 0 to 3 to run each fold respectively.

The training run have 4 stages. At each stage the image size, numebr of epochs and learning rate changes.
- Stage 1: LR: 4e-4, Image size: 224, epochs: 12
- Stage 2: LR: 2e-4, Image size: 224, epochs: 12
- Stage 3: LR: 5e-5, Image size: 288, epochs: 6
- Stage 4: LR: 2.5e-5, Image size: 352, epochs: 12

The average MAE is: 0.0983
