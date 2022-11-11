Code for the Molecular Transformer.

The script was run with these arguments on a 8 GPU V100 SXM2 16GB node. The first 2 commands can be run in parallel, same for last 2.

`
python cpmp_final_script.py --fname cpmp_final --fold 0 --input_dir /raid/pcqm4mv2ring --output_dir ../checkpoints \
  --split_path ../input/new_split_dict.pt --cuda_devices 4,5,6,7
`

`
python cpmp_final_script.py --fname cpmp_final --fold 1 --input_dir /raid/pcqm4mv2ring --output_dir ../checkpoints \
  --split_path ../input/new_split_dict.pt --cuda_devices 0,1,2,3
`
  
`
python cpmp_final_script.py --fname cpmp_final --fold 2 --input_dir /raid/pcqm4mv2ring --output_dir ../checkpoints \
  --split_path ../input/new_split_dict.pt --cuda_devices 4,5,6,7
`

`
python cpmp_final_script.py --fname cpmp_final --fold 3 --input_dir /raid/pcqm4mv2ring --output_dir ../checkpoints \
  --split_path ../input/new_split_dict.pt --cuda_devices 0,1,2,3
`
  
The code was run in a `nvidia/pytorch:22.08-py3` container. We installed these packages in he container before running the code:

`
pip install rdkit

pip install ogb
`

ogb version used was 1.3.4. The code seems to run the same with the newer ogb 1.3.5.

rdkit version is 2022.09.1
