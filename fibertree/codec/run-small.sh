python3 gen-small.py small.mtx small_dense.fr small_sparse.fr
python3 tiling_preproc.py small small.mtx small.yaml 32,32
python3 codec-knkn.py UU small.fr sdsd_small.yaml > small.out
