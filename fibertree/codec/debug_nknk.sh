mkdir -p inps

# generate graph with 100 nodes with just one nonzero at (50, 50)
python3 gen_unif.py 100 50 inps/unif_100_50.mtx

# tile matrix into 32,32 -> some empty tiles
python3 tiling_preproc.py a inps/unif_100_50.mtx unif_100_50.yaml inps/ 32,32

# make a frontier
python3 gen_frontier.py 100 1 inps/full_100.fr

# run on nknk
python3 codec-nknk.py UC inps/full_100.fr inps/dsds_unif_100_50.yaml > nknk_out
