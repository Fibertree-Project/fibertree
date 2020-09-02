mkdir -p inps

# generate graph with 50 nodes with an edge at every 10
python3 gen_unif.py 50 10 inps/unif_50_10.mtx

# tile matrix into 32,32 
python3 tiling_preproc.py a inps/unif_50_10.mtx unif_50_10.yaml inps/ 32,32

# make a frontier
python3 gen_frontier.py 50 1 inps/full_50.fr

# run on nknk
python3 codec-knkn.py UU inps/full_50.fr inps/sdsd_unif_50_10.yaml > knkn_out
