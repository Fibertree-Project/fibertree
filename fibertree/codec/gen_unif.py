import sys
import random
num_nodes = int(sys.argv[1])
sparsity = int(sys.argv[2]) 
nnz = num_nodes * num_nodes 
# mtx
with open(sys.argv[3], 'w') as f:
    f.write("{} {} {}\n".format(num_nodes, num_nodes, nnz))
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            num = random.random()
            # if num < er_prob:
            if i % sparsity == 0 and j % sparsity  == 0:
                f.write("{} {} {}\n".format(i, j, 1))
