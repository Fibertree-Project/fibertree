import sys
import random
num_nodes = int(sys.argv[1])
sparsity = int(sys.argv[2]) # (num_nodes / 100) * 3

# frontier
with open(sys.argv[3], 'w') as f:
    i = 0
    for i in range(1, int(num_nodes / sparsity)):
        num = random.randrange(1, num_nodes)
        f.write("{}\n".format(num))
