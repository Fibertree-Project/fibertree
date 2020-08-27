import sys

num_nodes = 100
nnz = num_nodes * num_nodes
# mtx
with open(sys.argv[1], 'w') as f:
    f.write("{} {} {}\n".format(num_nodes, num_nodes, nnz))
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i % 20 == 0 and j % 20  == 0:
                f.write("{} {} {}\n".format(i, j, 1))

# frontier
with open(sys.argv[2], 'w') as f:
    for i in range(1, num_nodes):
        # if i % 5 == 0:
        f.write("{}\n".format(i))

# frontier
with open(sys.argv[3], 'w') as f:
    for i in range(1, num_nodes):
        if i < num_nodes / 10 or i >= num_nodes - (num_nodes/10):
            f.write("{}\n".format(i))
