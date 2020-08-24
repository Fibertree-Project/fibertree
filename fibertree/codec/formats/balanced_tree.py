from .compression_format import CompressionFormat
from .redBlack import *

class RBTree(CompressionFormat):
    def __init__(self):
        CompressionFormat.__init__(self)
    
    @staticmethod 
    def getHeight(root):
        if root == None or root == NIL: return 0
        left_height = RBTree.getHeight(root.left)
        right_height = RBTree.getHeight(root.right)
        if left_height > right_height:
            return left_height + 1
        else: return right_height + 1

    # TODO: full binary tree serialization
    # preorder serializiation
    @staticmethod
    def serializeTree(root, output, depth, ind, empty, height):
        if depth == height:
        # if root == None: 
            return
        # otherwise, depth < height
        if root == NIL:
            output.append(empty)
            RBTree.serializeTree(NIL, output, depth + 1, ind, empty, height)
            RBTree.serializeTree(NIL, output, depth + 1, ind, empty, height)
            return
        # return
        # write data at node into a string
        
        strout = ''
        if isinstance(root.data, int):
            # strout = str(root.data)
            output.append(root.data)
        else:
            # strout = str(root.data[ind])
            # strout = ','.join(str(v) for v in root.data)
            if isinstance(root.data[ind], tuple):
                strout = "({})".format(','.join(str(v) for v in root.data[ind]))
                output.append(strout)
            else:
                output.append(root.data[ind])
        
        RBTree.serializeTree(root.left, output, depth + 1, ind, empty, height)
        RBTree.serializeTree(root.right, output, depth + 1, ind, empty, height)


    # TODO: merge with getPayloads?
    # preorder serializiation
    @staticmethod
    def treeToString(root, depth, height, strout):
        # otherwise, depth < height
        if root == NIL:
            return
        # return
        # write data at node into a string
        
        RBTree.treeToString(root.left, depth + 1, height, strout)
        strout.append(root.data)
        RBTree.treeToString(root.right, depth + 1, height, strout)
        return

    # populate output with payloads
    def getPayloadsHelper(self, node, depth, height, output):
        if node == NIL:
            return
        self.getPayloadsHelper(node.left, depth +1, height, output)
        output.append(node.data[1])
        self.getPayloadsHelper(node.right, depth + 1, height, output)
        return

    def getPayloads(self):
        output = list()

        height = RBTree.getHeight(self.tree.root)
        self.getPayloadsHelper(self.tree.root, 0, height, output)
        return output

    # encode fiber into T format
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output, output_tensor):
        # print("encode fiber for T")
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth)

        self.tree = RedBlackTree()

        # init vars
        fiber_occupancy = 0

        cumulative_occupancy = codec.get_start_occ(depth)
        occ_list = list()
        occ_list.append(cumulative_occupancy)
        prev_nz = 0
        
        # for each nonzero in fiber
        for ind, (val) in a:
            # internal levels encode explicit coords and corresponding offset / fiber ptr
            if depth < len(ranks) - 1:
                # keep track of actual occupancy (nnz in this fiber)
                fiber, child_occupancy = codec.encode(depth + 1, val, ranks, output, output_tensor)
                
                if isinstance(cumulative_occupancy, int):
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                else:
                    cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
                # encode (coord, payload)
                if codec.fmts[depth + 1].encodeUpperPayload():
                    if codec.cumulative_payloads[depth]:
                        self.tree.add((ind, cumulative_occupancy, ind)) # fiber))
                    else:
                        self.tree.add((ind, child_occupancy, ind)) # fiber))
                else:
                    self.tree.add(ind, ind) # fiber)
                            # if a leaf, encode (coord, value)
            else:
                self.tree.add((ind, val.value))
            
            # print("searching in tree for verification for coord {}, found {}".format(ind, self.tree.contains(ind).data))
            assert ind == self.tree.contains(ind).data[0]

            fiber_occupancy = fiber_occupancy + 1
            
        # serialize tree
        # null value for empty nodes
        empty = -1

        tree = self.tree
        result = list()
        height = RBTree.getHeight(tree.root)
        size_of_tree = 2**height - 1
        
        # serialize only coords
        if tree.root is None or isinstance(tree.root.data, int):
            RBTree.serializeTree(tree.root, result, 0, 0, empty, height)
            assert len(result) == size_of_tree
            # add to coords list
            output[coords_key].extend(result)
        else: # struct of arrays in yaml, write two serializations
            RBTree.serializeTree(tree.root, result, 0, 0, empty, height)
            # add to coords list
            output[coords_key].extend(result)

            # payloads
            result = list()

            RBTree.serializeTree(tree.root, result, 0, 1, empty, height)
            assert len(result) == size_of_tree
            
            # add to coords list
            output[payloads_key].extend(result)
        # explicit payloads for next level
        # return size of (serialized) tree representation
        return len(result)

    # get handle to the corresponding node
    def coordToHandle(self, coord):
        if self.tree.root is None:
            return None
        return self.tree.contains(coord)

    # iterator
    def nextInSlice(self):
        # self.printFiber()
        if self.num_to_ret is not None and self.num_to_ret < self.num_ret_so_far:
            return None
        print(self.coords_handle)
        # assert isinstance(self.coords_handle, RBNode)
        # if self.coords_handle is not None:
            # print("{} :: nextInSlice, cur coord {}".format(self.name, self.coords_handle.data))
        to_ret = self.coords_handle # keep track of current handle
        # print("next in slice to ret {}".format(to_ret))
        self.num_ret_so_far += 1
        
        # if you know where you are in the tree, you know where the successor is 
        # without having to read if you have to look right
        num_reads, self.coords_handle = self.tree.get_successor(to_ret) # advance handle
        # TODO: count number of accesses required to get successor
        return to_ret
    
    # handle to coord takes in a handle which is a node
    def handleToCoord(self, handle):
        if handle is None:
            return None
        # print("{} :: handleToCoord, handle {}".format(self.name, handle))
        if isinstance(handle, NilNode):
            return None
        
        assert isinstance(handle, RBNode)
        
        # count stats
        if handle is self.prevHandleAccessed:
            return self.prevCoordAtHandleAccessed
        # print("\t coords_read before increment {}".format(self.stats[self.coords_read_key]))
        self.stats[self.coords_read_key] += 1
        
        return handle.data[0]

    # given a handle (tree node ptr), update the payload there
    def handleToPayload(self, handle):
        # print("{} handleToPayload, handle = {}".format(self.name, handle))
        if handle is None:
            return None
        elif handle == self.prevPayloadHandle:
            return handle.data[-1]
        # TODO: add stats
        if self.count_payload_reads:
            # print("counting payloads read: handle {}, reads so far {}".format(handle, self.stats[self.payloads_read_key]))
            self.stats[self.payloads_read_key] += 1
        self.prevPayloadHandle = handle
        self.prevPayloadAtHandle = handle.data[-1]
        # print("handle.data = {}".format(handle.data))
        return handle.data[-1]

    def payloadToFiberHandle(self, handle):
        print("payload to fiber handle in T, ret {}".format(handle))
        return handle

    # return handle to inserted elt
    def insertElement(self, coord):
        print("insert {} into {}".format(coord, self.name))
        if coord is None:
            return None
        # TODO: stats

        # if cached, just write to the node at that handle
        if coord is self.prevCoordSearched:
            assert self.handleAtPrevCoordSearched.data[0] is coord
            return self.handleAtPrevCoordSearched
        # might have to do some reads and writes
        
        num_reads, num_writes, handle = self.tree.add([coord, 0])
        # print("insert into {}, {} :: num reads {}, num writes {}".format(self.name, coord, num_reads, num_writes))
        assert isinstance(handle, RBNode)
        self.stats[self.coords_read_key] += num_reads
        self.stats[self.coords_write_key] += num_writes

        # print("result after insert {}: {}".format(coord, handle))
        return handle
    # return a handle to the updated payload
    def updatePayload(self, handle, payload):
        if handle is None or handle is NIL:
            return None
        # print("update payload:: handle {}, payload {}".format(handle, payload))
        assert isinstance(handle, RBNode)
        if isinstance(payload, int):
            handle.data[1] = payload
        self.stats[self.payloads_write_key] += 1
        return handle

    # updated fiber handle returns (size of tree, internal fiber object)
    def getUpdatedFiberHandle(self):
        return (self.getSize(), self)

    # TODO: print the fiber (linearized?)
    def printFiber(self):
        output = list()
        RBTree.treeToString(self.tree.root, 0, RBTree.getHeight(self.tree.root), output)
        print("{} :: {}".format(self.name, output))

    # get size of the binary tree representation
    def getSize(self):
        height = RBTree.getHeight(self.tree.root)
        num_nodes = 2**height -1 
        node_size = self.tree.root.getSize()
        # print("tree get size, height {}, num nodes {}, node size {}".format(height, num_nodes, node_size))
        return num_nodes * node_size
    # encode coord explicitly
    @staticmethod
    def encodeCoord(prev_ind, ind):
        return [ind]

    @staticmethod
    def encodePayload(prev_ind, ind, payload):
        return [payload]

    # explicit coords
    @staticmethod
    def encodeCoords():
        return True

    # explicit prev payloads
    @staticmethod
    def encodeUpperPayload():
        return True
