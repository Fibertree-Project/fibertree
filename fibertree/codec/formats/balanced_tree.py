from .compression_format import CompressionFormat
from .redBlack import *

class RBTree(CompressionFormat):
    def __init__(self):
        CompressionFormat.__init__(self)

        self.curHandle = None
    
    # given a node as input, compute the height of that node
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
        if node == None or node == NIL:
            return
        self.getPayloadsHelper(node.left, depth +1, height, output)
        output.append(node.data[1])
        self.getPayloadsHelper(node.right, depth + 1, height, output)
        return

    # inorder traversal of the tree to serialize payloads
    def getPayloads(self):
        output = list()

        height = RBTree.getHeight(self.tree.root)
        self.getPayloadsHelper(self.tree.root, 0, height, output)
        return output

    # encode fiber into T format
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output, output_tensor, shape=None):
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
                fiber, child_occupancy = codec.encode(depth + 1, val, ranks, output, output_tensor, shape)
                
                if isinstance(cumulative_occupancy, int):
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                else:
                    cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
                # encode (coord, payload)
                if codec.fmts[depth + 1].encodeUpperPayload():
                    if codec.cumulative_payloads[depth]:
                        self.tree.add((ind, cumulative_occupancy, ind))
                    else:
                        self.tree.add((ind, child_occupancy, ind)) 
                else: # if a leaf, encode (coord, value)
                    self.tree.add(ind, ind)
            else:
                self.tree.add((ind, val.value))
            
            # search for it in the tree for verification
            # assert ind == self.tree.contains(ind).data[0]

            fiber_occupancy = fiber_occupancy + 1
            
        # serialize tree
        # null value for empty nodes
        empty = -1

        tree = self.tree
        result = list()
        height = RBTree.getHeight(tree.root)
        size_of_tree = 2**height - 1
        
        # serialize only coords
        if tree.root == None or isinstance(tree.root.data, int):
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
        if self.tree.root == None:
            return None
        return self.tree.contains(coord)

    # slice on coordinates
    def setupSlice(self, base = 0, bound = None, max_num = None):
        self.num_ret_so_far = 0
        self.num_to_ret = max_num
        self.base = base
        self.bound = bound
        res = self.coordToHandle(base)
        if res != None and not isinstance(res, NilNode):
            key = self.name + "_coordToHandle_" + str(res.data[0])
            # map coord to node
            self.cache.get(key)
            self.cache[key] = res
            self.curHandle = res.data[0]
    
    # iterator
    def nextInSlice(self):
        # self.printFiber()
        if self.num_to_ret != None and self.num_to_ret < self.num_ret_so_far:
            return None
        
        to_ret = self.curHandle  # keep track of current handle
        self.num_ret_so_far += 1
        
        if to_ret == None or isinstance(to_ret, NilNode):
            return None

        key = self.name + "_coordToHandle_" + str(self.curHandle)
        print(key)
        node_at_cur_handle = self.cache.get(key)
        assert(node_at_cur_handle != None)
        # if you know where you are in the tree, you know where the successor is 
        # without having to read if you have to look right
        num_reads, node_at_next_handle = self.tree.get_successor(node_at_cur_handle, self.cache, self.name)

        if node_at_next_handle == None:
            self.curHandle = None
        
        elif not isinstance(node_at_next_handle, NilNode):
            self.curHandle = node_at_next_handle.data[0]
            key = self.name + "_coordToHandle_" + str(self.curHandle)
            self.cache.get(key)
            self.cache[key] = node_at_next_handle
        print("\t{} nextInSlice, current handle {}, to ret {}".format(self.name, self.curHandle, to_ret))
        if self.curHandle != None and to_ret != None:
            assert self.curHandle != to_ret # make sure you advance
        # self.printFiber()

        return to_ret
    
    # handle to coord takes in a handle which is a node
    def handleToCoord(self, handle):
        if handle == None:
            return None
        print("\t\tin tree {} handleToCoord: handle {}, curHandle {}".format(self.name, handle, self.curHandle))

        self.stats[self.coords_read_key] += 1
        return handle

    # given a handle (tree node ptr), update the payload there
    def handleToPayload(self, handle):
        if handle == None:
            return None
        if self.count_payload_reads:
            self.stats[self.payloads_read_key] += 1
        key = self.name + "_coordToHandle_" + str(handle)
        self.cache.get(key)
        
        node = self.cache[key]
        assert(node != None)
        # print("{} handleToPayload: node {}, handle {}".format(self.name, node, handle))
        return node.data[-1]
    
    def payloadToValue(self, payload):
        # print("\t{}: payloadToValue in T, payloads {}, payload {}".format(self.name, self.getPayloads(), payload))
        return payload

    def payloadToFiberHandle(self, handle):
        # print("\tpayload to fiber handle in T, ret {}".format(handle))
        return handle

    # return handle to inserted elt
    # make the handle the coord
    def insertElement(self, coord):
        if coord == None:
            return None
        # print("{} insertElement {}".format(self.name, coord))
        assert self.cache is not None
        num_reads, num_writes, handle = self.tree.add([coord, 0], cache=self.cache, name=self.name)
        
        # handle must be something that can index into a list, we want the i-th
        assert isinstance(handle, RBNode)
        self.stats[self.coords_read_key] += num_reads
        self.stats[self.coords_write_key] += num_writes

        # handle needs to be indexable
        key = self.name + "_coordToHandle_" + str(coord)
        self.cache.get(key)
        self.cache[key] = handle
        print("{} tree insertElt {}, misses {}".format(self.name, coord, self.cache.miss_count))
        print(self.cache)
        return coord # self.curHandle
    
    # return a handle to the updated payload
    def updatePayload(self, handle, payload):
        print("{} updatePayload: handle {}, payload {}".format(self.name, handle, payload))
        if handle == None or handle == NIL:
            return None
        # print("update payload:: handle {}, payload {}".format(handle, payload))
        # assert handle is self.curHandle
        key = self.name + "_coordToHandle_" + str(handle)
        node_at_handle = self.cache.get(key)
        if node_at_handle != None:
            assert node_at_handle.data[0] == handle
            node_at_handle.data[1] = payload
        
        self.stats[self.payloads_write_key] += 1
        return handle

    # updated fiber handle returns (size of tree, internal fiber object)
    def getUpdatedFiberHandle(self):
        return self.getSize()

    def printFiber(self):
        output = list()
        RBTree.treeToString(self.tree.root, 0, RBTree.getHeight(self.tree.root), output)
        print("{} :: {}".format(self.name, output))

    # get size of the binary tree representation
    def getSize(self):
        height = RBTree.getHeight(self.tree.root)
        num_nodes = 2**height -1 
        node_size = 0
        if self.tree.root != None:
            self.tree.root.getSize()
        
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
