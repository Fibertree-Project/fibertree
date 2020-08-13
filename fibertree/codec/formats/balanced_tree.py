from .compression_format import CompressionFormat
from .redBlack import *

class RBTree(CompressionFormat):
    def __init__(self):
        self.name = "T"

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

    @staticmethod
    def encodeFiber(a, dim_len, codec, depth, ranks, output):
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth)

        tree = RedBlackTree()

        # init vars
        fiber_occupancy = 0

        cumulative_occupancy = codec.get_start_occ(depth)
        occ_list = list()
        occ_list.append(cumulative_occupancy)
        prev_nz = 0
        
        for ind, (val) in a:
            child_occupancy = codec.encode(depth + 1, val, ranks, output)
            # keep track of actual occupancy (nnz in this fiber)
            
            if isinstance(cumulative_occupancy, int):
                cumulative_occupancy = cumulative_occupancy + child_occupancy
            else:
                cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
            
            if depth < len(ranks) - 1:
                if codec.fmts[depth + 1].encodeUpperPayload():
                    if codec.cumulative_payloads[depth]:
                        tree.add((ind, cumulative_occupancy))
                    else:
                        tree.add((ind, child_occupancy))
                else:
                    tree.add(ind)
            else:
                tree.add((ind, val.value))
            fiber_occupancy = fiber_occupancy + 1
            
        # serialize tree
        # null value for empty nodes
        empty = -1

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
        # return size of tree representation
        return len(result), occ_list

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
