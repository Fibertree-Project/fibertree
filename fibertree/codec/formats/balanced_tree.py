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
        coords_key = "coords_{}".format(ranks[depth].lower())
        payloads_key = "payloads_{}".format(ranks[depth].lower())

        tree = RedBlackTree()

        # init vars
        fiber_occupancy = 0
        cumulative_occupancy = 0
        if depth < len(ranks) - 1:
            if codec.format_descriptor[depth + 1] is "Hf" or codec.format_descriptor[depth+1] is "T":
    	        cumulative_occupancy = [0, 0] 
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
                    tree.add((ind, cumulative_occupancy))
                else:
                    tree.add(ind)
            else:
                tree.add((ind, val.value))
            fiber_occupancy = fiber_occupancy + 1
            
        # serialize tree
        empty = -1

        # struct of arrays
        result = list()
        height = RBTree.getHeight(tree.root)
        size_of_tree = 2**height - 1
        if tree.root is None or isinstance(tree.root.data, int):
            RBTree.serializeTree(tree.root, result, 0, 0, empty, height)
            assert len(result) == size_of_tree
            # add to coords list
            output[coords_key].extend(result)
        else: 
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
        return len(result), occ_list

        # return [fiber_occupancy, len(result)], occ_list

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
