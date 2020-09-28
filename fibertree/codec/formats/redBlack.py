#!/usr/bin/python

class NilNode(object):
    """
    The nil class is specifically for balancing a tree by giving all  traditional leaf noes tw children that are null
     and waiting to be filled
    """
    def __init__(self):
        self.red = False
        self.size = 0
        self.data = None

NIL = NilNode() # Nil is the sentinel value for nodes


class RBNode(object):
    """
    Class for implementing the nodes that the tree will use
    For self.red:
        red == True
        black == False
        If the node is a leaf it will either
    """
    def __init__(self,data):
        self.red = True
        self.parent = None
        self.data = data
        self.left = NIL
        self.right = NIL
        self.size = 1

    # TODO: current leaving out the bit that says whether you are red or black
    def getSize(self):
        base = 3 # parent, left, right
        data_size = len(self.data) - 1 
        return base + data_size
class RedBlackTree(object):
    """
    Class for implementing a standard red-black tree
    """
    def __init__(self):
        self.root = None
        self.size = 0
        self.lastNodeAccessed = None

    # add a new node with data 
    # also return number of reads, writes
    def add(self,data,cache=None,name=None,curr = None):
        """
        :param data: an int, float, or any other comparable value
        :param curr:
        """
        # print("\t add to RB, {}".format(data))
        self.size += 1
        new_node = RBNode(data)
        # Base Case - Nothing in the tree
        if self.root == None:
            new_node.red = True
            self.root = new_node
            return 1, 1, new_node
       
        # Search to find the node's correct place
        currentNode = self.root
        num_reads = 0
        num_writes = 0
        current_dram = None
        if cache is not None:
            current_dram = cache.miss_count
        while currentNode != NIL:
            # increment size along root-to-leaf path
            currentNode.size += 1
            # print("current node data {}, size {}".format(currentNode.data, currentNode.size))
            potentialParent = currentNode
            num_reads += 1
            if cache is not None:
                key = name + '_coordToHandle_' + str(currentNode.data[0])
                res = cache.get(key) # check if its in the cache during the search
                print("\tin add, key {}, res {}".format(key, res))
            if new_node.data[0] == currentNode.data[0]:
                # go back up the tree and fix sizes
                temp = currentNode
                while temp is not None and temp != NIL:
                    temp.size -= 1
                    temp = temp.parent
                print("\t\tfound {}".format(new_node.data[0]))
                return num_reads, 0, currentNode
            if new_node.data[0] < currentNode.data[0]:
                currentNode = currentNode.left
            else:
                currentNode = currentNode.right

        # Assign parents and siblings to the new node
        new_node.parent = potentialParent
        if cache is not None:
            key = name + '_coordToHandle_' + str(new_node.parent.data[0])    
            res = cache.get(key) # check if its in the cache during the search
            print("\tin add, key {}, res {}".format(key, res))

        if new_node.data[0] < new_node.parent.data[0]:
            new_node.parent.left = new_node
        else:
            new_node.parent.right = new_node
        if cache is not None:
            key = name + '_coordToHandle_' + str(data)
            res = cache.get(key)
            cache[key] = new_node
            assert(res == None)
        
        # TODO: get num writes from fix tree after add
        num_writes = self.fix_tree_after_add(new_node,cache,name)
        num_writes += 1
        # print("\tinsert {}, reads {}, writes {}".format(data, num_reads, num_writes))
        assert(self.root.red == False)
        if cache is not None:
            assert cache.miss_count != current_dram
        return num_reads, num_writes, new_node # return handle to this node that was just added

    # search on coord and return the node that contains the elt
    # TODO: make sure that this is ok if coord is not present
    def contains(self,data, curr=None):
        """
        :return:
        """
        if curr == None:
            curr = self.root
        prev_parent = None
        while curr != NIL and data != curr.data[0]:
            # print("searching for {}, curr {}".format(data, curr.data[0]))
            prev_parent = curr
            if data < curr.data[0]:

                curr = curr.left
            else:
                curr = curr.right
        if isinstance(curr, NilNode):
            curr = prev_parent
        return curr

    def getRank(self,data):
        """

        :return:
        """
        curr = self.root
        result = 0
        prev_added = 0
        print("\tin getRank of {}, curr {}".format(data,curr.data))
        while curr != NIL and data != curr.data[0]:
            printf("find rank of {}, curr {}, curr size {}".format(data, curr.data[0], curr.size))
            if data < curr.data[0]:
                result -= curr.size
                curr = curr.left
            else:
                result += curr.size
                prev_added = curr.size
                curr = curr.right

            print("\trank {}".format(result))
        return result


    def fix_tree_after_add(self,new_node,cache=None,name=None):
        """
        This method is meant to check and rebalnce a tree back to satisfying all of the red-black properties
        :return: num additional reads / writes (assume we have new_node and new_node.parent)
        modifies tree
        """
        # print("new_node parent {}".format(new_node.parent.red))
        # print("new node data {}, parent {}, root data {}".format(new_node.data, new_node.parent.data, self.root.data))
        # print("\tin fix tree after add")
        num_writes = 0
        while new_node != self.root and new_node.parent.red == True and new_node.parent.parent is not None:
            if cache is not None:
                key1 = name + "_coordToHandle_" + str(new_node.parent.data[0])
                cache.get(key1)
                key2 = name + "_coordToHandle_" + str(new_node.parent.parent.data[0])
                cache.get(key2)

            # if you are in the left subtree
            if new_node.parent == new_node.parent.parent.left:
                uncle = new_node.parent.parent.right
                if cache is not None:
                    key = name + "_coordToHandle_" + str(uncle.data[0])
                    cache.get(key)
                if uncle.red:
                    # This is Case 1
                    new_node.parent.red = False
                    uncle.red = False
                    new_node.parent.parent.red = True
                    new_node = new_node.parent.parent
                    num_writes += 4
                    print("\t\tcase 1")
                else:
                    if new_node == new_node.parent.right:
                        # This is Case 2
                        new_node = new_node.parent
                        self.left_rotate(new_node)
                        # print("\t\tcase 2")
                    # This is Case 3
                    new_node.parent.red = False
                    new_node.parent.parent.red = True
                    self.right_rotate(new_node.parent.parent)
                    num_writes += 3
                    # print("\t\tcase 3")
            else:
                uncle = new_node.parent.parent.left
                if cache is not None and uncle.data is not None:
                    key = name + "_coordToHandle_" + str(uncle.data[0])
                    cache.get(key)
                if uncle.red:
                    # Case 1
                    new_node.parent.red = False
                    uncle.red = False
                    new_node.parent.parent.red = True
                    new_node = new_node.parent.parent
                    num_writes += 4
                    # print("\t\tcase 1b")
                else:
                    if new_node == new_node.parent.left:
                        # Case 2
                        new_node = new_node.parent
                        # print("second right rotate")
                        self.right_rotate(new_node)
                       #  print("\t\tcase 2b")
                    # Case 3
                    new_node.parent.red = False
                    new_node.parent.parent.red = True
                    # left rotate writes to input and one other node
                    self.left_rotate(new_node.parent.parent)
                    num_writes += 3
                    
                    # print("\t\tcase 3b")
            # print("new node {}".format(new_node.data))
        self.root.red = False
        return num_writes

    def delete(self):
        """

        :return:
        """
        pass
    def left_rotate(self,new_node):
        """

        :return:
        """
        # print("Rotating left!")
        sibling = new_node.right
        new_node.right = sibling.left

        # print("new node {}, sibling {}".format(new_node.data, sibling.data))
        # Turn sibling's left subtree into node's right subtree
        if sibling.left is not None:
            sibling.left.parent = new_node
        sibling.parent = new_node.parent
        if new_node.parent == None:
            self.root = sibling
        else:
            if new_node == new_node.parent.left:
                new_node.parent.left = sibling
            else:
                new_node.parent.right = sibling
        # from clrs
        sibling.size = new_node.size
        new_node.size = new_node.left.size + new_node.right.size + 1

        sibling.left = new_node
        new_node.parent = sibling


    def right_rotate(self,new_node):
        """

        :return:
        """
       #  print("Rotating right!")
        sibling = new_node.left
        new_node.left = sibling.right
        # print("new_node data {}, sibling {}".format(new_node.data, sibling.data))
        self.inorder(new_node.parent)
        print("\n")
        # Turn sibling's left subtree into node's right subtree
        if sibling.right is not None:
            sibling.right.parent = new_node
        sibling.parent = new_node.parent
        if new_node.parent == None:
            self.root = sibling
        else:
            if new_node == new_node.parent.right:
                new_node.parent.right = sibling
            else:
                new_node.parent.left = sibling
        sibling.right = new_node
        new_node.parent = sibling
        self.inorder(new_node.parent)
        # from clrs
        new_node.size = sibling.size
        sibling.size = sibling.left.size + sibling.right.size + 1

    def inorder(self, root):
        # Base Case - Nothing in the tree
        if root == None or root == NIL:
            print("NIL")
            return
        
        print("data {}, size {}, color {}".format(root.data, root.size, root.red))
        self.inorder(root.left)
        self.inorder(root.right)
        return

    def get_all_nodes(self):
        """

        :return:
        """
        pass
    def is_red(self):
        """
        This is the class that usually decides that a node is wither red or black, some implementations take the ecurrtra
        bit and will implement an is_black method for additional clarity.
        Generally, True == Red and False == Black

        :return:
        """
        return self.root is not None and self.root.red == 1;
    def is_black(self):
        """
        Note that this method is not necessary as some implementations only check is the is_red class method is True or False
        :return:
        True if the node is black or is a leaf
        """
        return self.root is not None and self.root.black == 1;

    # min-val is down the left spine if it exists
    def min_val(self, root, cache, name):
        p = root
        if p == NIL:
            return p
        node = root.left
        num_reads = 0

        key = name + "_coordToHandle_" + str(p.data[0])
        res = cache.get(key)
        cache[key] = p
        while node != NIL:
            key = name + "_coordToHandle_" + str(node.data[0])
            res = cache.get(key)
            cache[key] = node

            p = node
            node = node.left
            num_reads += 1
            # print("\tnum reads in min_val {}, current node {}".format(num_reads, node.left))
        return num_reads, p

    # given a node, find its successor 
    def get_successor(self, root, cache, name):
        # if successor is in the right subtree
        if root == None or root == NIL:
            return 0, None
        # if there is a right subtree, find the min value in it
        if root.right is not None and root.right != NIL:
            # print("\t\t going right")
            return self.min_val(root.right, cache, name)
        
        # else successor is higher up in the tree
        p = root.parent
        num_reads = 0
        while p is not None and p != NIL:
            if root != p.right:
                break
            root = p
            p = p.parent
            if p is not None and p != NIL:
                # look for the node in the cache
                key = name + "_coordToHandle_" + str(p.data[0])
                res = cache.get(key)
                cache[key] = p
                num_reads += 1
            # print("\tget_successor: num reads going up tree {}, node {}".format(num_reads, p))
        return num_reads, p

if __name__ == "__main__":
    tree = RedBlackTree()
    tree.add(1)
    # print(tree.root)
    tree.inorder(tree.root)
    tree.add(3)
    tree.inorder(tree.root)
    tree.add(4)
    tree.inorder(tree.root)
    tree.add(8)
    tree.inorder(tree.root)
    tree.add(5)
    # tree.add(6)
    # tree.add(2)
    tree.inorder(tree.root)
