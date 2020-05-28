import copy
import yaml

from fibertree.rank    import Rank
from fibertree.fiber   import Fiber
from fibertree.payload import Payload

""" Tensor """

class Tensor:
    """ Tensor Class """

    def __init__(self, yamlfile="", rank_ids=None):
        """__init__"""

        self.yamlfile = yamlfile

        # TBD: Encourage use of Tensor.fromYAMLfile instead...

        if (yamlfile != ""):
            assert(rank_ids is None)

            (rank_ids, fiber) = self.parse(yamlfile)

            self.set_rank_ids(rank_ids)
            self.setColor("red")
            self.setRoot(fiber)
            return

        #
        # Initialize an empty tensor with an empty root fiber
        #
        assert(not rank_ids is None)

        self.set_rank_ids(rank_ids)
        self.setColor("red")

        if rank_ids == []:
            # Create a rank zero tensor, i.e., just a payload

            self._root = Payload(0)
            return

        root_fiber = Fiber()
        self.setRoot(root_fiber)


    @classmethod
    def fromYAMLfile(cls, yamlfile):
        """Construct a Tensor from a YAML file"""

        (rank_ids, root) = Tensor.parse(yamlfile)

        if not isinstance(root, Fiber):
            t = Tensor(rank_ids=[])
            t._root = Payload(root)
            return t

        return Tensor.fromFiber(rank_ids, root)


    @classmethod
    def fromUncompressed(cls, rank_ids=None, root=None):
        """Construct a Tensor from uncompressed fiber tree"""

        assert(not root is None)

        if not isinstance(root, list):
            # Handle a rank zero tensor
            t = Tensor(rank_ids=[])
            t._root = Payload(root)
            return t

        assert(not rank_ids is None)

        fiber = Fiber.fromUncompressed(root)
        return Tensor.fromFiber(rank_ids, fiber)


    @classmethod
    def fromFiber(cls, rank_ids=None, fiber=None):
        """Construct a Tensor from a fiber"""

        assert(not rank_ids is None)
        assert(not fiber is None)

        tensor = cls(rank_ids=rank_ids)

        tensor.setColor("red")
        tensor.setRoot(fiber)

        return tensor


#
# Accessor methods
#

    # TBD: Fix style of this method name

    def set_rank_ids(self, rank_ids):
        """set_rank_ids"""

        self.rank_ids = rank_ids

        #
        # Create a linked list of ranks
        #
        self.ranks = []
        for id in rank_ids:
            new_rank = Rank(name=id)
            self.ranks.append(new_rank)

        old_rank = None
        for rank in self.ranks:
            if not old_rank is None:
                old_rank.set_next(rank)
            old_rank = rank


    def setRoot(self, root):
        """(Re-)populate self.ranks with "root"""

        # Note: rank 0 tensors are not allowed in this path
        assert isinstance(root, Fiber)

        self._root = root

        # Clear out existing rank information
        for r in self.ranks:
            r.clearFibers()

        self._addFiber(root)


    def _addFiber(self, fiber, level=0):
        """Recursively fill in ranks from "fiber"."""

        self.ranks[level].append(fiber)

        # Note: The code below handles the transistion from
        #       raw fibers as payloads to fibers in Payload

        for p in fiber.getPayloads():
            if Payload.contains(p, Fiber):
                self._addFiber(Payload.get(p), level+1)


    def getRoot(self):
        """root"""

        root = self._root

        # Either we have a 0-D tensor or the root is a Fiber
        assert (isinstance(root, Payload) or
                root == self.ranks[0].getFibers()[0])

        return root


    def root(self):
        """root"""

        Tensor._deprecated("Tensor.root() is deprecated, use getRoot()")

        return self.getRoot()


    def setColor(self, color):
        """setColor

        Set color for elements of tensor

        Parameters
        ----------
        color: Color to use for scalar values in tensor

        Returns
        -------
        self: So method can be used in a chain

        Raises
        ------
        None

        """

        self._color = color
        return self


    def getColor(self):
        """Getcolor

        Get color for elements of tensor

        Parameters
        ----------
        None

        Returns
        -------
        color: Color to use for scalar values in tensor

        Raises
        ------
        None

        """

        return self._color


    def countValues(self):
        """Count of values in the tensor"""

        return self.getRoot().countValues()

#
#  Comparison operations
#

    def __eq__(self, other):
        """__eq__"""

        return (self.rank_ids == other.rank_ids) and (self.getRoot() == other.getRoot())


#
# Split methods
#
# Note: all these methods return a new tensor
# TBD: Allow depth to be specified by rank_id
#

    def splitUniform(self, *args, depth=0, **kwargs):
        """ splitUniform """

        return self._splitGeneric(Fiber.splitUniform,
                                  Fiber.splitUniformBelow,
                                  *args,
                                  depth=depth,
                                  **kwargs)

    def splitNonUniform(self, *args, depth=0, **kwargs):
        """ splitNonUniform """

        return self._splitGeneric(Fiber.splitNonUniform,
                                  Fiber.splitNonUniformBelow,
                                  *args,
                                  depth=depth,
                                  **kwargs)


    def splitEqual(self, *args, depth=0, **kwargs):
        """ splitEqual """

        return self._splitGeneric(Fiber.splitEqual,
                                  Fiber.splitEqualBelow,
                                  *args,
                                  depth=depth,
                                  **kwargs)

    def splitUnEqual(self, *args, depth=0, **kwargs):
        """ splitUnEqual """

        return self._splitGeneric(Fiber.splitUnEqual,
                                  Fiber.splitUnEqualBelow,
                                  *args,
                                  depth=depth,
                                  **kwargs)


    def _splitGeneric(self, func, funcBelow, *args, depth=0, **kwargs):
        """ _splitGeneric... """

        #
        # Create new list of rank ids
        #
        rank_ids = copy.deepcopy(self.rank_ids)
        id = rank_ids[depth]
        rank_ids[depth] = f"{id}.1"
        rank_ids.insert(depth+1, f"{id}.0")

        #
        # Create new root fiber
        #
        root_copy = copy.deepcopy(self.getRoot())
        if depth == 0:
            root = func(root_copy, *args, **kwargs)
        else:
            root = root_copy
            funcBelow(root, *args, depth=depth-1, **kwargs)

        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root)
        tensor.setColor(self.getColor())

        return tensor

#
# Swap method
#
    def swapRanks(self, depth=0):
        """ swapRanks """

        #
        # Create new list of rank ids
        #
        rank_ids = copy.deepcopy(self.rank_ids)
        id = rank_ids[depth]
        rank_ids[depth] = rank_ids[depth+1]
        rank_ids[depth+1] = id

        root = self._modifyRoot(Fiber.swapRanks,
                                Fiber.swapRanksBelow,
                                depth=depth)
        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root)
        tensor.setColor(self.getColor())

        return tensor


    def flattenRanks(self, depth=0):
        """ swapRanks """

        #
        # Create new list of rank ids
        #
        rank_ids = copy.deepcopy(self.rank_ids)
        rank_ids[depth] = [rank_ids[depth], rank_ids[depth+1]]
        del rank_ids[depth+1]

        root = self._modifyRoot(Fiber.flattenRanks,
                                Fiber.flattenRanksBelow,
                                depth=depth)
        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root)
        tensor.setColor(self.getColor())

        return tensor


    def unflattenRanks(self, depth=0):
        """ swapRanks """

        #
        # Create new list of rank ids
        #
        rank_ids = copy.deepcopy(self.rank_ids)
        id = rank_ids[depth]
        rank_ids[depth] = id[0]
        rank_ids.insert(depth+1, id[1])

        root = self._modifyRoot(Fiber.unflattenRanks,
                                Fiber.unflattenRanksBelow,
                                depth=depth)
        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root)
        tensor.setColor(self.getColor())

        return tensor


    def _modifyRoot(self, func, funcBelow, depth=0):
        #
        # Create new root fiber
        #
        root_copy = copy.deepcopy(self.getRoot())
        if depth == 0:
            root = func(root_copy)
        else:
            root = root_copy
            funcBelow(root, depth=depth-1)

        #
        # Create Tensor from rank_ids and root fiber
        #
        return root


#
# String methods
#
    def print(self, title=None):
        """print"""

        if not title is None:
            print("%s" % title)

        print("%s" % self)
        print("")

    def __str__(self):
        
        str = "T(%s)/[" % ",".join(self.rank_ids)

        if self.ranks:
            str += "\n"
            for r in self.ranks:
                str += r.__str__(indent=2) + "\n"
        else:
            root = self.getRoot()
            fmt = "n*" if isinstance(root, Fiber) else ""
            str += f"{root:{fmt}}"

        str += "]"
        return str
        
    def __repr__(self):
        """__repr__"""

        str = "T(%s)/[" % ",".join(self.rank_ids)

        if self.ranks:
            str += "\n"
            for r in self.ranks:
                str += "  " + repr(r) + "\n"
        else:
            str += repr(self.getRoot())

        str += "]"

        return str

#
# Yaml input/output methods
#

    @staticmethod
    def parse(file):
        """Parse a yaml file containing a tensor"""

        with open(file, 'r') as stream:
            try:
                y_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)

        #
        # Make sure key "tensor" exists
        #
        if not isinstance(y_file, dict) or 'tensor' not in y_file:
            print("Yaml is not a tensor")
            exit(1)

        y_tensor = y_file['tensor']

        #
        # Make sure key "rank_ids" exists
        #
        if not isinstance(y_tensor, dict) or 'rank_ids' not in y_tensor:
            print("Yaml has no rank_ids")
            exit(1)

        rank_ids = y_tensor['rank_ids']

        #
        # Make sure key "root" exists
        #
        if 'root' not in y_tensor:
            print("Yaml has no root")
            exit(1)

        y_root = y_tensor['root']

        #
        # Generate the tree recursively
        #   Note: fibers are added into self.ranks inside method
        #
        fiber = Fiber.dict2fiber(y_root[0])

        return (rank_ids, fiber)


    def dump(self, filename):
        """Dump a tensor to a file in YAML format"""

        root = self.getRoot()
        
        if isinstance(root, Payload):
            root_dict = Payload.payload2dict(root)
        else:
            root_dict = root.fiber2dict()

        tensor_dict = { 'tensor':
                        { 'rank_ids': self.rank_ids,
                          'root': [ root_dict ]
                        } }
        with open(filename, 'w') as file:
            document = yaml.dump(tensor_dict, file)

#
# Utility methods
#

    @staticmethod
    def _deprecated(message):
        import warnings

        warnings.warn(message, FutureWarning, stacklevel=3)

