import copy
import yaml

from .rank    import Rank
from .fiber   import Fiber
from .payload import Payload

""" Tensor """

class Tensor:
    """ Tensor Class """

    def __init__(self, yamlfile="", rank_ids=None, shape=None, name=""):
        """__init__"""

        self.yamlfile = yamlfile

        # TBD: Encourage use of Tensor.fromYAMLfile instead...

        if (yamlfile != ""):
            assert(rank_ids is None and shape is None)

            (rank_ids, root, shape, name) = self.parse(yamlfile)

            if shape is None:
                shape = root.estimateShape()

            self.setRankInfo(rank_ids, shape)
            self.setRoot(root)
            self.setName(name)
            self.setColor("red")
            self.setMutable(False)
            return

        #
        # Initialize an empty tensor with an empty root fiber
        #
        assert(not rank_ids is None)

        self.setRankInfo(rank_ids, shape)
        self.setName(name)
        self.setColor("red")
        self.setMutable(True)

        if rank_ids == []:
            # Create a rank zero tensor, i.e., just a payload

            self._root = Payload(0)
            return

        root_fiber = Fiber()
        self.setRoot(root_fiber)


    @classmethod
    def fromYAMLfile(cls, yamlfile):
        """Construct a Tensor from a YAML file"""

        (rank_ids, root, shape, name) = Tensor.parse(yamlfile)

        if not isinstance(root, Fiber):
            t = Tensor(rank_ids=[], shape=shape, name=name)
            t.setMutable(False)
            t._root = Payload(root)
            return t

        return Tensor.fromFiber(rank_ids, root, shape)


    @classmethod
    def fromUncompressed(cls, rank_ids=None, root=None, shape=None):
        """Construct a Tensor from uncompressed fiber tree"""

        assert(not root is None)

        if not isinstance(root, list):
            # Handle a rank zero tensor
            t = Tensor(rank_ids=[], shape=[])
            t._root = Payload(root)
            return t

        assert(not rank_ids is None)

        fiber = Fiber.fromUncompressed(root)

        if shape is None:
            # TBD: Maybe this is not needed because fibers get a max_coord...
            shape = Tensor._calc_shape(root)

        return Tensor.fromFiber(rank_ids, fiber, shape)


    @staticmethod
    def _calc_shape(ll):
        """_calc_shape"""

        shape = [len(ll)]

        if not isinstance(ll[0], list):
            return shape

        if len(ll) == 1:
            shape.extend(Tensor._calc_shape(ll[0]))
            return shape

        ll0 = Tensor._calc_shape(ll[0])
        ll1 = Tensor._calc_shape(ll[1:])[1:]
        rest = [max(a, b) for a, b in zip(ll0, ll1)]
        shape.extend(rest)

        return shape


    @classmethod
    def fromFiber(cls, rank_ids=None, fiber=None, shape=None, name=""):
        """Construct a Tensor from a fiber"""

        assert(not rank_ids is None)
        assert(not fiber is None)

        tensor = cls(rank_ids=rank_ids, shape=shape)

        tensor.setRoot(fiber)
        tensor.setName(name)
        tensor.setColor("red")
        tensor.setMutable(False)

        return tensor


    @classmethod
    def fromRandom(cls, rank_ids=None, shape=None, density=None, interval=10, seed=None):
        """ Create a random tensor

        Parameters:
        -----------

        rank_ids - list
        The "rank ids" for the tensor

        shape - list
        The "shape" (i.e., size) of each level of the tree

        density - list
        The probability that an element of the fiber will not be empty
        for each level of the tree

        interval - number
        The range (from 0 to "interval") of each value at the leaf of the tree

        seed - a valid argument for random.seed
        A seed to pass to random.seed

        """

        f = Fiber.fromRandom(shape, density, interval, seed)

        return Tensor.fromFiber(rank_ids=rank_ids, fiber=f, shape=shape)



#
# Accessor methods
#

    def setRankInfo(self, rank_ids, shape):
        """setRankInfo"""

        if shape is None:
            shape = [None]*len(rank_ids)

        #
        # Create a linked list of ranks
        #
        self.ranks = []
        last_rank = None

        for id, dimension in reversed(list(zip(rank_ids, shape))):
            new_rank = Rank(id=id, shape=dimension, next_rank=last_rank)
            self.ranks.insert(0, new_rank)
            last_rank = new_rank


    def syncRankInfo(self, ranks):
        """resyncRankInfo"""

        # TBD: Currently unused and untested, so probably broken

        self.ranks = []
        last_rank = None

        for rank in reversed(ranks):
            rank.set_next(last_rank)
            last_rank = rank


    def getRankIds(self):
        """getRankIds"""

        #
        # Get the rank id for each rank
        #
        return [ r.getId() for r in self.ranks ]


    def getShape(self):
        """getShape"""

        #
        # Get the shape for each rank
        #
        # TBD: Fix awkward interface to getShape
        #
        return [ r.getShape(all_ranks=False)[0] for r in self.ranks ]


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

        # Note: The code below handles the (probably abandoned)
        #       transistion from raw fibers as payloads to fibers in
        #       Payload

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


    def setName(self, name):
        """setName

        Set name for the tensor

        Parameters
        ----------
        name: string
        Name to use for tensor

        Returns
        -------
        self: So method can be used in a chain

        Raises
        ------
        None

        """

        self._name = name
        return self


    def getName(self):
        """Getname

        Get name of tensor

        Parameters
        ----------
        None

        Returns
        -------
        name: Name of tensor

        Raises
        ------
        None

        """

        return self._name


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


    def setDefault(self, value):
        """setDefault

        Set the default value for the leaf payloads of the tensor

        Parameters
        ----------
        value: value
        A value to use for leaf payload values in tensor

        Returns
        -------
        self:
        So method can be used in a chain

        Raises
        ------
        None

        """

        assert value != Fiber, "Leaf payloads cannot be a Fiber"

        #
        # Set default for leaf rank
        #
        self.ranks[-1].setDefault(value)

        return self


    def getDefault(self):
        """GetDefault

        Get the default payload for leaf ranks

        Parameters
        ----------
        None

        Returns
        -------
        value: value
        Default payload of the leaf rank

        Raises
        ------
        None

        """

        return self.ranks[-1].getDefault()


    def setMutable(self, value):
        """setDefault

        Set the "hint" as to whether the tensor is mutable or not,
        i.e., its value will change. Note: this property is not
        enforced, but is useful for the *Canvas methods that want to
        save the current value of the tensor, so they know if they
        need to copy the tensor or not.

        Parameters
        ----------
        value: Bool
        Is the tensor mutable or not.

        Returns
        -------
        self:
        So method can be used in a chain

        Raises
        ------
        None

        """

        self._mutable = value

        return self


    def isMutable(self):
        """isMutable

        Returns the "hint" that the tensor is mutable

        Parameters
        ----------
        None

        Returns
        -------
        value: Bool
        Whether the tensor is set mutable or not

        Raises
        ------
        None

        """

        return self._mutable


    def countValues(self):
        """countValues

        Count of non-empty payload values in the leaf rank of tensor

        """

        return self.getRoot().countValues()

#
#  Comparison operations
#

    def __eq__(self, other):
        """__eq__

        Check for equivalence of two tensors by matching their rank
        ids and root fiber.

        Note: The tenor's names and colors do not need to match

        """

        rankid_match = (self.getRankIds() == other.getRankIds())
        fiber_match = (self.getRoot() == other.getRoot())

        return  rankid_match and fiber_match


#
# Tensor equivalent of Fiber methods where operating on the
# root fiber is the logical activity
#

    def getPayload(self, *args, **kwargs):
        """getPayload"""

        return self.getRoot().getPayload(*args, **kwargs)


    def getPayloadRef(self, *args, **kwargs):
        """getPayload"""

        return self.getRoot().getPayloadRef(*args, **kwargs)


    def countValues(self, *args, **kwargs):
        """getPayload"""

        return self.getRoot().countValues(*args, **kwargs)


    def __iter__(self):
        """__iter__"""

        return self.getRoot().__iter__()


    def __reversed__(self):
        """Return reversed fiber"""

        return self.getRoot().__reversed__()



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
        rank_ids = copy.deepcopy(self.getRankIds())
        id = rank_ids[depth]
        rank_ids[depth] = f"{id}.1"
        rank_ids.insert(depth+1, f"{id}.0")

        #
        # Create new shape list
        #
        # TBD: Create shape
        #
        shape = None

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
        tensor = Tensor.fromFiber(rank_ids, root, shape)
        tensor.setName(self.getName()+"+split")
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
        rank_ids = copy.deepcopy(self.getRankIds())
        id = rank_ids[depth]
        rank_ids[depth] = rank_ids[depth+1]
        rank_ids[depth+1] = id

        #
        # Create new shape list
        #
        # TBD: Create shape
        #
        shape = None

        root = self._modifyRoot(Fiber.swapRanks,
                                Fiber.swapRanksBelow,
                                depth=depth)
        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root, shape)
        tensor.setName(self.getName()+"+swapped")
        tensor.setColor(self.getColor())

        return tensor


    def flattenRanks(self, depth=0, levels=1):
        """ swapRanks """

        #
        # Create new list of rank ids
        #
        # Note: we need to handle the case where existing ranks are lists
        #
        rank_ids = copy.deepcopy(self.getRankIds())

        cur_rankid= rank_ids[depth]
        if not isinstance(cur_rankid, list):
            rank_ids[depth] = []
            rank_ids[depth].append(cur_rankid)

        for d in range(levels):
            next_rankid = rank_ids[depth+1]

            if isinstance(next_rankid, list):
                rank_ids[depth] = cur_rankid + next_rankid
            else:
                rank_ids[depth].append(next_rankid)

            del rank_ids[depth+1]

        #
        # Create new shape list
        #
        # TBD: Create shape
        #
        shape = None

        root = self._modifyRoot(Fiber.flattenRanks,
                                Fiber.flattenRanksBelow,
                                depth=depth,
                                levels=levels)
        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root, shape)
        tensor.setName(self.getName()+"+flattened")
        tensor.setColor(self.getColor())

        return tensor


    def unflattenRanks(self, depth=0, levels=1):
        """ swapRanks """

        #
        # Create new list of rank ids
        #
        rank_ids = copy.deepcopy(self.getRankIds())

        for d in range(levels):
            id = rank_ids[depth+d]
            rank_ids[depth+d] = id[0]
            if len(id) == 2:
                rank_ids.insert(depth+d+1, id[1])
            else:
                rank_ids.insert(depth+d+1, id[1:])

        #
        # Create new shape list
        #
        # TBD: Create shape
        #
        shape = None

        root = self._modifyRoot(Fiber.unflattenRanks,
                                Fiber.unflattenRanksBelow,
                                depth=depth,
                                levels=levels)
        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root, shape)
        tensor.setName(self.getName()+"+unflattened")
        tensor.setColor(self.getColor())

        return tensor


    def _modifyRoot(self, func, funcBelow, depth=0, **kwargs):
        #
        # Create new root fiber
        #
        root_copy = copy.deepcopy(self.getRoot())
        if depth == 0:
            root = func(root_copy, **kwargs)
        else:
            root = root_copy
            funcBelow(root, depth=depth-1, **kwargs)

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


    def __format__(self, format):
        """__format__"""

        #
        # Just format the root fiber
        #
        return self.getRoot().__format__(format)


    def __str__(self):
        """_str__"""

        # TBD: Fix to use a format from a fiber...
        
        str = "T(%s)/[" % ",".join(self.getRankIds())

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

        # TBD: Fix to use a repr from a fiber...

        str = "T(%s)/[" % ",".join(self.getRankIds())

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
        # Get shape information
        #
        if 'shape' in y_tensor:
            shape = y_tensor['shape']
        else:
            shape = None

        #
        # Get tensor name
        #
        if 'name' in y_tensor:
            name = y_tensor['name']
        else:
            # TBD: Maybe extract something from filename
            name = ""

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

        return (rank_ids, fiber, shape, name)


    def dump(self, filename):
        """Dump a tensor to a file in YAML format"""

        root = self.getRoot()
        
        if isinstance(root, Payload):
            root_dict = Payload.payload2dict(root)
        else:
            root_dict = root.fiber2dict()

        tensor_dict = {'tensor':
                       {'rank_ids': self.getRankIds(),
                        'shape': self.getShape(),
                        'name': self.getName(),
                        'root': [root_dict]
                       }}

        with open(filename, 'w') as file:
            document = yaml.dump(tensor_dict, file)

#
# Utility methods
#

    @staticmethod
    def _deprecated(message):
        import warnings

        warnings.warn(message, FutureWarning, stacklevel=3)

