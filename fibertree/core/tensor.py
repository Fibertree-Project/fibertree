# cython: language_level=3
# cython: profile=True
"""Tensor

A class used to implement the a tensor based on the **fibertree**
abstraction for representing tensors.

"""
import logging

import copy
import pickle
import yaml
from copy import deepcopy

from .rank    import Rank
from .fiber   import Fiber
from .payload import Payload

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.core.tensor')


class Tensor:
    """Tensor Class

    The Tensor class is a foundational class in this system and is
    used to model a tensor using a largely format-agnostic
    representation of the tensor. More specifically, this class uses a
    tree structure of fibers (called a **fibertree**) to represent the
    ranks of a tensor. More details on this representation can be
    found in sections 8.2 and 8.3 of the book "Efficient Processing of
    Deep Neural Networks" [1].

    Attributes
    ----------

    The principal attributes of a tensor are:

    - **name**: A name of the tensor.

    - **color**: A color to use to represent the tensor when it is
        drawn.

    - **root**: The root of a tensor is a reference to the top fiber
        of the fibertree comprising the structure of the tensor. The
        fibertree is implemented using the `Fiber` class.

    - **ranks**: The tensor contains a list of ranks, each of which
        contains all of the fibers at each level of the tensor's
        fibertree. The list contains instances of the `Rank` class,
        and each rank has a **rank id**, a **shape**, a **default
        value** and a **next_rank**. The **next rank** field is used
        to create a linked list of ranks in the tensor.

    - **rank ids**: The rank ids of a tensor is a list of the names
        (or rank ids) of each rank of the tensor.

    - **shape**: The shape of a tensor is a list of the shapes of the
        fibers in each rank.

    - **default value**: The default value of a tensor is the default
        value of payloads of the fibers in the leaf rank of the
        tensor.


    See the `Rank` and `Fiber` classes for more details on the
    attributes associated with those classes.


    Constructor
    -----------

    The main tensor constructor should be used to create an empty
    tensor, which has the given `rank_ids` and optionally the given
    `shape`, `name` and `color`.

    Parameters
    -----------

    rank_ids: list of strings
        List containing names of ranks.

    shape: list of integers, default=None
        A list of shapes of the ranks

    default: value, default=0
        A default value for elements in the leaf rank

    name: string, default=""
        A name for the tensor

    color: string, default="red"
        The color to paint values when displaying the tensor


    Notes
    -----

    For historical reasons, this constructor tries to get a Tensor
    from the specified "yamlfile", which is the first argument, so
    existing code uses it with the "yamlfile" keyword. That usage
    is deprecated in favor of using Tensor.fromYAMLfile().

    Bibliography
    ------------

    [1] "[Efficient Processing of Deep Neural Networks](http://www.morganclaypoolpublishers.com/catalog_Orig/product_info.php?products_id=1530)",
    Vivienne Sze, Yu-Hsin Chen, Tien-Ju Yang, and Joel S. Emer,
    Synthesis Lectures on Computer Architecture,
    June 2020, 15:2, 1-341.

    """

    def __init__(self,
                 yamlfile="",
                 rank_ids=None,
                 shape=None,
                 default=0,
                 name="",
                 color="red"):
        """__init__"""

        #
        # Set up logging
        #
        # self.logger = logging.getLogger('fibertree.core.tensor')

        self.yamlfile = yamlfile

        # TBD: Encourage use of Tensor.fromYAMLfile instead...

        if (yamlfile != ""):
            assert(rank_ids is None and shape is None)

            (rank_ids, root, shape, name) = self.parse(yamlfile)

            if shape is None:
                shape = root.estimateShape()

            self.setRankInfo(rank_ids, shape, default)
            self.setRoot(root)
            self.setName(name)
            self.setColor(color)
            self.setMutable(False)
            return

        #
        # Initialize an empty tensor with an empty root fiber
        #
        assert(rank_ids is not None)

        self.setRankInfo(rank_ids, shape, default)
        self.setName(name)
        self.setColor(color)
        self.setMutable(True)

        if rank_ids == []:
            # Create a rank zero tensor, i.e., just a payload

            self._root = Payload(0)
            return

        root_fiber = Fiber()
        self.setRoot(root_fiber)


    @classmethod
    def fromYAMLfile(cls, yamlfile):
        """Construct a tensor from a YAML file

        This constructor creates a Tensor from the specified
        `yamlfile`.

        Parameters
        -----------

        yamlfile: string
            Filename of file containing a YAML representation of a tensor


        Todo
        ----

        YAML file does not provide a non-zero default value

        """
        (rank_ids, root, shape, name) = Tensor.parse(yamlfile)

        if not isinstance(root, Fiber):
            t = Tensor(rank_ids=[], shape=shape, name=name)
            t.setMutable(False)
            t._root = Payload(root)
            return t

        return Tensor.fromFiber(rank_ids, root, shape=shape)


    @classmethod
    def fromUncompressed(cls,
                         rank_ids=None,
                         root=None,
                         shape=None,
                         name="",
                         color="red"):
        """Construct a Tensor from uncompressed nest of lists

        Parameters
        ----------

        rank_ids: list, default=["Rn", "Rn-1", ... "R0"]
            List containing names of ranks.

        root: list of lists
            A list of lists with an uncompressed represenation of the
            tensor, zero values are assumed empty.

        shape: list, default=(calculated from shape of "root")
            A list of shapes of the ranks

        name: string, default=""
            A name for the tensor

        color: string, default="red"
            The color to paint values when displaying the tensor

        """

        assert(root is not None)

        if not isinstance(root, list):
            # Handle a rank zero tensor
            t = Tensor(rank_ids=[], shape=[])
            t._root = Payload(root)
            return t

        assert(rank_ids is not None)

        fiber = Fiber.fromUncompressed(root)

        if shape is None:
            # TBD: Maybe this is not needed because fibers get a max_coord...
            shape = Tensor._calc_shape(root)

        return Tensor.fromFiber(rank_ids,
                                fiber,
                                shape=shape,
                                name=name,
                                color=color)


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
    def fromFiber(cls,
                  rank_ids=None,
                  fiber=None,
                  shape=None,
                  name="",
                  color="red"):
        """Construct a tensor from a fiber

        Parameters
        -----------

        rank_ids: list, default=["Rn", "Rn-1", ... "R0"]
            List containing names of ranks.

        fiber: Fiber
            A fiber to form the root of the new Tensor

        shape: list, default=(the shape of "fiber")
            A list of shapes of the ranks

        name: string, default=""
            A name for the tensor

        color: string, default="red"
            The color to paint values when displaying the tensor

        """

        assert(fiber is not None)

        #
        # If rank_ids is not given, synthesize something reasonable
        #
        if rank_ids is None:
            if shape is not None:
                maxrank = len(shape) - 1
            else:
                maxrank = fiber.getDepth() - 1

            rank_ids = [f"R{maxrank-i}" for i in range(maxrank + 1)]

        #
        # Create empty Tensor, which gets populated with a fiber below
        #
        tensor = cls(rank_ids=rank_ids,
                     shape=shape,
                     name=name,
                     color=color)

        tensor.setRoot(fiber)

        # setRoot may modify the shape, fix the shape if authoritative
        if shape:
            tensor.setShape(shape)

        tensor.setMutable(False)

        return tensor


    @classmethod
    def fromRandom(cls,
                   rank_ids=None,
                   shape=None,
                   density=None,
                   interval=10,
                   seed=None,
                   name="",
                   color="red"):
        """Create a random tensor

        Parameters
        ----------

        rank_ids: list
            The "rank ids" for the tensor

        shape: list
            The "shape" (i.e., size) of each level of the tree

        density: list
            The probability that an element of the fiber will not be
            *empty* for each level of the tree

        interval: integer
            The closed range [0:`interval`] of each value at the leaf
            level of the tree

        seed: a valid argument for `random.seed`
            A seed to pass to `random.seed`

        """

        f = Fiber.fromRandom(shape, density, interval, seed)

        return Tensor.fromFiber(rank_ids=rank_ids,
                                fiber=f,
                                shape=shape,
                                name=name,
                                color=color)



    @staticmethod
    def _shape2lists(shape):
        """ Return a nest of lists of "shape" filled with zeros"""

        if len(shape) > 1:
            subtree = Tensor._shape2lists(shape[1:])
        else:
            subtree = 0

        result = [subtree for _ in range(shape[0])]

        return result

#
# Accessor methods
#
    def setRankInfo(self, rank_ids, shape, default=0):
        """Initialize rank info

        This method creates and initializes the list of ranks in this
        tensor with the provided lists of `rank_ids` and `shape`. The
        fibers associated with the ranks get set separately with
        `Tensor.setRoot()`.

        Parameters
        ----------

        rank_ids: list of strings
            Names to assign to ranks

        shape: list of integers
            Shapes to assign to ranks

        default: value, default=0
            A value to use as the default for the leaf rank

        Returns
        -------
        Nothing

        """

        if shape is None:
            shape = [None] * len(rank_ids)

        #
        # Create a linked list of ranks
        #
        self.ranks = []
        last_rank = None

        #
        # Populate the list of ranks (in reverse) so the "next_rank" field
        # can be filled in
        #
        for id, dimension in reversed(list(zip(rank_ids, shape))):
            new_rank = Rank(id=id, shape=dimension, next_rank=last_rank)
            self.ranks.insert(0, new_rank)
            last_rank = new_rank

        #
        # If provided, set leaf rank with a non-zero default
        #
        if default != 0:
            self.ranks[-1].setDefault(default)


    def syncRankInfo(self, ranks):
        """.. deprecated::"""

        # TBD: Currently unused and untested, so probably broken

        self.ranks = []
        last_rank = None

        for rank in reversed(ranks):
            rank.set_next(last_rank)
            last_rank = rank


    def getRankIds(self):
        """Get the rank ids of the tensor

        Parameters
        ----------
        None

        Returns
        -------

        rank_ids: list of strings
            List of names of ranks

        """

        #
        # Get the rank id for each rank
        #
        return [r.getId() for r in self.ranks]


    def setRankIds(self, rank_ids):
        """Set the rank ids of the tensor

        Parameters
        ----------
        rank_ids: list of strings
            List of names of ranks

        Returns
        -------
        self: tensor
            Returns `self` so method can be used in a chain

        """
        rank = self._root.getOwner()

        for rank_id in rank_ids:
            rank.setId(rank_id)
            rank = rank.getNextRank()

        return self

    def setShape(self, shape):
        """Set the shape of the tensor

        Parameters
        ----------

        shape: list
            List of rank shapes

        Returns
        -------

        None
        """
        assert len(shape) == len(self.ranks)

        for rank, rank_shape in zip(self.ranks, shape):
            rank.getAttrs().setShape(rank_shape)


    def getShape(self, rank_ids=[], authoritative=False):
        """Get the shape of the tensor.

        Get the shape(s) of the ranks that comprise the tensor. If a
        single `rank_id` is provided the shape of that rank is
        returned as a scalar. If a list of `rank_ids` is provided a
        list will be returned with the shapes of the requested
        ranks. Or if no `rank_id` is provided a list of the shapes of
        **all** the ranks of the tensor is returned.

        Since the shape may sometimes be estimated, this method gives
        the option of insisting that the returned shape be known
        authoritatively (if not the method returns None).

        Parameters
        ----------
        rank_ids: list of strings or a string, default=[]
            A list of rankids or a single rankid

        authoritative: Boolean, default=False
            Control whether to return an estimated (non-authoritative) shape

        Returns
        -------
        shape: integer, list of integers or None
            The shape or a list of the shapes of the requested rank(s).


        Notes
        -----

        A rank zero tensor will return an empty list

        """
        #
        # Convert rankids into a list, but remember if it was list originally
        #
        if isinstance(rank_ids, str):
            return_scalar = True
            rank_ids = [rank_ids]
        else:
            return_scalar = False

        #
        # Rank-0 tensors have no shape
        #
        if len(self.ranks) == 0:
            return []

        #
        # Return shapes for desired rank_ids
        #
        all_rank_ids = self.getRankIds()
        all_shapes = self.ranks[0].getShape(all_ranks=True,
                                            authoritative=authoritative)

        #
        # Maybe there is no authoritative shape
        #
        if all_shapes is None:
            return None

        if len(rank_ids) == 0:
            requested_rank_ids = all_rank_ids
        else:
            requested_rank_ids = rank_ids

        #
        # Get shape for each requested rank
        #
        shapes = []

        for rank_id in requested_rank_ids:
            rank_num = all_rank_ids.index(rank_id)
            shapes.append(all_shapes[rank_num])

        #
        # If exactly one shape was requested return a scalar
        #
        if return_scalar:
            return shapes[0]

        #
        # Return list of requested ranks
        #
        return shapes


    def getDepth(self):
        """Get the depth of the tensor

        Get the depth, i.e., number of dimensions, of the tensor.

        Parameters
        ----------
        None

        Returns
        -------
        depth: integer
            Number of dimensions in the tensor

        Raises
        ------
        None

        """

        return len(self.ranks)


    def setRoot(self, root):
        """Set the root fiber of tensor

        The method will (re-)populate the ranks of the tensor
        (`self.ranks`) with the fibertree contents of the provided
        `root` fiber.

        Parameters
        ----------

        root: Fiber
            The fiber that will be root of the tensor


        Returns
        --------
        Nothing

        """
        #
        # Note: rank 0 tensors are not allowed in this path
        #
        assert isinstance(root, Fiber)

        #
        # Copy fiber if it already belongs to another tensor
        #
        # Note: shapes and owners will be overwritten in _addFiber()
        #
        if root.getOwner() is not None:
            root = deepcopy(root)

        self._root = root

        #
        # Clear out existing rank information
        #
        for r in self.ranks:
            r.clearFibers()

        self._addFiber(root)

    def _addFiber(self, fiber, level=0):
        """Recursively fill in ranks from "fiber"."""
        shape = fiber.getShape(authoritative=True, all_ranks=False)
        if shape:
            attrs = self.ranks[level].getAttrs()
            if attrs.getShape():
                attrs.setShape(max(attrs.getShape(), shape))
            else:
                attrs.setShape(shape)
                attrs.setEstimatedShape(False)

        self.ranks[level].append(fiber)

        # Note: The code below handles the (probably abandoned)
        #       transistion from raw fibers as payloads to fibers in
        #       Payload

        for p in fiber.getPayloads():
            if Payload.contains(p, Fiber):
                self._addFiber(Payload.get(p), level + 1)

    def getRoot(self):
        """Get the root fiber of the tensor

        Parameters
        ----------
        None

        Returns
        -------

        root: Fiber
            The fibertree at the root of the tensor

        """

        root = self._root

        #
        # Either we have a 0-D tensor or the root is a Fiber
        #
        # TBD: This is broken if Fibers are wrapped in a Payload
        #
        assert (isinstance(root, Payload) or
                root == self.ranks[0].getFibers()[0])

        return root


    def root(self):
        """.. deprecated::"""

        Tensor._deprecated("Tensor.root() is deprecated, use getRoot()")

        return self.getRoot()


    def setName(self, name):
        """Set name for the tensor

        Parameters
        ----------

        name: string
            Name to use for tensor

        Returns
        -------
        self: Tensor
            So method can be used in a chain

        Raises
        ------
        None

        """

        self._name = name
        return self


    def getName(self):
        """Get name of tensor

        Parameters
        ----------
        None

        Returns
        -------
        name: string
            Name of tensor

        Raises
        ------
        None

        """

        return self._name


    def setColor(self, color):
        """Set color for elements of tensor

        Parameters
        ----------
        color: string
             Color to use for scalar values in tensor

        Returns
        -------
        self: Tensor
           So method can be used in a chain

        Raises
        ------
        None

        """

        self._color = color
        return self


    def getColor(self):
        """Get color for elements of tensor

        Parameters
        ----------
        None

        Returns
        -------
        color: string
            Color being used for scalar values in the tensor

        Raises
        ------
        None

        """

        return self._color


    def setDefault(self, value):
        """Set the default value for the leaf payloads of the tensor

        Parameters
        ----------
        value: value
            A value to use for leaf payload values in tensor

        Returns
        -------
        self: Tensor
            So method can be used in a chain

        Raises
        ------
        None

        Notes
        -----

        The **default** value will be **boxed** by
        `Rank.getDefault()`.

        """

        assert value != Fiber, "Leaf payloads cannot be a Fiber"

        #
        # Set default for leaf rank
        #
        self.ranks[-1].setDefault(value)

        return self


    def getDefault(self):
        """Get the default payload for leaf ranks

        Parameters
        ----------
        None

        Returns
        -------
        value: value
            A copy of the default payload of the leaf rank

        Raises
        ------
        None

        Notes
        -----

        A `deepcopy()` of the **default** value will have been
        performed in `Rank.getDefault()` so the value returned will be
        unique.

        """

        return self.ranks[-1].getDefault()


    def setMutable(self, value):
        """Set the mutabilility hint

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
        self: Tensor
            So method can be used in a chain

        Raises
        ------
        None

        """

        self._mutable = value

        return self


    def isMutable(self):
        """Returns mutability attribute

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

    def setFormat(self, rank_id, fmt):
        """Set the format for the given rank

        Sets the format of the rank specified by the `rank_id` to the given
        value

        Parameters
        ----------

        rank_id: string
            The ID of the rank whose format to modify

        fmt: string
            The format of the rank; "C" = compressed, "U" = uncompressed

        Returns
        -------
        None


        Raises
        ------

        ValueError
            rank_id is not a named rank in the tensor

        AssertionError
            Illegal format

        """

        rank_ids = self.getRankIds()
        self.ranks[rank_ids.index(rank_id)].setFormat(fmt)

    def getFormat(self, rank_id):
        """Get the format of the given rank

        Gets the format of the rank specified by the `rank_id`

        Parameters
        ----------

        rank_id: string
            The ID of the rank whose format to modify

        Returns
        -------

        fmt: string
            The format of the rank; "C" = compressed, "U" = uncompressed


        Raises
        ------

        ValueError
            rank_id is not a named rank in the tensor
        """

        rank_ids = self.getRankIds()
        return self.ranks[rank_ids.index(rank_id)].getFormat()


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

        return rankid_match and fiber_match


#
# Tensor equivalent of Fiber methods where operating on the
# root fiber is the logical activity
#
    def getPayload(self, *args, **kwargs):
        """Get payload at a point

        Tensor-level version of method that operates on the root
        fiber of the tensor. See `Fiber.getPayload()` for details.

        """

        root = self.getRoot()

        if isinstance(root, Payload):
            # Handle rank-0 tensor
            return root

        return root.getPayload(*args, **kwargs)


    def getPayloadRef(self, *args, **kwargs):
        """Get a reference to a payloat at at point

        Tensor-level version of method that operates on the root
        fiber of the tensor. See `Fiber.getPayloadRef()` for details.

        """

        root = self.getRoot()

        if isinstance(root, Payload):
            # Handle rank-0 tensor
            return root

        return root.getPayloadRef(*args, **kwargs)


    def countValues(self):
        """Get count on non-empty values in tensor

        Count of non-empty payload values in the leaf rank of tensor

        Tensor-level version of method that operates on the root
        fiber of the tensor. See `Fiber.countValues()` for details.

        """
        return self.getRoot().countValues()


    def __iter__(self):
        """__iter__"""

        return self.getRoot().__iter__()


    def __reversed__(self):
        """Return reversed fiber"""

        return self.getRoot().__reversed__()


    def __getitem__(self, keys):
        """__getitem__

        Tensor-level version of method that operates on the root
        fiber of the tensor. See `Fiber.__getitem__()` for details.

        """

        return self.getRoot().__getitem__(keys)

    def __setitem__(self, key, newvalue):
        """__setitem__

        Tensor-level version of method that operates on the root
        fiber of the tensor. See `Fiber.__setitem__()` for details.

        """

        self.getRoot().__setitem__(key, newvalue)


    def updateCoords(self, func, depth=0, **kwargs):
        """Update coordinates of root fiber

        Tensor-level version of method that operates on the root
        fiber of the tensor. See `Fiber.updateCoords()` for details.

        """

        new_tensor = copy.deepcopy(self)

        new_tensor.getRoot().updateCoords(func, depth=depth, **kwargs)

        return new_tensor


    def updatePayloads(self, func, depth=0, **kwargs):
        """Update payloads of root fiber

        Tensor-level version of method that operates on the root
        fiber of the tensor. See `Fiber.updatePayloads()` for details.

        """

        new_tensor = copy.deepcopy(self)

        new_tensor.getRoot().updatePayloads(func, depth=depth, **kwargs)

        return new_tensor

#
# Split methods
#
# Note: all these methods return a new tensor
#
    def __truediv__(self, arg):
        """Split root fiber in coordinate space

        Tensor-level version of method that operates on the root
        fiber of the tensor. See `Fiber.__truediv()` for details.

        """

        return self._splitGeneric(Fiber.__truediv__, arg)

    def __floordiv__(self, arg):
        """Split root fiber in position space

        Tensor-level version of method that operates on the root
        fiber of the tensor. See `Fiber.__floordiv()` for details.

        """

        return self._splitGeneric(Fiber.__floordiv__, arg)


    def splitUniform(self, *args, **kwargs):
        """Split tensor's fibertree uniformly in coordinate space

        Tensor-level version of method that operates on the tensor's fibertree
        at depth `depth`. See `Fiber.splitUniform()` for more details.

        Parameters
        ----------

        See `Fiber.splitUniform()` for arguments.

        Returns
        -------

        split_tensor: Tensor
             A new split tensor

        """

        return self._splitGeneric(Fiber.splitUniform,
                                  *args,
                                  **kwargs)

    def splitNonUniform(self, *args, **kwargs):
        """Split tensor's fibertree non-uniformly in coordinate space

        Tensor-level version of method that operates on the tensor's fibertree
        at depth `depth`. See `Fiber.splitNonUniform()` for more details.

        Parameters
        ----------

        See `Fiber.splitNonUniform()` for arguments.

        Returns
        -------

        split_tensor: Tensor
             A new split tensor

        """

        return self._splitGeneric(Fiber.splitNonUniform,
                                  *args,
                                  **kwargs)


    def splitEqual(self, *args, **kwargs):
        """Split tensor's fibertree equally in position space

        Tensor-level version of method that operates on the tensor's fibertree
        at depth `depth`. See `Fiber.splitEqual()` for more details.

        Parameters
        ----------

        See `Fiber.splitEqual()` for arguments.

        Returns
        -------

        split_tensor: Tensor
             A new split tensor

        """

        return self._splitGeneric(Fiber.splitEqual,
                                  *args,
                                  **kwargs)


    def splitUnEqual(self, *args, **kwargs):
        """Split tensor's fibertree unequally in postion space

        Tensor-level version of method that operates on the tensor's fibertree
        at depth `depth`. See `Fiber.splitUnEqual()` for more details.

        Parameters
        ----------

        See `Fiber.splitUnEqual()` for arguments.

        Returns
        -------

        split_tensor: Tensor
             A new split tensor

        """

        return self._splitGeneric(Fiber.splitUnEqual,
                                  *args,
                                  **kwargs)


    def _splitGeneric(self, func, *args, **kwargs):
        """ _splitGeneric... """

        rank_ids = copy.deepcopy(self.getRankIds())

        #
        # Determine depth
        #
        if "rankid" in kwargs:
            depth = rank_ids.index(kwargs["rankid"])
        elif "depth" in kwargs:
            depth = kwargs["depth"]
        else:
            depth = 0

        #
        # Create new list of rank ids
        #
        id = rank_ids[depth]
        rank_ids[depth] = f"{id}.1"
        rank_ids.insert(depth + 1, f"{id}.0")

        #
        # Create new shape list
        #
        shape = copy.deepcopy(self.getShape(authoritative=True))
        if shape:
            shape.insert(depth + 1, shape[depth])

        #
        # Create new root fiber
        #
        root = func(self.getRoot(), *args, **kwargs)

        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root, shape)
        tensor.setName(self.getName() + "+split")
        tensor.setColor(self.getColor())
        tensor.setMutable(self.isMutable())

        # Maintain the formats
        for rank_id in tensor.getRankIds():
            if rank_id == id + ".1":
                old_id = id
            elif rank_id == id + ".0":
                old_id = id
            else:
                old_id = rank_id
            tensor.setFormat(rank_id, self.getFormat(old_id))


        return tensor

#
# Swizzle and swap methods
#
    def swizzleRanks(self, rank_ids):
        """Swizzle the ranks of the tensor

        Re-arrange (swizzle) the ranks of the tensor so they match the
        given `rank_ids`. This is accompished via a series of rank swaps.

        Parameters
        ----------

        rank_ids: list of strings
            List of names of ranks in desired order (top to bottom)


        Returns
        -------

        swizzled_tensor: Tensor
            New tensor with ranks swizzed

        """
        # Ensure that these old and new rank_ids are permutations of each other
        old_rank_ids = self.getRankIds()
        assert sorted(old_rank_ids) == sorted(rank_ids)

        old_name = self.getName()
        copied = copy.deepcopy(self)

        if old_rank_ids == rank_ids:
            copied.setName(f"{old_name}+swizzled")
            return copied

        # Find the point after which the two lists are the same
        for i, (old, new) in \
                enumerate(zip(reversed(old_rank_ids), reversed(rank_ids))):
            if old != new:
                break
        swiz_len = len(rank_ids) - i

        # Figure out where each coordinate in the input needs to be moved
        guide = []
        for rank_id in rank_ids:
            guide.append(old_rank_ids.index(rank_id))

        coords = []
        payloads = {}
        frontier = [(copied.getRoot(), None, -1)]
        frontier_coords = [None] * swiz_len

        # Depth-first search through the fibertree and extract the coordinate
        # payload pairs
        while frontier:
            head, coord, depth = frontier.pop()
            if coord is not None:
                frontier_coords[depth] = coord

            # If this is the last point we need to swizzle, save the payload
            if depth == swiz_len - 1:
                new_c = tuple(frontier_coords[guide[i]] for i in range(swiz_len))

                coords.append(new_c)
                payloads[new_c] = head
                continue

            # Otherwise, add this fiber to the frontier
            for c, p in zip(head.coords, head.payloads):
                frontier.append((p, c, depth + 1))

        # Sort the coordinates
        coords.sort(reverse=True)


        # Add back all of the payloads
        root = Fiber()
        fibers = [root] + [None] * (swiz_len - 1)
        last_coord = (None,) * swiz_len
        while coords:
            coord = coords.pop()

            # Reuse the payloads we have gotten so far
            same = True
            for i, c in enumerate(coord[:-1]):
                same = same and c == last_coord[i]

                # Get a new payload if we are on a new tree
                if not same:
                    child = Fiber()
                    fibers[i].append(c, child)
                    fibers[i + 1] = child

            # Append the payload
            fibers[-1].append(coord[-1], payloads[coord])

            last_coord = coord

        # Build the new tensor
        kwargs = {"fiber": root, "rank_ids": rank_ids}
        old_shape = self.getShape(authoritative=True)
        if old_shape:
            new_shape = [old_shape[guide[i]] for i in range(swiz_len)] \
                + old_shape[swiz_len:]
            kwargs["shape"] = new_shape
        swizzled = Tensor.fromFiber(**kwargs)
        swizzled.setName(f"{old_name}+swizzled")

        return swizzled


    def swapRanks(self, depth=0):
        """Swap a pair of ranks in the  tensor's fibertree.

        Tensor-level version of method that operates on the tensor's fibertree
        at depth `depth`. See `Fiber.swapRanks()` for more details.

        Parameters
        ----------

        depth: integer, default=0
            Level of fibertree to split

        See `Fiber.swapRanks()` for other arguments.

        Returns
        -------

        swapped_tensor: Tensor
             A new tensor with two ranks swapped

        """

        #
        # Create new list of rank ids
        #
        rank_ids = copy.deepcopy(self.getRankIds())
        id = rank_ids[depth]
        rank_ids[depth] = rank_ids[depth + 1]
        rank_ids[depth + 1] = id

        #
        # Create new shape list
        #
        # TBD: Create shape
        #
        shape = None

        # Only call Fiber.swapRanks if there are actually payloads to swap
        if not all(fiber.isEmpty() for fiber in self.ranks[depth].fibers):
            root = self._modifyRoot(Fiber.swapRanks,
                                    Fiber.swapRanksBelow,
                                    depth=depth)
        else:
            root = copy.deepcopy(self.getRoot())

        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root, shape)
        tensor.setName(self.getName() + "+swapped")
        tensor.setColor(self.getColor())
        tensor.setMutable(self.isMutable())

        # Maintain the formats
        for rank_id in tensor.getRankIds():
            tensor.setFormat(rank_id, self.getFormat(rank_id))

        return tensor


    def flattenRanks(self, depth=0, levels=1, coord_style="tuple"):
        """Flatten ranks in the  tensor's fibertree.

        Tensor-level version of method that operates on the tensor's fibertree
        at depth `depth`. See `Fiber.flattenRanks()` for more details.

        Parameters
        ----------

        See `Fiber.flattenRanks()` for arguments.

        Returns
        -------

        flattened_tensor: Tensor
             A new tensor with some ranks flattened

        """
        rank_ids, shape = self._flattenRankIdsShape(depth=depth, levels=levels, coord_style=coord_style)

        root = self.getRoot().flattenRanks(depth=depth, levels=levels, style=coord_style)

        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root, shape)
        tensor.setName(self.getName() + "+flattened")
        tensor.setColor(self.getColor())
        tensor.setMutable(self.isMutable())

        # Maintain the formats for unflattened rank_ids
        # Compress everything else
        for rank_id in tensor.getRankIds():
            if rank_id in self.getRankIds():
                tensor.setFormat(rank_id, self.getFormat(rank_id))
            else:
                tensor.setFormat(rank_id, "C")

        return tensor

    def mergeRanks(self, depth=0, levels=1, coord_style="tuple", merge_fn=None):
        """Merge ranks in the  tensor's fibertree.

        Tensor-level version of method that operates on the tensor's fibertree
        at depth `depth`. See `Fiber.mergeRanks()` for more details.

        Parameters
        ----------

        See `Fiber.mergeRanks()` for arguments.

        Returns
        -------

        merged_tensor: Tensor
             A new tensor with some ranks merged

        """
        rank_ids, shape = self._flattenRankIdsShape(depth=depth, levels=levels, coord_style=coord_style)

        root = self.getRoot().mergeRanks(depth=depth, levels=levels, style=coord_style, merge_fn=merge_fn)

        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root, shape)
        tensor.setName(self.getName() + "+merged")
        tensor.setColor(self.getColor())
        tensor.setMutable(self.isMutable())

        # Maintain the formats for unmerged rank_ids
        # Compress everything else
        for rank_id in tensor.getRankIds():
            if rank_id in self.getRankIds():
                tensor.setFormat(rank_id, self.getFormat(rank_id))
            else:
                tensor.setFormat(rank_id, "C")

        return tensor

    def _flattenRankIdsShape(self, depth, levels, coord_style):
        """Compute the rank ids and shape after a flattening"""
        #
        # Create new list of rank ids
        #
        # Note: we need to handle the case where existing ranks are lists
        #
        rank_ids = copy.deepcopy(self.getRankIds())

        cur_rankid = rank_ids[depth]
        if not isinstance(cur_rankid, list):
            rank_ids[depth] = []
            rank_ids[depth].append(cur_rankid)

        for d in range(levels):
            next_rankid = rank_ids[depth + 1]

            if isinstance(next_rankid, list):
                rank_ids[depth] = cur_rankid + next_rankid
            else:
                rank_ids[depth].append(next_rankid)

            del rank_ids[depth + 1]

        #
        # Create new shape list
        #
        old_shape = self.getShape(authoritative=True)
        new_shape = None
        if old_shape:
            new_shape = []
            curr_shape = ()
            for i, shape in enumerate(old_shape):
                if i < depth:
                    new_shape.append(shape)

                elif i > depth + levels:
                    new_shape.append(shape)

                # Shape: [S0, S1, ... SN] -> (S0, S1, ... SN)
                elif coord_style == "tuple":
                    curr_shape += (shape,)
                    if i == depth + levels:
                        new_shape.append(curr_shape)

                # Shape: [S0, S1, ... SN] -> (S0, (S1, (... (SN-1, SN))))
                elif coord_style == "pair":
                    curr_shape += (shape,)

                    if i == depth + levels:
                        # Nest the shape:
                        nested = curr_shape[-2:]
                        for val in reversed(curr_shape[:-2]):
                            nested = (val, nested)
                        new_shape.append(nested)

                # Shape: [S0, S1, ... SN] -> SN
                elif coord_style == "absolute":
                    if i == depth + levels:
                       new_shape.append(shape)

                # Shape: [S0, S1, ... SN] -> S0
                elif coord_style == "relative":
                    if i == depth:
                        new_shape.append(shape)

                # Shape: [S0, S1, ... SN] -> S0 * S1 * ... * SN
                elif coord_style == "linear":
                    if i == depth:
                        new_shape.append(shape)
                    else:
                        new_shape[-1] *= shape

                else:
                    assert False, \
                        f"Supported coordinate styles are: tuple, pair, absolute, relative, and linear. Got: {coord_style}"

        return rank_ids, new_shape

    def unflattenRanks(self, depth=0, levels=1):
        """Unflatten ranks in the  tensor's fibertree.

        Tensor-level version of method that operates on the tensor's fibertree
        at depth `depth`. See `Fiber.unflattenRanks()` for more details.

        Parameters
        ----------

        depth: integer, default=0
            Level of fibertree to split

        See `Fiber.unflattenRanks()` for other arguments.

        Returns
        -------

        unflattened_tensor: Tensor
             A new tensor with some ranks unflattened

        """

        #
        # Create new list of rank ids
        #
        rank_ids = copy.deepcopy(self.getRankIds())

        for d in range(levels):
            id = rank_ids[depth + d]
            rank_ids[depth + d] = id[0]
            if len(id) == 2:
                rank_ids.insert(depth + d + 1, id[1])
            else:
                rank_ids.insert(depth + d + 1, id[1:])

        #
        # Create new shape list
        #
        # TBD: Create shape
        #
        shape = None

        # Only call Fiber.unflattenRanks if there are actually ranks to unflatten
        if not all(fiber.isEmpty() for fiber in self.ranks[depth].fibers):
            root = self._modifyRoot(Fiber.unflattenRanks,
                                    Fiber.unflattenRanksBelow,
                                    depth=depth,
                                    levels=levels)
        else:
            root = Fiber()

        #
        # Create Tensor from rank_ids and root fiber
        #
        tensor = Tensor.fromFiber(rank_ids, root, shape)
        tensor.setName(self.getName() + "+unflattened")
        tensor.setColor(self.getColor())
        tensor.setMutable(self.isMutable())

        # Maintain the formats for all untouched rank_ids
        # Compress everything else
        for rank_id in tensor.getRankIds():
            if rank_id in self.getRankIds():
                tensor.setFormat(rank_id, self.getFormat(rank_id))
            else:
                tensor.setFormat(rank_id, "C")


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
            funcBelow(root, depth=depth - 1, **kwargs)

        #
        # Create Tensor from rank_ids and root fiber
        #
        return root

    def clearStats(self):
        """clearStats
        NDN: add comment
        """
        for rank in self.ranks:
            for fiber in rank.getFibers():
                fiber.clearStats()


#
# String methods
#
    def print(self, title=None):
        """print"""

        if title is not None:
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
                        'root': [root_dict]}}

        with open(filename, 'w') as file:
            yaml.dump(tensor_dict, file)

#
# Copy operation
#
    def __deepcopy__(self, memo):
        """__deepcopy__

        Note: to ensure maintainability, we want to automatically copy
        everything. We use pickling because it is much more performant
        than the default deepcopy
        """
        return pickle.loads(pickle.dumps(self))

#
# Utility methods
#

    @staticmethod
    def _deprecated(message):
        import warnings

        warnings.warn(message, FutureWarning, stacklevel=3)

#
# Pdoc stuff
#
__pdoc__ = {'Tensor.parse':         False,
            'Tensor.__setitem__':   True,
            'Tensor.__getitem__':   True,
            'Tensor.__ilshift__':   True,
            'Tensor.__truediv__':   True,
            'Tensor.__floordiv__':  True,
           }
