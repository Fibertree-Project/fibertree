#cython: language_level=3
"""RankAttrs

A class used to store all attributes of a Rank

"""

from copy import deepcopy
import pickle

from .payload import Payload

class RankAttrs:
    """Class that represents the attributes of a rank of fibers

    Attributes
    ----------

    rank_id: string
        The name of the rank

    estimated_shape: Boolean
        Is the shape estimated or was it provided explicitly

    shape: integer
        The shape of the fibers in the rank

    fmt: string
        What format is this rank in (either "C" (compressed) or
        "U" (uncompressed))

    collecting: bool
        Whether we are collecting metrics for this rank

    Constructor
    -----------

    The `RankAttrs` constructor creates an empty rank attributes.

    Parameters
    -----------

    id: string
        The name (rank_id) of the rank

    shape: integer, default=None
        The shape of the fibers in the rank

    fmt: Boolean, default="C"
        What format is this rank in (either "C" (compressed) or
        "U" (uncompressed))

    """

    def __init__(self, rank_id="Unknown", shape=None, fmt="C"):
        """__init__"""

        self.setId(rank_id)

        if shape is None:
            self._estimated_shape = True
            self._shape = None
        else:
            self._estimated_shape = False
            self._shape = shape

        self.setFormat(fmt)

        self._default_is_set = False
        self._default = None

#
#
# Accessor methods
#

    def getId(self):
        """Return id of rank.

        Get the rank id of this rank, i.e., the name of this
        rank/dimension.

        Parameters
        ----------
        None

        Returns
        -------
        rank_id: string
            Rank id of this rank

        """

        return self._id


    def setId(self, rank_id):
        """Set id of rank.

        Set the rank id of this rank, i.e., the name of this
        rank/dimension.

        Parameters
        ----------
        rank_id: string
            Rank id of this rank


        Returns
        -------
        self: RankAttrs
            Returns `self` so method can be used in a chain

        """

        self._id = rank_id
        return self

    #
    # Default payload methods
    #
    def setDefault(self, value):
        """Set the default payload value for fibers in this rank.

        Parameters
        ----------
        value: value
            A value to use as the payload value for fibers in this rank

        Returns
        -------
        self: RankAttrs
           So method can be used in a chain

        Raises
        ------
        None

        Notes
        -----

        We make sure that the value saved is **boxed**.

        """

        self._default_is_set = True
        self._default = Payload.maybe_box(value)

        return self


    def getDefault(self):
        """Get the default payload for fibers in this rank

        Parameters
        ----------
        None

        Returns
        -------
        value: value
            A copy of the (boxed) default payload of fibers in this
            rank

        Raises
        ------
        AssertionError
            Raised if default is not set

        Notes
        -----

        We `deepcopy()` the return value so that everyone has their
        own unique **boxed** value

        """

        assert self._default_is_set

        #
        # Return a copy of the default
        #
        value = Payload.get(self._default)

        if isinstance(value, int):
            return Payload(value)

        return deepcopy(self._default)

    def setFormat(self, fmt):
        """Set the format for this rank

        Set the format to the given value

        Parameters
        ----------

        fmt: string
            The format of the rank; "C" = compressed, "U" = uncompressed

        Returns
        -------

        self: RankAttrs
           So method can be used in a chain

        Raises
        ------

        AssertionError
            Illegal format

        """
        assert fmt == "C" or fmt == "U"
        self._fmt = fmt

        return self

    def getFormat(self):
        """Get the format of this rank

        Parameters
        ----------
        None

        Returns
        -------

        fmt: string
            The format of the rank; "C" = compressed, "U" = uncompressed


        Raises
        ------
        None

        """
        return self._fmt

    def setShape(self, shape):
        """Set the shape for this rank

        Parameters
        ----------

        shape: Optional[int]
            Shape of the rank

        Returns
        -------

        self: Optional[RankAttrs]
           So method can be used in a chain

        Raises
        ------

        AssertionError
            shape not an int

        """
        self._shape = shape
        return self

    def getShape(self):
        """Get the shape for this rank

        Parameters
        ----------
        None

        Returns
        -------

        shape: int
            Shape of the rank

        Raises
        ------

        None

        """
        return self._shape

    def setEstimatedShape(self, estimated_shape):
        """Get whether the shape was estimated or specified

        Parameters
        ----------
        estimated_shape: bool
            True if the shape was estimated

        Returns
        -------

        None

        Raises
        ------
        None
        """
        self._estimated_shape = estimated_shape


    def getEstimatedShape(self):
        """Get whether the shape was estimated or specified

        Parameters
        ----------
        None

        Returns
        -------

        estimated_shape: bool
            True if the shape was estimated

        Raises
        ------
        None
        """
        return self._estimated_shape

    def __key(self):
        """__key"""
        return (self._id, self._estimated_shape, self._shape, self._fmt, self._default)

    def __eq__(self, other):
        """__eq__"""
        if not isinstance(other, type(self)):
            return False

        return self.__key() == other.__key()

    def __repr__(self):
        """__repr__"""
        return "(RankAttrs, " + ", ".join(repr(val) for val in self.__key()) + ")"
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
