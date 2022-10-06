# cython: language_level=3
# cython: profile=True
"""Rank

A class used to implement a rank (or dimension) of a tensor.

"""

import logging
import pickle

from .fiber import Fiber
from .payload import Payload
from .rank_attrs import RankAttrs

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.core.rank')

class Rank:
    """Class representing a "rank" (or dimension) of a tensor.

    An instance of this class holds a list of all the fibers at a
    rank, common attributes of the fibers in the rank, and a pointer
    to the next rank.

    A `Tensor` contains a list of the ranks it is comprised of, and
    the "next rank" pointer is used to create a linked list of those
    ranks..

    Attributes
    ----------

    rank_id: string
        The name of the rank

    estimated_shape: Boolean
        Is the shape estimated or was it provided explicitly

    shape: integer
        The shape of the fibers in the rank

    fibers: list of Fibers
        A list of the fibers in the rank

    fmt: string
        What format is this rank in (either "C" (compressed) or
        "U" (uncompressed))

    Constructor
    -----------

    The `Rank` constructor creates an empty rank.

    Parameters
    -----------

    id: string
        The name (rank_id) of the rank

    shape: integer, default=None
        The shape of the fibers in the rank

    next_rank: Rank, default=None
        The next rank in the tensor

    fmt: Boolean, default="C"
        What format is this rank in (either "C" (compressed) or
        "U" (uncompressed))


    Notes
    -----

    The fibers in a rank are NOT provided as part of the contructor
    but are added incrementally using `Rank.append()`.

    """


    def __init__(self, id, shape=None, next_rank=None, fmt="C"):
        """__init__"""

        #
        # Set up logging
        #
        # self.logger = logging.getLogger('fibertree.core.rank')
        self._attrs = RankAttrs(id, shape, fmt)

        self.setNextRank(next_rank)

        self.fibers = []

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

        return self._attrs.getId()


    def getRankIds(self, all_ranks=True):
        """Get a list of ranks ids.

        Get a list of rank ids starting at this rank and optionally
        including the rank ids all succeeding (lower level) ranks.

        Parameters
        ----------
        None

        Returns
        -------
        rank_id: list of strings
            List of rank ids

        Todo
        ----

        There is an asymmetry between this method and
        `Rank.getShape()` because it always returns a list,
        irrespective of the value of `all_ranks`.

        """

        rankids = [self.getId()]

        if all_ranks and self.next_rank is not None:
            rankids.extend(self.next_rank.getRankIds(all_ranks=True))

        return rankids

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
        self: rank
            Returns `self` so method can be used in a chain

        """

        self._attrs.setId(rank_id)
        return self


    def getName(self):
        """.. deprecated::"""

        Rank._deprecated("Use of Rank.getName() is deprecated - use Rank.getId()")

        return self._attrs.getId()


    def getShape(self, all_ranks=True, authoritative=False):
        """Return shape of rank.

        Since the shape may sometimes be estimated, this method gives
        the option of insisting that the returned shape be known
        authoritatively (if not the method returns None).

        Parameters
        ----------
        all_ranks: Boolean, default=True
            Control whether to return shape of all ranks or just this one

        authoritative: Boolean, default=False
            Control whether to return an estimated (non-authoritative) shape

        Returns
        -------
        shape: integer, list of integers or None
            The shape of this rank or this rank and all succeeding ranks

        Todo
        ----

        There is an asymmetry between this method and
        `Rank.getRankIds()` because it sometimes returns a list and
        sometimes a scalar depending on the value of `all_ranks`.

        """

        if all_ranks == False:
            #
            # Handle case where user just wants shape of this rank
            #
            if authoritative and self._attrs.getEstimatedShape():
                #
                # We do not know the shape authoritatively
                #
                return None

            if self._attrs.getShape() == 0:
                #
                # We do not actually know the shape, but we can estimate it
                #
                return max([f.estimateShape(all_ranks=False) for f in self.fibers])

            return self._attrs.getShape()

        #
        # Get shape of all ranks
        #
        if authoritative and self._attrs.getEstimatedShape():
            #
            # This will cause the final return to be None
            #
            return None

        if self._attrs.getShape() is not None:
            shape = [self._attrs.getShape()]
        elif len(self.fibers) == 0:
            shape = [0]
        else:
            shape = [max([f.estimateShape(all_ranks=False) for f in self.fibers])]

        if self.next_rank is not None:
            rest_of_shape = self.next_rank.getShape(all_ranks=True, authoritative=authoritative)
            if rest_of_shape is None:
                return None

            shape.extend(rest_of_shape)

        #
        # If we didn't have a shape for any rank, assume we don't know anything
        #
        #if any([s == 0 for s in shape]):
        #    return None

        return shape


    def getFibers(self):
        """Return list of fibers in the rank.

        Parameters
        ----------
        None

        Returns
        -------
        fibers: list of Fibers
            All the fibers in this rank

        """

        return self.fibers

    def clearFibers(self):
        """Empty rank of all fibers.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        self.fibers = []

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
        self: Rank
           So method can be used in a chain

        Raises
        ------
        None

        Notes
        -----

        We make sure that the value saved is **boxed**.

        """

        self._attrs.setDefault(value)

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

        """

        return self._attrs.getDefault()

    def setFormat(self, fmt):
        """Set the format for this rank

        Set the format to the given value

        Parameters
        ----------

        fmt: string
            The format of the rank; "C" = compressed, "U" = uncompressed

        Returns
        -------

        self: Rank
           So method can be used in a chain

        Raises
        ------

        AssertionError
            Illegal format

        """

        self._attrs.setFormat(fmt)

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
        return self._attrs.getFormat()

    def getAttrs(self):
        """
        Get the rank attributes

        Parameters
        ----------
        None

        Returns
        -------
        attrs: RankAttrs
            Rank attributes
        """
        return self._attrs


#
# Fundamental methods
#
    def append(self, fiber):
        """Append the provided fiber into a rank

        Parameters
        ----------
        fiber: Fiber
            A fiber to add to the rank

        Returns
        -------
        Nothing


        Notes
        -----

        If the **shape** of the rank is being estimated, this method
        might update the rank's shape.

        TODO
        ----

        Maybe should rename to appendFiber()

        """

        #
        # Get the raw fiber (if it was wrapped in a payload)
        #
        fiber = Payload.get(fiber)

        if self._attrs.getEstimatedShape():
            #
            # Get shape from fiber and see it is larger that current shape
            # making sure we don't get info from a prior owning rank
            #
            # TBD: If the fiber really has a definitive shape then
            # change estimated_shape to True
            #
            fiber.setOwner(None)

            old = self._attrs.getShape()
            new = fiber.getShape(all_ranks=False)
            if old is None and new != 0:
                self._attrs.setShape(new)
            elif new != 0:
                self._attrs.setShape(max(old, new))

        #
        # Set this rank as owner of the fiber
        #
        fiber.setOwner(self)

        #
        # Check default value for new coordinates in the fiber
        #
        if self.next_rank is None:
            assert self.getDefault() != Fiber, \
                "Leaf rank default should not be Fiber"
        else:
            assert self.getDefault() == Fiber, \
                "Non-leaf rank default should be Fiber"

        #
        # Add fiber to list of fibers of rank
        #
        self.fibers.append(fiber)

    def pop(self):
        """
        Remove the most-recently added fiber

        Parameters
        ----------

        None

        Returns
        _______

        fiber: Fiber
            The popped Fiber
        """
        fiber = self.fibers.pop()
        fiber.setOwner(None)
        return fiber

#
# Linked list methods
#
    def setNextRank(self, next_rank):
        """Set the next rank

        Record a reference to the next rank. If that rank exists then the
        default payload of fibers in this rank must be a fiber,
        otherwise set the default payload to zero.

        Parameters
        ----------
        next_rank: None

        Returns
        -------
        Nothing


        Todo
        ----
            The default payload probably shouldn't be zero.

        """

        self.next_rank = next_rank

        if next_rank is None:
            self.setDefault(0)
        else:
            self.setDefault(Fiber)


    def getNextRank(self):
        """Get the next rank

        Parameters
        ----------
        None

        Returns
        -------
        next_rank: Rank
            The next rank

        """

        return self.next_rank

#
# String methods
#

    def __str__(self, indent=0):
        """__str__"""

        string = indent * ' '
        string += f"Rank: {self._attrs.getId()} "

        next_indent = len(string)

        separator = ",\n" + " " * next_indent
        fibers = [x.__str__(indent=next_indent, cutoff=1000, newline=True) for x in self.fibers]
        string += separator.join(fibers)

        return string

    def __repr__(self):
        """__repr__"""

        string = "R(%s)/[" % self._attrs.getId()
        string += ", ".join([x.__repr__() for x in self.fibers])
        string += "]"
        return string

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
# Utility functions
#

    @staticmethod
    def _deprecated(message):
        import warnings

        warnings.warn(message, FutureWarning, stacklevel=3)


