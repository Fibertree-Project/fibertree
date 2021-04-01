from copy import deepcopy

from .fiber import Fiber
from .payload import Payload

""" Rank """


class Rank:
    """Rank class

    Class representing a "rank" of a tensor. It holds all the fibers
    at a rank, information about them and a pointer to the next rank.
    """

    def __init__(self, id, shape=None, next_rank=None):
        """
        Create a new rank.

         Parameters
        -----------

        id: String
        The name of the rank

        shape: Number
        The shape of the fibers in the rank

        next_rank: Rank
        The next rank in the tensor

        Attributes
        ----------

        estimated_shape: Boolean
        Is the shape estimated or given

        fibers: List
        A list of the fibers in the rank

        """

        self._id = id

        if shape is None:
            self._estimated_shape = True
            self._shape = 0
        else:
            self._estimated_shape = False
            self._shape = shape

        self.setNextRank(next_rank)

        self.fibers = []

#
# Accessor methods
#
    def getId(self):
        """Return id of rank"""

        return self._id


    def getRankIds(self, all_ranks=True):
        """Return list of ranks"""

        rankids = [self._id]

        if all_ranks and self.next_rank is not None:
            rankids.extend(self.next_rank.getRankIds(all_ranks=True))

        return rankids


    def getName(self):
        """Return name of rank"""

        Rank._deprecated("Use of Rank.getName() is deprecated - use Rank.getId()")

        return self._id


    def getShape(self, all_ranks=True, authoritative=False):
        """Return shape of rank.


        """

        if all_ranks == False:
            #
            # Handle case where user just wants shape of this rank
            #
            if authoritative and self._estimated_shape:
                #
                # We do not know the shape authoritatively
                #
                return None

            if self._shape == 0:
                #
                # We do not actually know the shape
                #
                return 0

            return self._shape

        #
        # Get shape of all ranks
        #
        if authoritative and self._estimated_shape:
            #
            # This will cause the final return to be None
            #
            return None

        shape = [self._shape]

        if self.next_rank is not None:
            rest_of_shape = self.next_rank.getShape(all_ranks=True, authoritative=authoritative)
            if rest_of_shape is None:
                return None

            shape.extend(rest_of_shape)

        #
        # If we didn't have a shape for any rank, assume we don't know anything
        #
        if any([s == 0 for s in shape]):
            return None

        return shape


    def getFibers(self):
        """Return list of fibers in the rank"""

        return self.fibers

    def clearFibers(self):
        """Empty rank of all fibers"""

        self.fibers = []

    #
    # Default payload methods
    #
    def setDefault(self, value):
        """setDefault

        Set the default payload value for fibers in this rank

        Parameters
        ----------
        value: value
        An (unboxed) value to use as the payload value for fibers in this rank

        Returns
        -------
        self:
        So method can be used in a chain

        Raises
        ------
        None

        """

        self._default_is_set = True
        self._default = value

        return self


    def getDefault(self):
        """getDefault

        Get the default payload for fibers in this rank

        Parameters
        ----------
        None

        Returns
        -------
        value: value
        The (unboxed) default payload of fibers in this rank

        Raises
        ------
        None

        """

        assert self._default_is_set

        return self._default


#
# Fundamental methods
#
    def append(self, fiber):
        """
        Append the provided fiber into a rank

        """

        #
        # Get the raw fiber (if it was wrapped in a payload)
        #
        fiber = Payload.get(fiber)

        if self._estimated_shape:
            #
            # Get shape from fiber and see it is larger that current shape
            # making sure we don't get info from a prior owning rank
            #
            # TBD: If the fiber really has a definitive shape then
            # change estimated_shape to True
            #
            fiber.setOwner(None)
            self._shape = max(self._shape, fiber.getShape(all_ranks=False))

        #
        # Set this rank as owner of the fiber
        #
        fiber.setOwner(self)

        #
        # Check default value for new coordinates in the fiber
        #
        # Deprecating use of rank.getDefault()
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

#
# Linked list methods
#
    def setNextRank(self, next_rank):
        """setNextRank

        Record a reference to the next rank. If that rank exists then the
        default payload of fibers in this rank must be a fiber,
        otherwise set the default payload to zero.

        """

        self.next_rank = next_rank

        if next_rank is None:
            self.setDefault(0)
        else:
            self.setDefault(Fiber)


    def getNextRank(self):
        """getNextRank"""

        return self.next_rank

#
# String methods
#

    def __str__(self, indent=0):
        """__str__"""

        string = indent * ' '
        string += f"Rank: {self._id} "

        next_indent = len(string)

        separator = ",\n" + " " * next_indent
        fibers = [x.__str__(indent=next_indent, cutoff=1000, newline=True) for x in self.fibers]
        string += separator.join(fibers)

        return string

    def __repr__(self):
        """__repr__"""

        string = "R(%s)/[" % self._id
        string += ", ".join([x.__repr__() for x in self.fibers])
        string += "]"
        return string

#
# Utility functions
#

    @staticmethod
    def _deprecated(message):
        import warnings

        warnings.warn(message, FutureWarning, stacklevel=3)


