from fibertree.fiber import Fiber
from fibertree.payload import Payload

""" Rank """

class Rank:
    """ Rank class """

    def __init__(self, id, shape=None, next_rank=None):
        """__init__"""

        self.id = id

        if shape is None:
            self.estimated_shape = True
            self.shape= 0
        else:
            self.estimated_shape = False
            self.shape = shape

        self.setNextRank(next_rank)

        self.fibers = []

#
# Accessor methods
#
    def getId(self):
        """Return id of rank"""

        return self.id


    def getName(self):
        """Return name of rank"""

        # Deprecated

        return self.id


    def getShape(self, all_ranks=True):
        """Return shape of rank"""

        shape = [self.shape]

        if all_ranks and self.next_rank is not None:
            shape.extend(self.next_rank.getShape(all_ranks=True))

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
        """append"""

        #
        # Get the raw fiber (if it was wrapped in a payload)
        #
        fiber = Payload.get(fiber)

        if self.estimated_shape:
            #
            # Get shape from fiber and see it is larger that current shape
            # making sure we don't get info from a prior owning rank
            #
            # TBD: If the fiber really has a definitive shape then
            # change estimated_shape to True
            #
            fiber.setOwner(None)
            self.shape = max(self.shape, fiber.getShape(all_ranks=False)[0])

        #
        # Set this rank as owner of the fiber
        #
        fiber.setOwner(self)

        #
        # Check default value for new coordinates in the fiber
        #
        if self.next_rank is None:
            assert self.getDefault() != Fiber, "Leaf rank default should not be Fiber"
        else:
            assert self.getDefault() == Fiber, "Non-leaf rank default should be Fiber"

        #
        # Add fiber to list of fibers of rank
        #
        self.fibers.append(fiber)

#
# Linked list methods
#
    def setNextRank(self, next_rank):
        """setNextRank"""

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

        string = indent*' '
        string += f"Rank: {self.id} "

        next_indent = len(string)

        separator = ",\n" + " "*next_indent
        fibers = [x.__str__(indent=next_indent, cutoff=1000, newline=True) for x in self.fibers]
        string += separator.join(fibers)

        return string
    
    def __repr__(self):
        """__repr__"""

        string = "R(%s)/[" % self.id
        string += ", ".join([x.__repr__() for x in self.fibers])
        string += "]"
        return string
