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

        self.next_rank = next_rank
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
        # Set proper default value for new coordinates in the fiber
        #
        # TBD: Move this information to rank...
        #
        if self.next_rank is None:
            fiber.setDefault(0)
        else:
            fiber.setDefault(Fiber)

        # Add fiber to list of fibers of rank
        self.fibers.append(fiber)

#
# Linked list methods
#
    def get_next(self):
        """get_next"""

        return self.next_rank

    def set_next(self, next_rank):
        """set_next"""

        self.next_rank = next_rank

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
