from fibertree.fiber import Fiber
from fibertree.payload import Payload

""" Rank """

class Rank:
    """ Rank class """

    def __init__(self, name, next_rank=None):
        """__init__"""

        self.name = name
        self.next_rank = next_rank
        self.fibers = []

#
# Accessor methods
#
    def getName(self):
        """Return name of rank"""

        return self.name


    def getFibers(self):
        """Return list of fibers in the rank"""

        return self.fibers

    def clearFibers(self):
        """Return list of fibers in the rank"""

        self.fibers = []

#
# Fundamental methods
#
    def append(self, fiber):
        """append"""

        fiber = Payload.get(fiber)

        # Set this rank as owner of the fiber
        fiber.setOwner(self)

        # Set proper default value for new coordinates in the fiber
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
        string += f"Rank: {self.name} "

        next_indent = len(string)

        string += ", ".join([x.__str__(indent=next_indent) for x in self.fibers])
        return string
    
    def __repr__(self):
        """__repr__"""

        string = "R(%s)/[" % self.name
        string += ", ".join([x.__repr__() for x in self.fibers])
        string += "]"
        return string
