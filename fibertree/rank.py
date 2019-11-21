from fibertree.fiber import Fiber

class Rank:

    def __init__(self, name, next_rank=None):
        self.name = name
        self.next_rank = next_rank
        self.fibers = []

#
# Accessor methods
#
    def get_name(self):
        return self.name

#
# Fundamental methods
#
    def append(self, fiber):
        # Set this rank as owner of the fiber
        fiber.set_owner(self)

        # Set proper default value for new coordinates in the fiber
        if self.next_rank is None:
            fiber.set_default(0)
        else:
            fiber.set_default(Fiber)

        # Add fiber to list of fibers of rank
        self.fibers.append(fiber)

#
# Linked list methods
#
    def get_next(self):
        return self.next_rank

    def set_next(self, next_rank):
        self.next_rank = next_rank

#
# String methods
#

    def __str__(self):
        string = "Rank: %s " % self.name
        string += ", ".join([str(x) for x in self.fibers])
        return string
    
    def __repr__(self):
        string = "R(%s)/[" % self.name
        string += ", ".join([str(x) for x in self.fibers])
        string += "]"
        return string
