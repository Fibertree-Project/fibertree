from fibertree.fiber import Fiber

class Rank:

    def __init__(self, name):
        self.name = name
        self.fibers = []


    def append(self, fiber):
        self.fibers.append(fiber)

    def __str__(self):
        string = "Rank: %s " % self.name
        string += ", ".join([str(x) for x in self.fibers])
        return string
    
    def __repr__(self):
        string = "Rank: %s " % self.name
        string += ", ".join([str(x) for x in self.fibers])
        return string
