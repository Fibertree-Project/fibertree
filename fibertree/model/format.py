#cython: language_level=3
"""Format

A class for computing the true memory footprint of a tensor
"""

from fibertree import Fiber

class Format:
    """Class that represents the true memory footprint of a tensor

    Attributes
    ----------

    tensor: Tensor
        A tensor to store in memory

    spec: dict
        A dictionary where the keys are all ranks, and the values are
        dictionaries with a subset of the following keys:
            - format: "C" or "U" for compressed or uncompressed
            - rhbits: bits required for the rank header (any per-rank bits)
            - fhbits: bits required for the fiber header (any per-fiber bits)
            - cbits: bits required per coordinate
            - pbits: bits required per payload
        There can also be a "root" rank with keys:
            - hbits: bits required for the tensor header
            - pbits: bits required for the payload to the root fiber

    Constructor
    -----------

    The `Format` constructor creates a new queryable Format object

    Parameters
    ----------

    tensor: Tensor
        The tensor we want to represent

    spec: dict
        A dictionary with the properties specified above
    """

    def __init__(self, tensor, spec):
        """__init__"""

        self.tensor = tensor
        self.spec = spec

        self._checkFillSpec()

    def _checkFillSpec(self):
        """Check the spec and fill the spec with any missing fields"""
        # Check root specs
        if "root" not in self.spec.keys():
            self.spec["root"] = {}

        self._checkFillIntField("root", "hbits")
        self._checkFillIntField("root", "pbits")

        assert len(self.spec["root"]) == 2

        # Check rank specs
        ranks = self.tensor.getRankIds()
        for rank in ranks:
            if rank not in self.spec.keys():
                self.spec[rank] = {}

            if "format" not in self.spec[rank].keys():
                self.spec[rank]["format"] = "C"

            assert self.spec[rank]["format"] == "C" \
                or self.spec[rank]["format"] == "U"

            self._checkFillIntField(rank, "rhbits")
            self._checkFillIntField(rank, "fhbits")
            self._checkFillIntField(rank, "cbits")
            self._checkFillIntField(rank, "pbits")

            # No additional fields beyond format, hbits, cbits, and pbits
            # should be specified
            assert len(self.spec[rank]) == 5

    def _checkFillIntField(self, rank, field):
        """Check the specific integer field"""
        if field not in self.spec[rank].keys():
            self.spec[rank][field] = 0

        assert isinstance(self.spec[rank][field], int)

    def getCBits(self, rank):
        """Get the number of bits required to represent the coordinates of
        the given rank"""

        return self.spec[rank]["cbits"]

    def getFHBits(self, rank):
        """Get the number of bits required to represent the fiber headers of
        the given rank"""

        return self.spec[rank]["fhbits"]

    def getPBits(self, rank):
        """Get the number of bits required to represent the payloads of
        the given rank"""

        return self.spec[rank]["pbits"]

    def getRHBits(self, rank):
        """Get the number of bits required to represent the rank headers of
        the given rank"""

        return self.spec[rank]["rhbits"]

    def getFiber(self, *coords):
        """Get the footprint of a single fiber"""
        fiber = self._getFiberFromCoords(*coords)
        rank = fiber.getRankAttrs().getId()

        return self._getFiberFootprint(rank, fiber)

    def getRank(self, rank_id):
        """Get the footprint of a full rank"""
        i = self.tensor.getRankIds().index(rank_id)
        rank = self.tensor.ranks[i]

        total = self.spec[rank_id]["rhbits"]

        for fiber in rank.getFibers():
            total += self._getFiberFootprint(rank_id, fiber)

        return total

    def getRoot(self):
        """Get the footprint of the root"""
        return self.spec["root"]["hbits"] + self.spec["root"]["pbits"]

    def getSubTree(self, *coords):
        """Get a subtree under a given fiber (does not include rhbits)"""
        if len(coords) == len(self.tensor.getShape()):
            return self.spec[self.tensor.getRankIds()[-1]]["cbits"] + \
                self.spec[self.tensor.getRankIds()[-1]]["pbits"]
        fibers = [self._getFiberFromCoords(*coords)]

        total = 0

        while len(fibers) > 0:
            # Add this fiber
            fiber = fibers.pop()
            rank = fiber.getRankAttrs().getId()

            total += self._getFiberFootprint(rank, fiber)

            # Add child fibers
            if self.spec[rank]["format"] == "U":
                iter_ = fiber.iterShape()
            else:
                iter_ = fiber.iterOccupancy()

            for _, payload in iter_:
                if isinstance(payload, Fiber):
                    fibers.append(payload)

        return total

    def getTensor(self):
        """Get the footprint of the entire tensor"""

        total = self.getRoot()
        for rank in self.tensor.getRankIds():
            total += self.getRank(rank)

        return total

    def _getFiberFromCoords(self, *coords):
        """Get a fiber from the given coordinates"""
        # Get the fiber
        if len(coords) == 0:
            fiber = self.tensor.getRoot()
        else:
            fiber = self.tensor.getPayload(*coords)

        # Should not be a payload
        assert isinstance(fiber, Fiber)

        return fiber

    def _getFiberFootprint(self, rank, fiber):
        """Get the footprint of a fiber"""
        # Get the number of elements
        if self.spec[rank]["format"] == "C":
            num_elems = len(fiber)
        else:
            num_elems = fiber.getShape(all_ranks=False)

        return self.spec[rank]["fhbits"] \
            + self.spec[rank]["pbits"] * num_elems \
            + self.spec[rank]["cbits"] * num_elems
