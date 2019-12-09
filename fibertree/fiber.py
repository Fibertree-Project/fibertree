"""Fiber"""

import yaml

from fibertree.payload import Payload

class Fiber:
    """Fiber class"""

    def __init__(self, coords=None, payloads=None, default=0):
        """__init__"""

        if coords is None:
            if payloads is None:
                # If neither coords or payloads are given create an empty fiber
                coords = []
                payloads = []
            else:
                # If only payloads are given create a "dense" fiber
                coords = range(len(payloads))
        else:
            if payloads is None:
                # If only coords are given create a blank set of payloads
                payloads = [default for x in range(len(coords))]


        assert (len(coords) == len(payloads)), "Coordinates and payloads must be same length"

        # Note:
        #    We do not eliminate explicit zeros in the payloads
        #    so zeros will be preserved

        self.coords = coords
        self.payloads = [self._maybe_box(p) for p in payloads]

        # Owner rank... set on append to rank
        self.owner = None

        # Default value assigned to new coordinates
        self.setDefault(default)


    @classmethod
    def fromYAMLfile(cls, yamlfile, default=0):
        """Construct a Fiber from a YAML file"""

        (coords, payloads) = Fiber.parse(yamlfile, default)

        return cls(coords, payloads, default=default)


    @classmethod
    def fromUncompressed(cls, payload_list):
        """Construct a Fiber from an uncompressed nest of lists

        Note: Zero values and sub-fibers that are all zeros
        are squeezed out, i.e., they will have no coordinates.
        Unless the entire input is zeros.

        """

        f = Fiber._makeFiber(payload_list)

        if f is None:
            # Return something for an entirely empty input
            return Fiber([], [])

        return f


    @staticmethod
    def _makeFiber(payload_list):
        """Recursively make a fiber out of an uncompressed nest of lists"""

        assert(isinstance(payload_list, list))

        # Create zipped list of (non-empty) coordinates/payloads
        zipped = [ (c, p) for c, p in enumerate(payload_list) if p != 0 ]

        # Recursively unzip the lists into a Fiber
        if len(zipped) == 0:
            # Got an empty subtree
            return None

        if isinstance(payload_list[0], list):
            coords = []
            payloads = []
            for c,p in zipped:
                real_p = Fiber._makeFiber(p)
                if not real_p is None:
                    coords.append(c)
                    payloads.append(real_p)
        else:
            coords = [ c for c,_ in zipped ]
            payloads = [ p for _,p in zipped ]

        if len(coords) == 0:
            return None

        return Fiber(coords, payloads)




#
# Accessor methods
#
    def getCoords(self):
        """Return list of coordinates in fiber"""

        return self.coords

    def getPayloads(self):
        """Return list of payloads in fiber"""

        return self.payloads

    def getPayload(self, coord):
        """payload"""

        try:
            index = self.coords.index(coord)
            return self.payloads[index]
        except:
            return None

    def setDefault(self, default):
        """setDefault"""

        self.default = default

    def setOwner(self, owner):
        """setOwner"""

        self.owner = owner

    def minCoord(self):
        """min_coord"""

        return min(self.coords)

    def maxCoord(self):
        """max_coord"""

        return max(self.coords)

    def values(self):
        """Count values in the fiber tree

        Note: an explcit zero scalar value will count as a value
        """

        count = 0
        for p in self.payloads:
            if Payload.contains(p, Fiber):
                count += Payload.get(p).values()
            else:
                count += 1

        return count


    def __len__(self):
        """__len__"""

        return len(self.coords)


    def isEmpty(self):
        """isEmpty() - check if Fiber is empty

        Empty is defined as of zero length, only containing zeros
        or only containing subfibers that are empty.
        """

        return all(map(Fiber._checkEmpty, self.payloads))


    @staticmethod
    def _checkEmpty(p):

        if isinstance(p, Fiber):
            return p.isEmpty()

        if (p == 0):
            return  True

        return False
#
# Iterator methods
#

    def __iter__(self):
        """__iter__"""

        for i in range(len(self.coords)):
            yield (self.coords[i], self.payloads[i])

    def __reversed__(self):
        """Return reversed fiber"""

        for coord, payload in zip(reversed(self.coords), reversed(self.payloads)):
            yield (coord, payload)


#
# Core methods
#

    def payload(self, coord):
        """payload"""

        assert(False)
        print("Info: Used deprecated function Fiber.payload() should be getPayload()")

        return self.getPayload(coord)



    def insert(self, coord, value):
        """insert"""

        payload = self._maybe_box(value)

        try:
            index = next(x for x, val in enumerate(self.coords) if val > coord)
            self.coords.insert(index, coord)
            self.payloads.insert(index, payload)
        except StopIteration:
            self.coords.append(coord)
            self.payloads.append(payload)


    def project(self, trans_fn=None, interval=None):
        """project"""

        if trans_fn is None:
            # Default trans_fn is identify function (inefficient but easy implementation)
            trans_fn = lambda x: x

        # Invariant: trans_fn is order preserving, but we check for reversals

        if interval is None:
            # All coordinates are legal

            coords = [ trans_fn(c) for c in self.coords ]
            payloads = self.payloads
        else:
            # Only pass coordinates in [ interval[0], interval[1] )

            min = interval[0]
            max = interval[1]

            coords = []
            payloads = []

            for c,p in zip(self.coords, self.payloads):
                new_c = trans_fn(c)
                if new_c >= min and new_c < max:
                    coords.append(new_c)
                    payloads.append(p)

            # Note: This reversal implies a complex read order

            if len(coords) > 1 and coords[1] < coords[0]:
                coords.reverse()
                payloads.reverse()

        return Fiber(coords, payloads)

    def updatePayloads(self, func):

        for i in range(len(self.payloads)):
            self.payloads[i] = func(self.payloads[i])


    def unzip(self):
        """Unzip"""

        coords_a = list(self.coords)
        coords_b = list(self.coords)

        (payloads_a, payloads_b) = zip(*self.payloads)

        return (Fiber(coords_a, payloads_a), Fiber(coords_b, payloads_b))

#
# Shape-related methods
#

    def getShape(self):
        """Return shape of fiber tree"""
        
        max_coord = self.maxCoord()

        return self._calcShape(shape=[], level=0)

    def _calcShape(self, shape, level):
        """Find the maximum coordinate at each level of the tree"""

        #
        # Conditionaly append a new level to the shape array
        #
        if len(shape) < level+1:
            shape.append(0)
        
        #
        # Update shape for this Fiber at this level
        #
        shape[level] = max(shape[level], self.maxCoord()+1)

        if Payload.contains(self.payloads[0], Fiber):
            #
            # Process payloads that are Fibers
            #
            for p in self.payloads:
                Payload.get(p)._calcShape(shape, level+1)

        return shape


    def uncompress(self, shape=None, level=0):
        """Return an uncompressed fiber tree (i.e., a nest of lists)"""

        if shape is None:
            shape = self.getShape()

        f = [ ]

        for c, (mask, p, _) in self | Fiber(range(shape[level])):

            if (mask == "AB"):
                if Payload.contains(p, Fiber):
                    f.append(Payload.get(p).uncompress(shape, level+1))
                else:
                    f.append(Payload.get(p))

            if (mask == "B"):
                f.append(self._fillempty(shape, level+1))

        return f

    def _fillempty(self, shape, level):
        """Recursive fill empty"""

        if level+1 > len(shape):
            return 0

        f = []
    
        for i in range(shape[level]):
            f.append(self._fillempty(shape, level+1))

        return f

            
#
# Split methods
#


    def splitUniform(self, step, partitions=1):
        """splitUniform"""

        class _SplitterUniform():

            def __init__(self, step):
                self.step = step
                self.cur_group = 0

            def nextGroup(self, i, c):
                if c >= self.cur_group:
                    self.cur_group += self.step
                    return True

                return False

        splitter = _SplitterUniform(step)

        return self._splitGeneric(splitter, partitions)


    def splitNonUniform(self, splits, partitions=1):
        """splitNonUniform"""

        class _SplitterNonUniform():

            def __init__(self, splits):
                self.splits = splits
                self.cur_split = self.splits.pop(0)

            def nextGroup(self, i, c):
                if c >= self.cur_split:
                    if self.splits:
                        self.cur_split = self.splits.pop(0)
                    else:
                        self.cur_split = float("inf")

                    return True

                return False

        splitter = _SplitterNonUniform(splits)

        return self._splitGeneric(splitter, partitions)


    def splitEqual(self, step, partitions=1):
        """splitEqual"""

        class _SplitterEqual():

            def __init__(self, step):
                self.step = step
                self.cur_count=0

            def nextGroup(self, i, c):
                if i >= self.cur_count:
                    self.cur_count += self.step
                    return True

                return False

        splitter = _SplitterEqual(step)

        return self._splitGeneric(splitter, partitions)


    def splitUnEqual(self, sizes, partitions=1):
        """splitUnEqual"""

        class _SplitterUnEqual():

            def __init__(self, sizes):
                self.sizes = sizes
                self.cur_count = -1

            def nextGroup(self, i, c):
                if i > self.cur_count:
                    if self.sizes:
                        self.cur_count += self.sizes.pop(0)
                    else:
                        self.cur_count = float("inf")

                    return True

                return False

        assert len(self.coords) <= sum(sizes)

        splitter = _SplitterUnEqual(sizes)

        return self._splitGeneric(splitter, partitions)


    def _splitGeneric(self, splitter, partitions):
        """_splitGeneric"""

        fiber_coords = []
        fiber_payloads = []
        fibers = []

        # Create arrays of |partitions| 

        for i in range(partitions):
            fiber_coords.append([])
            fiber_payloads.append([])
            fibers.append([])

        partition =  0

        # Split apart the fiber into groups according to "splitter"

        for i, (c,p) in enumerate(zip(self.coords,self.payloads)):
            # Check if we need to start a new fiber
            if splitter.nextGroup(i, c):
                cur_coords = []
                fiber_coords[partition].append(cur_coords)
                cur_payloads = []
                fiber_payloads[partition].append(cur_payloads)
                partition = (partition + 1) % partitions

            cur_coords.append(c)
            cur_payloads.append(p)

        # Deal the split fibers out to the partitions

        for i in range(partitions):
            for c, p in zip(fiber_coords[i], fiber_payloads[i]):
                fibers[i].append(Fiber(c, p))

        # For 1 partition don't return a extra level of Fiber

        if partitions == 1:
            return Fiber(payloads=fibers[0])

        # For >1 partitions return a Fiber with a payload for each partition

        payloads = []

        for f in fibers:
            payload = Fiber(payloads=f)
            payloads.append(payload)

        return Fiber(payloads=payloads)



#
# Merge methods
#
    def __and__(self, other):
        """__and__

        Return the intersection of "self" and "other" by considering all possible
        coordinates and returning a fiber consisting of payloads containing
        a tuple of the payloads of the inputs for coordinates where the
        following truth table returns True:


                         coordinate not     |      coordinate
                        present in "other"  |    present in "other"
                    +-----------------------+-----------------------+
                    |                       |                       |
        coordinate  |                       |                       |
        not present |         False         |        False          |
        in "self"   |                       |                       |
                    |                       |                       |
        ------------+-----------------------+-----------------------+
                    |                       |                       |
        coordinate  |                       |                       |
        present in  |         False         |        True           |
        "self"      |                       |                       |
                    |                       |                       |
        ------------+-----------------------+-----------------------+

        """

        def get_next(iter):
            """get_next"""

            try:
                coord, payload = next(iter)
            except StopIteration:
                return (None, None)
            return (coord, payload)

        a = self.__iter__()
        b = other.__iter__()

        z_coords = []
        z_payloads = []

        a_coord, a_payload = get_next(a)
        b_coord, b_payload = get_next(b)

        while not (a_coord is None or b_coord is None):
            if a_coord == b_coord:
                z_coords.append(a_coord)
                z_payloads.append((a_payload, b_payload))

                a_coord, a_payload = get_next(a)
                b_coord, b_payload = get_next(b)
                continue

            if a_coord < b_coord:
                a_coord, a_payload = get_next(a)
                continue

            if a_coord > b_coord:
                b_coord, b_payload = get_next(b)
                continue

        return Fiber(z_coords, z_payloads)


    def __or__(self, other):
        """__or__

        Return the union of "self" and "other" by considering all possible
        coordinates and returning a fiber consisting of payloads containing
        a tuple of the payloads of the inputs for coordinates where the
        following truth table returns True:


                         coordinate not     |      coordinate
                        present in "other"  |    present in "other"
                    +-----------------------+-----------------------+
                    |                       |                       |
        coordinate  |                       |                       |
        not present |         False         |        True           |
        in "self"   |                       |                       |
                    |                       |                       |
        ------------+-----------------------+-----------------------+
                    |                       |                       |
        coordinate  |                       |                       |
        present in  |         True          |        True           |
        "self"      |                       |                       |
                    |                       |                       |
        ------------+-----------------------+-----------------------+

        """


        def get_next(iter):
            """get_next"""

            try:
                coord, payload = next(iter)
            except StopIteration:
                return (None, None)
            return (coord, payload)

        a = self.__iter__()
        b = other.__iter__()

        z_coords = []
        z_payloads = []

        a_coord, a_payload = get_next(a)
        b_coord, b_payload = get_next(b)

        while not (a_coord is None or b_coord is None):
            if a_coord == b_coord:
                z_coords.append(a_coord)
                z_payloads.append(("AB", a_payload, b_payload))

                a_coord, a_payload = get_next(a)
                b_coord, b_payload = get_next(b)
                continue

            if a_coord < b_coord:
                z_coords.append(a_coord)
                # TODO: Append the right b_payload, e.g., maybe a Fiber()
                z_payloads.append(("A", a_payload, 0))

                a_coord, a_payload = get_next(a)
                continue

            if a_coord > b_coord:
                z_coords.append(b_coord)
                # TODO: Append the right a_payload, e.g., maybe a Fiber()
                z_payloads.append(("B", 0, b_payload))

                b_coord, b_payload = get_next(b)
                continue

        while not a_coord is None:
            z_coords.append(a_coord)
            z_payloads.append(("A", a_payload, 0))

            a_coord, a_payload = get_next(a)

        while  not b_coord is None:
            z_coords.append(b_coord)
            z_payloads.append(("B", 0, b_payload))

            b_coord, b_payload = get_next(b)

        return Fiber(z_coords, z_payloads)



    def __lshift__(self, other):
        """__lshift__

        Return the "assignment" of "other" to "self" by considering all possible
        coordinates and returning a fiber consisting of payloads containing
        a tuple of the payloads of the inputs for coordinates where the
        following truth table returns True:


                         coordinate not     |      coordinate
                        present in "other"  |    present in "other"
                    +-----------------------+-----------------------+
                    |                       |                       |
        coordinate  |                       |                       |
        not present |         False         |        True           |
        in "self"   |                       |                       |
                    |                       |                       |
        ------------+-----------------------+-----------------------+
                    |                       |                       |
        coordinate  |                       |                       |
        present in  |         False         |        True           |
        "self"      |                       |                       |
                    |                       |                       |
        ------------+-----------------------+-----------------------+

        """


        def get_next(iter):
            """get_next"""

            try:
                coord, payload = next(iter)
            except StopIteration:
                return (None, None)
            return (coord, payload)

        # "a" is self!
        b = other.__iter__()

        z_coords = []
        z_payloads = []

        b_coord, b_payload = get_next(b)

        while not b_coord is None:
            z_coords.append(b_coord)

            # TBD: Optimize with co-iteration...

            a_payload = self.getPayload(b_coord)
            if a_payload is None:
                # Iemporary value (should be None)
                if callable(self.default):
                    value = self.default()
                else:
                    value = self.default

                self.insert(b_coord, value)

                # TBD: Inefficient since it does yet another search

                a_payload = self.getPayload(b_coord)

                if Payload.contains(value, Fiber):
                    assert(not self.owner is None)
                    next_rank = self.owner.get_next()
                    if not next_rank is None:
                        next_rank.append(a_payload)

            z_payloads.append((a_payload, b_payload))
            b_coord, b_payload = get_next(b)

        return Fiber(z_coords, z_payloads)

    def __sub__(self, other):
        """__sub__

        Return the "diffence" of "other" from "self" by considering all possible
        coordinates and returning a fiber consisting of payloads containing
        a tuple of the payloads of the inputs for coordinates where the
        following truth table returns True:


                         coordinate not     |      coordinate
                        present in "other"  |    present in "other"
                    +-----------------------+-----------------------+
                    |                       |                       |
        coordinate  |                       |                       |
        not present |         False         |        False          |
        in "self"   |                       |                       |
                    |                       |                       |
        ------------+-----------------------+-----------------------+
                    |                       |                       |
        coordinate  |                       |                       |
        present in  |          True         |        False          |
        "self"      |                       |                       |
                    |                       |                       |
        ------------+-----------------------+-----------------------+

        """


        def get_next(iter):
            """get_next"""

            try:
                coord, payload = next(iter)
            except StopIteration:
                return (None, None)
            return (coord, payload)

        a = self.__iter__()
        b = other.__iter__()

        z_coords = []
        z_payloads = []

        a_coord, a_payload = get_next(a)
        b_coord, b_payload = get_next(b)

        while not (a_coord is None or b_coord is None):
            if a_coord == b_coord:
                a_coord, a_payload = get_next(a)
                b_coord, b_payload = get_next(b)
                continue

            if a_coord < b_coord:
                z_coords.append(a_coord)
                # TODO: Append the right b_payload, e.g., maybe a Fiber()
                z_payloads.append(a_payload)

                a_coord, a_payload = get_next(a)
                continue

            if a_coord > b_coord:
                b_coord, b_payload = get_next(b)
                continue

        while not a_coord is None:
            z_coords.append(a_coord)
            z_payloads.append(a_payload)

            a_coord, a_payload = get_next(a)

        return Fiber(z_coords, z_payloads)


#
# Multilayer methods
#
    def swapRanks(self):
        """Swap the (highest) two ranks of the fiber.
        This function relies on flattenRanks() and unflattenRanks().
        FIXME: flattenRanks() could be more general to support all p1 types,
        including tuples."""

        # Flatten the (highest) two ranks
        flattened = self.flattenRanks()
        # Make sure the coord is a 2-element tuple
        assert(len(flattened.coords[0]) == 2)

        # Swap the ranks and sort based on the swapped (new) structure
        swapped = sorted([ (c[::-1], p) for c, p in flattened ])

        # Return the unflattened fiber
        coords = [ c for c,_ in swapped]
        payloads = [ p for _,p in swapped ]
        return Fiber(coords, payloads).unflattenRanks()

    def flattenRanks(self):
        """Flatten two ranks into one - COO-style"""

        coords = []
        payloads = []

        for c1, p1 in zip(self.coords, self.payloads):
            if Payload.contains(p1, Fiber):
                for c0, p0 in p1:
                    coords.append( (c1, c0) )
                    payloads.append(p0)
            elif Payload.contains(p1, tuple):
                # zgw: p1 could be tuples.
                # In general, a payload contains one the the following three
                # 1) a Fiber
                # 2) a tuple
                # 3) a (fibertree-)structure-irrelevant type (e.g., a Python number)
                # If p1 is a tuple, flattenRanks() flattens rank c1 and c0,
                # where c0 is the highest rank of the fiber p1[0]
                assert(isinstance(p1[0], Fiber))
                for c0, p0 in p1[0]:
                    coords.append( (c1, c0) )
                    payloads.append( (p0,) + p1[1:] )
        return Fiber(coords, payloads)


    def unflattenRanks(self):
        """Unflatten two ranks into one"""

        assert(isinstance(self.coords[0], tuple))

        coords1 = []
        payloads1 = []

        c1_last = -1

        for ( (c1, c0), p0 ) in zip(self.coords, self.payloads):
            if (c1 > c1_last):
                if c1_last != -1:
                     coords1.append(c1_last)
                     payloads1.append(Fiber(coords0, payloads0))

                c1_last = c1
                coords0 = []
                payloads0 = []

            coords0.append(c0)
            payloads0.append(p0)

        coords1.append(c1_last)
        payloads1.append(Fiber(coords0, payloads0))

        return Fiber(coords1, payloads1)

#
#  Comparison operations
#

    def __eq__(self, other):
        """__eq__ - Equality check for Fibers

        Note: explict zeros do not result in inequality
        """

        for c, (mask, ps, po) in self | other:
            if mask == "A" and not Fiber._checkEmpty(ps):
                return False

            if mask == "B" and not Fiber._checkEmpty(po):
                return False

            if mask == "AB" and not (ps == po):
                return False

        return True

#
#  String methods
#
    def print(self, title=None):
        """print"""

        if not title is None:
            print("%s" % title)

        print("%s" % self)
        print("")

    def __repr__(self):
        """__repr__"""

        if self.owner is None:
            str = "F/["
        else:
            str = "F(%s)/[" % self.owner.getName()

        for i in range(len(self.coords)):
            str += "(%s -> %s) " % (self.coords[i], self.payloads[i])

        str += "]"
        return str

#
# Yaml input/output methods
#

    @staticmethod
    def parse(yamlfile, default):
        """Parse a yaml file containing a tensor"""

        with open(yamlfile, 'r') as stream:
            try:
                y_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)

        #
        # Make sure key "fiber" exists
        #
        if not isinstance(y_file, dict) or 'fiber' not in y_file:
            print("Yaml is not a fiber")
            exit(1)

        newfiber = Fiber.dict2fiber(y_file)

        return (newfiber.getCoords(), newfiber.getPayloads())



    def dump(self, yamlfile):
        """Dump a tensor to a file in YAML format"""

        fiber_dict = self.fiber2dict()

        with open(yamlfile, 'w') as stream:
            document = yaml.dump(fiber_dict, stream)


# Conversion methods - to/from dictionaries
#

    @staticmethod
    def dict2fiber(y_payload_dict, level=0):
        """Parse a yaml-based tensor payload, creating Fibers as appropriate"""

        if isinstance(y_payload_dict, dict) and 'fiber' in y_payload_dict:
            # Got a fiber, so need to get into the Fiber class

            y_fiber = y_payload_dict['fiber']

            #
            # Error checking
            #
            if not isinstance(y_fiber, dict):
                print("Malformed payload")
                exit(0)

            if 'coords' not in y_fiber:
                print("Malformed fiber")
                exit(0)

            if 'payloads' not in y_fiber:
                print("Malformed fiber")
                exit(0)

            #
            # Process corrdinates and payloads
            #
            f_coords = y_fiber['coords']
            y_f_payloads = y_fiber['payloads']

            f_payloads = []
            for y_f_payload in y_f_payloads:
                f_payloads.append(Fiber.dict2fiber(y_f_payload, level+1))

            #
            # Turn into a fiber
            #
            subtree = Fiber(coords=f_coords, payloads=f_payloads)
        else:
            # Got scalars, so format is unchanged
            subtree = y_payload_dict

        return subtree



    def fiber2dict(self):
        """Return dictionary with fiber information"""

        f = { 'fiber' :
              { 'coords'   : self.coords,
                'payloads' : [ self.payload2dict(p) for p in self.payloads ]
              }
        }

        return f

    def payload2dict(self, payload):
        """Return payload converted to dictionry or simple value"""

        if isinstance(payload, Fiber):
            # Note: this leg is deprecated and should be removed
            return payload.fiber2dict()
        elif isinstance(payload, Payload):
            if Payload.contains(payload, Fiber):
                return payload.value.fiber2dict()
            else:
                return payload.value
        else:
            return payload

#
# Utility functions
#

    def _maybe_box(self, value):
        """_maybe_box"""

        if isinstance(value, (float, int)):
            return Payload(value)

        return value



if __name__ == "__main__":

    a = Fiber([2, 4, 6], [3, 5, 7])

    print("Simple print")
    a.print()
    print("----\n\n")

