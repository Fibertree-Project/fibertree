"""Fiber"""

import yaml

from fibertree.payload import Payload

class Fiber:
    """Fiber class"""

    def __init__(self, coords=None, payloads=None, default=0, yamlfile=None):
        """__init__"""

        if not yamlfile is None:
            self.parse(yamlfile, default)
            return

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

        self.coords = coords
        self.payloads = [self._maybe_box(p) for p in payloads]

        # Owner rank... set on append to rank
        self.owner = None

        # Default value assigned to new coordinates
        self.setDefault(default)


    @classmethod
    def fromUncompressed(cls, payload_list):
        return Fiber._makeFiber(payload_list)


    @staticmethod
    def _makeFiber(payload_list):
        """Make a fiber out of an uncompressed list"""
        coords = []
        payloads = []

        for rownum, row in enumerate(payload_list):
            if isinstance(row[0], list):
                subfibers = Fiber._makeFiber(row)
                if len(subfibers) > 0:
                    coords.append(rownum)
                    payloads.append(subfibers)
            else:
                zipped = [ (c, p) for c, p in enumerate(row) if p != 0 ]
                if len(zipped) > 0:
                    subcoords = [ c for c,_ in zipped ]
                    subpayloads = [ p for _,p in zipped ]

                    new_fiber = Fiber(subcoords, subpayloads)
                    coords.append(rownum)
                    payloads.append(new_fiber)

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
        """Count values in the fiber tree"""

        count = 0
        for p in self.payloads:
            if isinstance(p, Fiber):
                count += p.values()
            else:
                count += 1

        return count

    def __len__(self):
        """__len__"""

        return len(self.coords)

    def __iter__(self):
        """__iter__"""

        for i in range(len(self.coords)):
            yield (self.coords[i], self.payloads[i])
#
# Fundamental methods
#

    def payload(self, coord):
        """payload"""

        try:
            index = self.coords.index(coord)
            return self.payloads[index]
        except:
            return None


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


    def unzip(self):
        """Unzip"""

        coords_a = list(self.coords)
        coords_b = list(self.coords)

        (payloads_a, payloads_b) = zip(*self.payloads)

        return (Fiber(coords_a, payloads_a), Fiber(coords_b, payloads_b))

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
                if self.splits and c >= self.cur_split:
                    self.cur_split = self.splits.pop(0)
                    return True

                return False

        splitter = _SplitterNonUniform(splits)

        return self._splitGeneric(splitter, partitions)


    def splitEqual(self, step, partitions=1):
        """splitEqual"""

        class _SplitterEqual():

            def __init__(self, splits):
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
                if self.sizes and i > self.cur_count:
                    self.cur_count += self.sizes.pop(0)
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
#  Merge operatons
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
                    |                       |                       |                    |
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
                    |                       |                       |                    |
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

            a_payload = self.payload(b_coord)
            if a_payload is None:
                # Iemporary value (should be None)
                if callable(self.default):
                    value = self.default()
                else:
                    value = self.default

                self.insert(b_coord, value)
                # Inefficient: another search
                a_payload = self.payload(b_coord)

                # Try to insert fiber into next rank
                if not self.owner is None:
                    next_rank = self.owner.get_next()
                    if not next_rank is None:
                        next_rank.append(a_payload)

            z_payloads.append((a_payload, b_payload))
            b_coord, b_payload = get_next(b)

        return Fiber(z_coords, z_payloads)

#
#  Comparison operations
#

    def __eq__(self, other):
        """__eq__"""

        if self.coords != other.coords:
            return False

        if self.payloads != other.payloads:
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

    def parse(self, yamlfile, default):
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

        #
        # Copy newfile into self
        #    Note: make sure this is all the fields!
        #
        self.coords = newfiber.coords
        self.payloads = newfiber.payloads

        # Owner rank... set on append to rank
        self.owner = None

        # Default value assigned to new coordinates
        self.setDefault(default)


    def dump(self, yamlfile):
        """Dump a tensor to a file in YAML format"""

        fiber_dict = self.fiber2dict()

        with open(yamlfile, 'w') as stream:
            document = yaml.dump(fiber_dict, stream)


# Conversion methods - to/from dictionaries
#

    @staticmethod
    def dict2fiber(y_payload_dict, ranks=None, level=0):
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
                f_payloads.append(Fiber.dict2fiber(y_f_payload, ranks, level+1))

            #
            # Turn into a fiber
            #
            subtree = Fiber(coords=f_coords, payloads=f_payloads)

            #
            # Add fiber into appropriate rank
            #  Hack: used when called as part of tensor creation
            #
            if isinstance(ranks, list):
                ranks[level].append(subtree)
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
        """Convert payload to dictionry"""

        if isinstance(payload, Fiber):
            return payload.fiber2dict()
        elif isinstance(payload, Payload):
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

