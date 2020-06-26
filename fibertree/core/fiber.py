"""Fiber"""

from collections import namedtuple
from functools import partialmethod
import yaml
import random

from .payload import Payload

#
# Define a named tuple for coordinate/payload pairs
#
CoordPayload = namedtuple('CoordPayload', 'coord payload')

#
# Define an error class
#
class CoordinateError(Exception):
    """CoordinateError"""
    pass


#
# Define the fiber class
#
class Fiber:
    """ Fiber class

    A fiber of a tensor containing a list of coordinates and
    asssociated payloads.

    Note: The coords and payloads instance variables are currently
    left public...

    """

    def __init__(self,
                 coords=None,
                 payloads=None,
                 default=0,
                 initial=None,
                 max_coord=None):
        """__init__

        Create a fiber....

        Parameters:

        coords: list (default: [])
        List of coordinate values

        payloads: list (default: [])
        List of corresponding payloads for the coordinates

        initial: value (default: None)
        A value to initialize all payloads.

        max_coord: value (default: no maximum coordinate)
        The maximum legal coordinate value

        """

        assert default == 0, "Fiber defaults are no longer supported in constructor"

        if coords is None:
            if payloads is None:
                #
                # If neither coords or payloads are given create an empty fiber
                #
                coords = []
                payloads = []
            else:
                #
                # If only payloads are given create a "dense" fiber
                #
                coords = range(len(payloads))
        else:
            if payloads is None:
                #
                # If only coords are given create a set of payloads
                #
                # TBD: Creating a set of zeros is odd...
                #
                payloads = [initial for x in range(len(coords))]


        assert (len(coords) == len(payloads)), "Coordinates and payloads must be same length"

        #
        # Note:
        #    We do not eliminate explicit zeros in the payloads
        #    so zeros will be preserved
        #
        self.coords = coords
        self.payloads = [self._maybe_box(p) for p in payloads]

        #
        # Set a specific constant value to the maximum legal coordinate
        #
        # TBD: If not None there are lots of places this should be checked
        #
        self.max_legal_coord = max_coord

        #
        # Owner rank... set on append to rank
        #
        self.setOwner(None)

        #
        # Create default value
        #
        self._default = 0

        #
        # Initialize "saved position"
        #
        self._saved_pos = 0

        #
        # Clear all stats
        #
        self.clearStats()


    @classmethod
    def fromCoordPayloadList(cls, *cp, default=0):
        """Construct a Fiber from a coordinate/payload list

        Parameters
        ----------

        cp: sequence of (coord, payload) tuples

        """

        (coords, payloads) = zip(*cp)

        return cls(coords, payloads, default=default)


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

        #
        # Check if the list was all zeros, so return an empty fiber
        #
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

        return Fiber(coords, payloads, max_coord=len(coords)-1)


    @classmethod
    def fromRandom(cls, shape, density, interval=10, seed=None):
        """ Recursively create a random fiber tree

        Parameters:
        -----------

        shape - list
        The "shape" (i.e., size) of each level of the tree

        density - list
        The probability that an element of the fiber will not be empty
        for each level of the tree

        interval - number
        The range (from 0 to "interval") of each value at the leaf of the tree

        seed - a valid argument for random.seed
        A seed to pass to random.seed

        """

        if seed is not None:
            random.seed(seed)

        coords = []
        payloads = []

        for c in range(shape[0]):
            if random.random() < density[0]:
                if len(shape) == 1:
                    payload = random.randint(1, interval)
                    if payload == 0:
                        break
                else:
                    payload = Fiber.fromRandom(shape[1:],
                                               density[1:],
                                               interval)
                    if payload.isEmpty():
                        break

                coords.append(c)
                payloads.append(payload)

        f = Fiber(coords, payloads)

        return f


#
# Stats-related methods
#

    def clearStats(self):
        """clearStats"""

        self._clearSavedPosStats()


#
# Accessor methods
#
    def getCoords(self):
        """Return list of coordinates in fiber"""

        return self.coords

    def getPayloads(self):
        """Return list of payloads in fiber"""

        return self.payloads


#
# Coordinate-based methods
#
# The following set of methods all reference an element in the fiber
# by coordinate. Some just obtain information about the element (non-mutating)
# others change the content of the fiber (mutating)
#

    def getPayload(self, *coords, default=None, allocate=True, start_pos=None):
        """getPayload

        Return the final payload after recursively traversing the
        levels of the fiber tree for at each coordinate in coords. If
        the list of coordinates reaches a leaf of the tree, it returns
        a value otherwise it will return a fiber. [Non-mutating]

        This method operates in two modes: allocate=True and False.

        For "allocate=True" mode, if any coordinate refers to a
        non-existant element, a payload is created (a fiber for at a
        non-leaf level or a zero at the leaf) recursively, but not
        inserted into the fiber tree. The final such payload is
        returned to the caller.

        For "allocate=False" mode, if any coordinate refers to a
        non-existant element, nothing is created and the "default"
        value is returned.

        If "start_pos" is specified it is used as a shortcut to start the 
        search for the coordinate. And a new position is saved for use in
        a later search. Only works for a one-deep search.

        Parameters
        ----------
        coords: list
        list of coordinates to traverse

        allocate: Boolean
        Automatically generate the default value if needed at each
        level of the tree, but don't insert into the tree.

        default: value
        A constant default value to return if coordinate is empty on
        no-allocate

        start_pos: scalar or Payload() containing a scalar
        An optional shortcut value to optimize search

        Returns
        -------
        payload: a scalar or Fiber

        Raises
        ------

        None

        """

        assert default is None or not allocate
        assert start_pos is None or len(coords) == 1

        start_pos = Payload.get(start_pos)

        # TBD: Actually optimize the search

        try:
            index = self.coords.index(coords[0])
            payload = self.payloads[index]
        except:
            #
            # The requested coordinate did not exist
            #
            # If we are allocating missing elements or
            # are not at the last coordinate in the given coord list
            # create a default value to return (or recurse into),
            # but do not change anything in the actual fiber.
            #
            # Otherwise return the provided default.
            #
            # Note: we record we found it at the final index
            #
            index = len(self.coords)

            if allocate or len(coords) > 1:
                payload = self._createDefault()
            else:
                payload = default


        if len(coords) > 1:
            assert isinstance(payload, Fiber), "getPayload too many coordinates"

            # Recurse to the next level's fiber
            return payload.getPayload(*coords[1:],
                                      default=default,
                                      allocate=allocate)

        if start_pos is not None:
            self.setSavedPos(index, distance=index-start_pos)

        return payload


    def getPayloadRef(self, *coords, start_pos=None):
        """payload

        Return the final payload after recursively traversing the
        levels of the fiber tree for at each coordinate in coords.
        If the payload is empty, then recursively return the default payload

        If "start_pos" is specified it is used as a shortcut to start the 
        search for the coordinate. And a new position is saved for use in
        a later search. Only works for a one-deep search.

        Parameters
        ----------
        coords: coordinates
        List of coordinates to traverse

        start_pos: scalar or Payload() containing a scalar
        Optional shortcut value to optimize search

        Returns
        -------
        payload: a scalar or Fiber

        Raises
        ------

        None

        """

        assert start_pos is None or len(coords) == 1

        # TBD: Actually optimize the search

        start_pos = Payload.get(start_pos)

        try:
            index = self.coords.index(coords[0])
            payload = self.payloads[index]
        except:
            # Coordinate didn't exist so create a payload
            payload = self._create_payload(coords[0])
            index = self.coords.index(coords[0])

        if len(coords) > 1:
            # Recurse to the next level's fiber
            assert isinstance(payload, Fiber), "Too many coordinates"

            return payload.getPayloadRef(*coords[1:])

        if start_pos is not None:
            self.setSavedPos(index, distance=index-start_pos)

        return payload


    def _create_payload(self, coord):
        """Create a payload in the fiber at coord

        Optionally insert into the owners rank.

        """

        # Create a payload at coord
        # Iemporary value (should be None)

        value = self._createDefault()

        payload = self._maybe_box(value)

        try:
            index = next(x for x, val in enumerate(self.coords) if val > coord)
            self.coords.insert(index, coord)
            self.payloads.insert(index, payload)
        except StopIteration:
            index = len(self.coords)
            self.coords.append(coord)
            self.payloads.append(payload)


        #
        # Get the payload out of the payloads array
        # TBD: Not sure why I felt this was needed
        #
        payload = self.payloads[index]

        #
        # If the payload we created is a fiber,
        # then insert it into the its proper rank
        #
        if Payload.contains(value, Fiber):
            assert(not self._owner is None)
            next_rank = self._owner.getNextRank()
            if not next_rank is None:
                next_rank.append(payload)

        return payload


    def getRange(self, start_coord, size, trans_fn=None, start_pos=None):
        """getRange"""

        if trans_fn is None:
            # Default trans_fn is identify function (inefficient but easy implementation)
            trans_fn = lambda x: x

        start_pos = Payload.get(start_pos)

        if start_pos is not None:
            assert start_pos < len(self.coords)
            assert start_pos == 0 or self.coords[start_pos-1] < start_coord

            range_start = start_pos
        else:
            range_start = 0

        # Invariant: trans_fn is order preserving, but we check for reversals

        min = start_coord
        max = start_coord + size - 1

        coords = []
        payloads = []

        # Start at start_pos (if any)

        first_pos = None

        for pos in range(range_start, len(self.coords)):
            c = self.coords[pos]
            p = self.payloads[pos]
            new_c = trans_fn(c)
            if new_c > max:
                break
            if new_c >= min:

                # For statistics
                if first_pos is None:
                    first_pos = pos

                coords.append(new_c)
                payloads.append(p)


        # Note: This reversal implies a complex read order

        if len(coords) > 1 and coords[1] < coords[0]:
            coords.reverse()
            payloads.reverse()

        if start_pos is not None:
            if first_pos is None:
                self.setSavedPos(pos)
            else:
                self.setSavedPos(pos, distance=first_pos-start_pos)


        return Fiber(coords, payloads)


    def prune(self, trans_fn=None, start_pos=None):
        """prune"""

        start_pos = Payload.get(start_pos)

        if start_pos is not None:
            assert start_pos < len(self.coords)
            range_start = start_pos
        else:
            range_start = 0

        coords = []
        payloads = []

        # Start at start_pos (if any)

        first_pos = None

        for pos in range(range_start, len(self.coords)):
            c = self.coords[pos]
            p = self.payloads[pos]

            status = trans_fn(pos, c, p)

            if status is None: break

            if status:
                coords.append(c)
                payloads.append(p)


#        if start_pos is not None:
#            if first_pos is None:
#                self.setSavedPos(pos)
#            else:
#                self.setSavedPos(pos, distance=first_pos-start_pos)

        return Fiber(coords, payloads)


    def getPosition(self, coord, start_pos=None):
        """getPosition

        Return the position of the coord, if any. [Non-mutating]

        If "start_pos" is specified it is used as a shortcut to start the
        search for the coordinate. And a new position is saved for use in
        a later search. Only works for a one-deep search.

        Parameters
        ----------
        coord: coordinate
        Coordinate to look up

        start_pos: scalar or Payload() containing a scalar
        Optional shortcut value to optimize search

        Returns
        -------
        position: an index that can be used to _getitem_(), or None

        Raises
        ------

        None

        """

        # TBD: Actually optimize the search

        start_pos = Payload.get(start_pos)

        try:
            index = self.coords.index(coord)
        except:
            index = None

        if start_pos is not None and index is not None:
            self.setSavedPos(index, distance=index-start_pos)

        return index


    def getPositionRef(self, coord, start_pos=None):
        """getPositionRef

        Return the position of the given coordinate. If the coordinate
        has no payload, then create it and assign the appropriatge
        default payload (either an empty Fiber or zero). [Mutating]

        If "start_pos" is specified it is used as a shortcut to start the
        search for the coordinate. And a new position is saved for use in
        a later search. Only works for a one-deep search.


        Parameters
        ----------
        coord: coordinate
        Coordinate to look up

        start_pos: scalar or Payload() containing a scalar
        Optional shortcut value to optimize search

        Returns
        -------
        position: an index that can be used by _getitem_()

        Raises
        ------

        None

        """

        start_pos = Payload.get(start_pos)

        # TBD: Actually optimize the search

        try:
            index = self.coords.index(coord)
        except:
            self._create_payload(coord)
            index = len(self.payloads)-1 # TODO: This is wrong...

        if start_pos is not None and index is not None:
            self.setSavedPos(index, distance=index-start_pos)

        return index


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

#
# Deprecated coordinate-based methods
#

    def insertOrLookup(self, coord, value=None):
        """insertOrLookup"""

        Fiber._deprecated("Fiber.insertOrLookup() is deprecated use getPayloadRef()")

        if value is None:
            value = self._createDefault()

        payload = self._maybe_box(value)

        index = 0
        try:
            index = next(x for x, val in enumerate(self.coords) if val >= coord)
            if self.coords[index] == coord:
                return self.payloads[index]
            self.coords.insert(index, coord)
            self.payloads.insert(index, payload)
            return self.payloads[index]
        except StopIteration:
            self.coords.append(coord)
            self.payloads.append(payload)
            return self.payloads[-1]


    def insert(self, coord, value):
        """insert"""

        Fiber._deprecated("Fiber.insert() is deprecated use getPayloadRef()")

        payload = self._maybe_box(value)

        try:
            index = next(x for x, val in enumerate(self.coords) if val > coord)
            self.coords.insert(index, coord)
            self.payloads.insert(index, payload)
        except StopIteration:
            self.coords.append(coord)
            self.payloads.append(payload)

        return None


    #
    # Owner rank related methods
    #
    def setOwner(self, owner):
        """setOwner"""

        self._owner = owner

    def getOwner(self):
        """getOwner"""

        return self._owner


    #
    # Default payload methods
    #
    def setDefault(self, default):
        """setDefault"""

        Fiber._deprecated("Fiber.setDefault() default values should be set by the owning rank")

        self._default = default


    def getDefault(self):
        """getDefault

        Get the default payload for this fiber. Ideally from the owner rank

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

        #
        # Try to get default from owning rank
        #
        owner = self.getOwner()

        if owner is not None:
            return owner.getDefault()

        #
        # For unowned fibers, try to guess a default value
        #
        if len(self.payloads) > 0 and isinstance(self.payloads[0], Fiber):
            return Fiber

        if self._default != 0:
            Fiber._deprecated("Fiber.getDefault() used a fiber-level default values")

            return self._default

        return 0



    def _createDefault(self):
        """_createDefault"""

        default = self.getDefault()

        if callable(default):
            value = default()

            #
            # Conditionaly set the owning rank
            #
            # TBD: This is a messy interaction with rank
            #      See Rank.append()
            #
            if isinstance(value, Fiber):
                owner = self.getOwner()
                if owner:
                    value.setOwner(owner.next_rank)
        else:
            value = default

        return value


    #
    # Saved position shortcut related methods
    #
    def setSavedPos(self, position, distance=None):
        """setSavedPos"""

        position = Payload.get(position)

        self._saved_pos = position

        #
        # Optionally save distanced moved statistics
        #
        if distance is not None:
            self._saved_count += 1
            self._saved_dist += abs(distance)


    def getSavedPos(self):
        """getSavedPos"""

        return self._saved_pos


    def getSavedPosStats(self, clear=True):
        """getSavedDistance"""

        stats = (self._saved_count, self._saved_dist)

        if clear:
            self._clearSavedPosStats()

        return stats


    def _clearSavedPosStats(self):
        """_clearSavedPosStats"""

        self._saved_count = 0
        self._saved_dist = 0

    #
    # Computed attribute acccessors
    #
    def minCoord(self):
        """min_coord

        Return the minimum coordinate that exists in the fiber

        """

        # TBD: Should check that the candidate is not an explicit zero

        if len(self.coords) == 0:
            return None

        return min(self.coords)

    def maxCoord(self):
        """max_coord

        Return the maximum coordinate that exists in the fiber

        """

        # TBD: Should check that the candidate is not an explicit zero

        if len(self.coords) == 0:
            return None

        if not isinstance(self.coords[0], int):
            #
            # Coordinates aren't integers, so maxCoord doesn't make sense
            #
            return None

        return max(self.coords)


    def countValues(self):
        """Count values in the fiber tree

        Note: an explcit zero scalar value will NOT count as a value
        """

        count = 0
        for p in self.payloads:
            if Payload.contains(p, Fiber):
                count += Payload.get(p).countValues()
            else:
                count += 1 if not Payload.isEmpty(p) else 0

        return count

#
# Position based methods
#
    def __getitem__(self, keys):
        """__getitem__

        For an integer key return a (coordinate, payload) tuple
        containing the contents of a fiber at "position", i.e., an
        offset in the coordinate and payload arrays. For a slice key
        return a new fiber for the slice

        Parameters
        ----------
        keys: single integer/slicr or tuple of integers/slices
        The positions or slices in an n-D fiber

        Returns
        -------
        tuple or Fiber
        A tuple of a coordinate and payload or a Fiber of the slice

        Raises
        ------

        IndexError
        Index out of range

        TypeError
        Invalid key type

        """

        if not isinstance(keys, tuple):
            # Keys is a single value for 1-D access
            key = keys
            key_cdr = ()
        else:
            # Keys is a tuple for for n-D access
            key = keys[0]
            key_cdr = keys[1:]

        if isinstance(key, int):
            # Handle key as single index

            if key < 0:
                #Handle negative indices
                key += len(self)

            if key < 0 or key >= len(self):
                   raise(IndexError,
                         f"The index ({key}) is out of range")

            new_payload = self.payloads[key]

            if len(key_cdr):
                # Recurse down the fiber tree
                new_payload = new_payload[key_cdr]

            return CoordPayload(self.coords[key], new_payload)

        if isinstance(key, slice) :
            # Key is a slice

            #Get the start, stop, and step from the slice
            slice_range = range(*key.indices(len(self)))

            coords = [self.coords[ii] for ii in slice_range]

            if len(key_cdr):
                # Recurse down the fiber tree for each payload in slice
                payloads = [self.payloads[ii][key_cdr] for ii in slice_range]
            else:
                # Just use each payload in slice
                payloads = [self.payloads[ii] for ii in slice_range]

            return Fiber(coords, payloads)

        raise(TypeError, "Invalid key type.")


    def __setitem__(self, key, newvalue):
        """__setitem__

        The "newvalue" parameter is either a CoordPayload an arbitrary
        value to assign to the position "key" in the fiber.  If
        "newvalue" is not a CoordPayload or the Coord in the
        CoordPayload is None the current coordinate will be left
        unchanged. The payload will be boxed if appropriate. If the
        payload is None, then the payload will be left unchanged.

        Parameters
        ----------
        key: single integer
        The position in the fiber to be set

        newvalue: a CoordPayload or a payload value
        The coordinate/payload or just payload to assign

        Returns
        -------
        Nothing

        Raises
        ------

        IndexError
        Index out of range

        TypeError
        Invalid key type

        CoordinateError
        Invalid coordinate

        """

        position = key

        #
        # TBD: Get isinstance of CoordPayload to work...
        #
        try:
            coord = newvalue.coord
            payload = newvalue.payload
        except:
            coord = None
            payload = newvalue

        if coord is not None:
            #
            # Check that coordinate order is maintained
            #
            if position > 0 and coord <= self.coords[position-1]:
                raise CoordinateError

            if position+1 < len(self.coords) and coord >= self.coords[position+1]:
                raise CoordinateError

            self.coords[position] = coord

        #
        # A payload of None just updates the coordinate
        #
        if payload is not None:
            self.payloads[position] = self._maybe_box(payload)


    def __len__(self):
        """__len__"""

        return len(self.coords)


    def isEmpty(self):
        """isEmpty() - check if Fiber is empty

        Empty is defined as of zero length, only containing zeros
        or only containing subfibers that are empty.
        """

        return all(map(Payload.isEmpty, self.payloads))


    def nonEmpty(self):
        """nonEmpty() - return Fiber only with non-empty elements

        Because our fiber representation might have explicit zeros
        in it this method creates a new fiber with those elements
        pruned out.

        """
        coords = []
        payloads = []

        for c, p in zip(self.coords, self.payloads):
            if not Payload.isEmpty(p):
                coords.append(c)
                if Payload.contains(p, Fiber):
                    payloads.append(p.nonEmpty())
                else:
                    payloads.append(p)

        return Fiber(coords, payloads)

#
# Iterator methods
#

    def __iter__(self):
        """__iter__"""

        for i in range(len(self.coords)):
            yield CoordPayload(self.coords[i], self.payloads[i])

    def __reversed__(self):
        """Return reversed fiber"""

        for coord, payload in zip(reversed(self.coords), reversed(self.payloads)):
            yield CoordPayload(coord, payload)


#
# Core methods
#

    def clear(self):
        self.coords.clear()
        self.payloads.clear()


    def payload(self, coord):
        """payload"""

        Fiber._deprecated("Fiber.payload() is deprecated use getPayload()")

        return self.getPayload(coord)


    def append(self, coord, value):
        """append - Add element at end of fiber"""

        assert self.maxCoord() is None or self.maxCoord() < coord, \
               "Fiber coordinates must be monotonically increasing: {}, {}".format(self.maxCoord(), coord)

        payload = self._maybe_box(value)

        self.coords.append(coord)
        self.payloads.append(payload)


    def extend(self, other):
        """extend - Extend a fiber with another fiber"""

        assert isinstance(other, Fiber), \
               "Fibers can only be extended with another fiber"

        if other.isEmpty():
            # Extending with an empty fiber is a nop
            return None

        assert self.maxCoord() is None or self.maxCoord() < other.coords[0], \
               "Fiber coordinates must be monotonically increasing"

        self.coords.extend(other.coords)
        self.payloads.extend(other.payloads)

        return None


    def updateCoords(self, func, depth=0):
        """updateCoords

        Update each coordinate in the the fibers at a depth of "depth"
        below "self" by invoking "func" on it.  Therefore, a depth of
        zero will update the coordinates in the current fiber. Higher
        depths with result in a depth first search down to "depth"
        before traversing the coordinates.

        Note: Function currently does not check that coordinates remain
              monotonically increasing.

        Parameters
        ----------

        func: function
        A function that is invoked with each coordinate as its argument

        depth: integer
        The depth in the fiber tree to dive before traversing

        Returns
        --------

        None

        Raises
        ------

        TBD: currently nothing

        """
        if depth > 0:
            # Recurse down to depth...
            for p in self.payloads:
                p.updateCoords(func, depth=depth-1)
        else:
            # Update my coordinates
            for i in range(len(self.coords)):
                self.coords[i] = func(i, self.coords[i], self.payloads[i])

        return None


    def updatePayloads(self, func, depth=0):
        """updatePayloads

        Update each payload in the the fibers at a depth of "depth"
        below "self" by invoking "func" on it.  Therefore, a depth of
        zero will update the payloads in the current fiber. Higher
        depths with result in a depth first search down to "depth"
        before traversing the payloads.

        Parameters
        ----------

        func: function
        A function that is invoked with each payload as its argument

        depth: integer
        The depth in the fiber tree to dive before traversing

        Returns
        --------

        None

        Raises
        ------

        TBD: currently nothing

        """
        if depth > 0:
            # Recurse down to depth...
            for p in self.payloads:
                p.updatePayloads(func, depth=depth-1)
        else:
            # Update my payloads
            for i in range(len(self.payloads)):
                self.payloads[i] = func(self.payloads[i])

        return None


    def unzip(self):
        """Unzip

        Unzip a fiber whose payloads are a tuple into two fibers each
        with the same coordinates

        """

        coords_a = list(self.coords)
        coords_b = list(self.coords)

        (payloads_a, payloads_b) = zip(*self.payloads)

        return (Fiber(coords_a, payloads_a), Fiber(coords_b, payloads_b))

#
# Shape-related methods
#

    def getShape(self, all_ranks=True):
        """Return shape of fiber tree"""

        owner = self.getOwner()

        shape = None

        if owner is not None:
            shape = owner.getShape(all_ranks=all_ranks)

        if shape is None or any([s == 0 for s in shape]):
            # Backup for cases where there is no  owner
            # or owner didn't know shape

            shape = self.estimateShape()

            if not all_ranks:
                shape = shape[0:1]

        return shape


    def estimateShape(self):
        """estimateShape

        Traverse a fiber tree to estimate its shape

        """

        return self._calcShape()


    def _calcShape(self, shape=None, level=0):
        """ _calcShape()

        Find the maximum coordinate at each level of the tree

        TBD: Using maximum coordinate isn't really right because
             the original array may have a empty value at its
             maximum coordinate location

        """

        #
        # Start recursion
        #
        if shape is None:
            shape = []

        #
        # Conditionaly append a new level to the shape array
        #
        if len(shape) < level+1:
            shape.append(0)

        #
        # If fiber is empty then shape doesn't change
        #
        if not len(self.coords):
            return shape

        #
        # Try to determine the maximum coordinate
        #
        if self.max_legal_coord is not None:
            max_coord = self.max_legal_coord
        else:
            max_coord = self.maxCoord()

        #
        # If fiber is not empty, but max_coord isn't meaningful,
        # assume fiber is dense and return count of elements
        #
        if max_coord is None:
            max_coord = len(self.coords)

        #
        # Update shape for this Fiber at this level
        #
        shape[level] = max(shape[level], max_coord+1)

        #
        # Recursively process payloads that are Fibers
        #
        if Payload.contains(self.payloads[0], Fiber):
            for p in self.payloads:
                Payload.get(p)._calcShape(shape, level+1)

        return shape


    def uncompress(self, shape=None, level=0):
        """Return an uncompressed fiber tree (i.e., a nest of lists)"""

        if shape is None:
            shape = self.getShape(all_ranks=True)

        f = [ ]

        for c, (mask, p, _) in self | Fiber(coords=range(shape[level]), initial=1):

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
# Arithmetic operations
#


    def __ilshift__(self, other):
        """__ilshift__

        This operator will make a recursive assignment of all the
        elements of one fiber ("other") into another fiber ("self")
        using getPayloadRef(), so subfibers in new fiber are
        properly inserted into their owning rank/tensor

        Note: we use <<= in place of base '=' since we don't want a
        pointer to the existing fiber but an copy of "other" in the
        new fiber.

        Parameters
        ----------

        other: Fiber
        A fiber whose elements will be inserted into "self"

        """

        assert isinstance(other, Fiber)

        if len(self.coords) != 0:
            #
            # Clear out any existing data
            #
            self.coords = []
            self.payloads = []


        for c, p in other:
            #
            # For each non-empty element of other, insert it into the
            # target, note that this works regardless of whether p is
            # a Fiber or a Payload
            #
            if Payload.isEmpty(p):
                continue
            
            ref = self.getPayloadRef(c)
            ref <<= p


#
# Split methods
#
# Note: all these methods return a new fiber
#


    def splitUniform(self, step, partitions=1, relativeCoords=False):
        """splitUniform"""

        class _SplitterUniform():

            def __init__(self, step):
                self.step = step
                self.cur_group = 0

            def nextGroup(self, i, c):
                count = 0
                last_group = self.cur_group

                while c >= self.cur_group:
                    count += 1
                    last_group = self.cur_group
                    self.cur_group += self.step

                return count, last_group

        splitter = _SplitterUniform(step)

        return self._splitGeneric(splitter, partitions, relativeCoords=relativeCoords)


    def splitNonUniform(self, splits, partitions=1, relativeCoords=False):
        """splitNonUniform"""

        class _SplitterNonUniform():

            def __init__(self, splits):
                if isinstance(splits, Fiber):
                    self.splits = splits.coords.copy()
                else:
                    self.splits = splits.copy()

                self.cur_split = self.splits.pop(0)

            def nextGroup(self, i, c):
                count = 0
                last_group = self.cur_split

                while c >= self.cur_split:
                    count += 1
                    last_group = self.cur_split
                    if self.splits:
                        self.cur_split = self.splits.pop(0)
                    else:
                        self.cur_split = float("inf")

                return count, last_group

        splitter = _SplitterNonUniform(splits)

        return self._splitGeneric(splitter, partitions, relativeCoords=relativeCoords)


    def splitEqual(self, step, partitions=1, relativeCoords=False):
        """splitEqual"""

        class _SplitterEqual():

            def __init__(self, step):
                self.step = step
                self.cur_count=0

            def nextGroup(self, i, c):
                count = 0

                while i >= self.cur_count:
                    count += 1
                    self.cur_count += self.step

                return count, c

        splitter = _SplitterEqual(step)

        return self._splitGeneric(splitter, partitions, relativeCoords=relativeCoords)


    def splitUnEqual(self, sizes, partitions=1, relativeCoords=False):
        """splitUnEqual

        Split root fiber by the sizes in "sizes".

        If there are more coordinates than the sum of the "sizes" all
        remaining coordinates are put into the final split.

        """

        class _SplitterUnEqual():

            def __init__(self, sizes):
                self.sizes = sizes.copy()
                self.cur_count = -1

            def nextGroup(self, i, c):
                count = 0

                while i > self.cur_count:
                    count += 1
                    if self.sizes:
                        self.cur_count += self.sizes.pop(0)
                    else:
                        self.cur_count = float("inf")

                return count, c

        splitter = _SplitterUnEqual(sizes)

        return self._splitGeneric(splitter, partitions, relativeCoords=relativeCoords)


    def _splitGeneric(self, splitter, partitions, relativeCoords):
        """_splitGeneric

        Takes the current fiber and splits it according to the boundaries defined by splitter().
        The result is a new rank (for paritions = 1) or two new ranks (for partitions > 1).

        rank2 - uppermost rank with one coordinate per partition (only exists for partitions > 1)
        rank1 - middle rank with one coordinate per split
        rank0 - lowest rank with fibers split out from the single original fiber

        """

        rank0_fiber_group = []
        rank0_fiber_coords = []
        rank0_fiber_payloads = []

        rank1_fiber_coords = []
        rank1_fiber_payloads = []

        # Create arrays for rank1 fibers per partition

        for i in range(partitions):
            rank1_fiber_coords.append([])
            rank1_fiber_payloads.append([])

        cur_coords = None
        rank1_count = -1

        # Split apart the fiber into groups according to "splitter"

        for i0, (c0,p0) in enumerate(zip(self.coords,self.payloads)):
            # Check if we need to start a new rank0 fiber
            count,next_rank1_coord = splitter.nextGroup(i0, c0)
            if (count > 0):
                rank1_count += count

                # Old style: upper rank's coordinates were a dense range
                # rank1_coord = rank1_count

                # New style: upper rank's coordinates are first coordinate of group
                rank1_coord = next_rank1_coord
                rank0_offset = rank1_coord

                rank0_fiber_group.append(rank1_coord)

                cur_coords = []
                rank0_fiber_coords.append(cur_coords)

                cur_payloads = []
                rank0_fiber_payloads.append(cur_payloads)

            # May not be in a group yet
            if not cur_coords is None:
                if relativeCoords:
                    cur_coords.append(c0-rank0_offset)
                else:
                    cur_coords.append(c0)

                cur_payloads.append(p0)


        # Deal the split fibers out to the partitions

        partition =  0

        for c1, c0, p0 in zip(rank0_fiber_group, rank0_fiber_coords, rank0_fiber_payloads):
            rank1_fiber_coords[partition].append(c1)
            rank1_fiber_payloads[partition].append(Fiber(c0, p0))
            partition = (partition + 1) % partitions

        # For 1 partition don't return a extra level of Fiber

        if partitions == 1:
            return Fiber(rank1_fiber_coords[0], rank1_fiber_payloads[0])

        # For >1 partitions return a Fiber with a payload for each partition

        payloads = []

        for c1, p1 in zip(rank1_fiber_coords, rank1_fiber_payloads):
            payload = Fiber(c1, p1)
            payloads.append(payload)

        return Fiber(payloads=payloads)

#
# Operation methods
#

    def __add__(self, other):
        """__add__"""

        assert isinstance(other, Fiber), \
               "Fiber addition must involve two fibers"

        return Fiber(coords = self.coords+other.coords,
                     payloads = self.payloads+other.payloads)


    def __iadd__(self, other):
        """__iadd__"""

        self.extend(other)

        return self

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

            return CoordPayload(coord, payload)


        def get_next_nonempty(iter):
            """get_next_nonempty"""

            (coord, payload) = get_next(iter)

            while Payload.isEmpty(payload):
                (coord, payload) = get_next(iter)

            return CoordPayload(coord, payload)

        a = self.__iter__()
        b = other.__iter__()

        z_coords = []
        z_payloads = []

        a_coord, a_payload = get_next_nonempty(a)
        b_coord, b_payload = get_next_nonempty(b)

        while not (a_coord is None or b_coord is None):
            if a_coord == b_coord:
                z_coords.append(a_coord)
                z_payloads.append((a_payload, b_payload))

                a_coord, a_payload = get_next_nonempty(a)
                b_coord, b_payload = get_next_nonempty(b)
                continue

            if a_coord < b_coord:
                a_coord, a_payload = get_next_nonempty(a)
                continue

            if a_coord > b_coord:
                b_coord, b_payload = get_next_nonempty(b)
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

            return CoordPayload(coord, payload)

        def get_next_nonempty(iter):
            """get_next_nonempty"""

            (coord, payload) = get_next(iter)

            while Payload.isEmpty(payload):
                (coord, payload) = get_next(iter)

            return CoordPayload(coord, payload)

        a_fiber = self
        b_fiber = other

        a = self.__iter__()
        b = other.__iter__()

        z_coords = []
        z_payloads = []

        a_coord, a_payload = get_next_nonempty(a)
        b_coord, b_payload = get_next_nonempty(b)

        while not (a_coord is None or b_coord is None):
            if a_coord == b_coord:
                z_coords.append(a_coord)

                z_payloads.append(("AB", a_payload, b_payload))

                a_coord, a_payload = get_next_nonempty(a)
                b_coord, b_payload = get_next_nonempty(b)
                continue

            if a_coord < b_coord:
                z_coords.append(a_coord)

                b_default = b_fiber._createDefault()
                z_payloads.append(("A", a_payload, b_default))

                a_coord, a_payload = get_next_nonempty(a)
                continue

            if a_coord > b_coord:
                z_coords.append(b_coord)

                a_default = a_fiber._createDefault()
                z_payloads.append(("B", a_default, b_payload))

                b_coord, b_payload = get_next_nonempty(b)
                continue

        while not a_coord is None:
            z_coords.append(a_coord)

            b_default = b_fiber._createDefault()
            z_payloads.append(("A", a_payload, b_default))

            a_coord, a_payload = get_next_nonempty(a)

        while  not b_coord is None:
            z_coords.append(b_coord)

            a_default = a_fiber._createDefault()
            z_payloads.append(("B", a_default, b_payload))

            b_coord, b_payload = get_next_nonempty(b)

        return Fiber(z_coords, z_payloads)


    def __xor__(self, other):
        """__xor__

        Return the xor of "self" and "other" by considering all possible
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
        present in  |         True          |        False          |
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
            return CoordPayload(coord, payload)

        def get_next_nonempty(iter):
            """get_next_nonempty"""

            (coord, payload) = get_next(iter)

            while Payload.isEmpty(payload):
                (coord, payload) = get_next(iter)

            return CoordPayload(coord, payload)

        a_fiber = self
        b_fiber = other

        a = self.__iter__()
        b = other.__iter__()

        z_coords = []
        z_payloads = []

        a_coord, a_payload = get_next_nonempty(a)
        b_coord, b_payload = get_next_nonempty(b)

        while not (a_coord is None or b_coord is None):
            if a_coord == b_coord:
                a_coord, a_payload = get_next_nonempty(a)
                b_coord, b_payload = get_next_nonempty(b)
                continue

            if a_coord < b_coord:
                z_coords.append(a_coord)

                b_default = b_fiber._createDefault()
                z_payloads.append(("A", a_payload, b_default))

                a_coord, a_payload = get_next_nonempty(a)
                continue

            if a_coord > b_coord:
                z_coords.append(b_coord)

                a_default = a_fiber._createDefault()
                z_payloads.append(("B", a_default, b_payload))

                b_coord, b_payload = get_next_nonempty(b)
                continue

        while not a_coord is None:
            z_coords.append(a_coord)

            b_default = b_fiber._createDefault()
            z_payloads.append(("A", a_payload, b_default))

            a_coord, a_payload = get_next_nonempty(a)

        while  not b_coord is None:
            z_coords.append(b_coord)

            a_default = a_fiber._createDefault()
            z_payloads.append(("B", a_default, b_payload))

            b_coord, b_payload = get_next_nonempty(b)

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


        Note: an explcit zero in the input will NOT generate a corresponding
              coordinate in the output!

        """


        def get_next(iter):
            """get_next"""

            try:
                coord, payload = next(iter)
            except StopIteration:
                return (None, None)
            return CoordPayload(coord, payload)

        def get_next_nonempty(iter):
            """get_next_nonempty"""

            (coord, payload) = get_next(iter)

            while Payload.isEmpty(payload):
                (coord, payload) = get_next(iter)

            return CoordPayload(coord, payload)


        # "a" is self!
        b = other.__iter__()

        z_coords = []
        z_payloads = []

        b_coord, b_payload = get_next_nonempty(b)

        while not b_coord is None:
            z_coords.append(b_coord)

            # TBD: Optimize with co-iteration...

            a_payload = self.getPayload(b_coord, allocate=False)
            if a_payload is None:
                a_payload = self._create_payload(b_coord)

            z_payloads.append((a_payload, b_payload))
            b_coord, b_payload = get_next_nonempty(b)

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
            return CoordPayload(coord, payload)

        def get_next_nonempty(iter):
            """get_next_nonempty"""

            (coord, payload) = get_next(iter)

            while Payload.isEmpty(payload):
                (coord, payload) = get_next(iter)

            return CoordPayload(coord, payload)

        a = self.__iter__()
        b = other.__iter__()

        z_coords = []
        z_payloads = []

        a_coord, a_payload = get_next(a)
        b_coord, b_payload = get_next_nonempty(b)

        while not (a_coord is None or b_coord is None):
            if a_coord == b_coord:
                a_coord, a_payload = get_next(a)
                b_coord, b_payload = get_next_nonempty(b)
                continue

            if a_coord < b_coord:
                z_coords.append(a_coord)
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
# Note: all these methods return a new fiber
#
    def swapRanks(self):
        """Swap the (highest) two ranks of the fiber.

        This function relies on flattenRanks() and unflattenRanks().
        FIXME: flattenRanks() could be more general to support all p1 types,
        including tuples.

        """

        #
        # Flatten the (highest) two ranks
        #
        flattened = self.flattenRanks(nested_coords=True)

        # Make sure the coord is a >=2-element tuple
        assert(len(flattened.coords[0]) >= 2)

        #
        # Sort on secord coordinate of flattened fiber
        # and create new sorted fiber
        #
        sorted_cp = sorted([ (c[::-1], p) for c, p in flattened ])

        coords = [c for c, _ in sorted_cp]
        payloads = [p for _, p in sorted_cp]

        flattened_sorted = Fiber(coords, payloads)

        #
        # Unflatten to get original coordinates in swapped ranks
        #
        swapped = flattened_sorted.unflattenRanks()

        return swapped


    def flattenRanks(self, levels=1, nested_coords=False):
        """Flatten two ranks into one - COO-style"""

        #
        # Flatten deeper levels first, if requested
        #
        if levels == 1:
            cur_payloads = self.payloads
        else:
            assert (isinstance(self.payloads[0], Fiber)), \
                   "Insuffient levels to flatten"

            cur_payloads = []

            for p in self.payloads:
                cur_payloads.append(p.flattenRanks(levels=levels-1))

        #
        # Flatten this level
        #
        coords = []
        payloads = []

        for c1, p1 in zip(self.coords, cur_payloads):

            if Payload.contains(p1, Fiber):
                for c0, p0 in p1:
                    coords.append(self._flattenCoords(c1,
                                                      c0,
                                                      nested_coords=nested_coords))
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
                    coords.append(self._flattenCoords(c1,
                                                      c0,
                                                      nested_coords=nested_cords))
                    payloads.append( (p0,) + p1[1:] )

        return Fiber(coords, payloads)

    @staticmethod
    def _flattenCoords(c1, c0, nested_coords=False):
        """_flattenCoords"""

        if not nested_coords:
            #
            # Combine coordinates into a single flat tuple, flattening
            # contents of the individual coordinates that are tuples
            #
            # Convert c1 to tuple, if necessary
            #
            if not isinstance(c1, tuple):
                c1 = (c1, )

            #
            # Convert c0 to tuple, if necessary
            #
            if not isinstance(c0, tuple):
                c0 = (c0,)

            #
            # Concatenate the two coordinates
            #a
            c1_c0 = c1 + c0
        else:
            #
            # Create a new coordinante as a two element tuple
            #
            c1_c0 = (c1, c0)

        return c1_c0


    def unflattenRanks(self, levels=1):
        """Unflatten two ranks into one"""

        assert(isinstance(self.coords[0], tuple))

        #
        # Place to collect cordinates/payloads for new top rank
        #
        coords1 = []
        payloads1 = []

        first = True

        #
        # Traverse the rank being unflattened in order to collect all
        # the coordinates/payloads to put in the final top (c1) rank
        #
        for cx, p0 in zip(self.coords, self.payloads):
            #
            # Little dance to get the coordinates of the two ranks,
            # which also deals with the case where the coordinates
            # were not nested.
            #
            c1 = cx[0]
            if len(cx) > 2:
                c0 = cx[1:]
            else:
                c0 = cx[1]

            #
            # Check if we finished all the elements of the lower (c0)
            # rank and have moved on to a new coordinate in the higher
            # (c1) rank. If so add the collected c0 coordinates and
            # payloads as a fiber to the c1 rank, and start a new
            # collection of coordinates and fibers.
            #
            if first or (c1 > c1_last):
                if not first:
                    #
                    # Except when starting to work on the first c1
                    # coordinate, add a new coordinate/payload pair
                    # (maybe after a recursive unflattening) to the
                    # new top (c1) rank
                    #
                    coords1.append(c1_last)

                    cur_fiber = Fiber(coords0, payloads0)
                    if levels > 1:
                        cur_fiber = cur_fiber.unflattenRanks(levels=levels-1)

                    payloads1.append(cur_fiber)

                #
                # Start working on a new c1 coordinate (c1_last) and
                # create a place to collect lower rank (c0)
                # coordinates/payloads to form the payload (fiber) of
                # that c1 coordinate.
                #
                first = False
                c1_last = c1
                coords0 = []
                payloads0 = []

            coords0.append(c0)
            payloads0.append(p0)

        #
        # Pick up the last element of the new top rank
        #
        coords1.append(c1_last)

        cur_fiber = Fiber(coords0, payloads0)
        if levels > 1:
            cur_fiber = cur_fiber.unflattenRanks(levels=levels-1)

        payloads1.append(cur_fiber)

        #
        # Create (and return) the new top (c1) rank
        #
        return Fiber(coords1, payloads1)

#
# Closures to operate on all payloads at a specified depth
#
# Note: all these methods mutate the fibers
#
# TBD: Reimpliment with Guowei's cleaner Python closure/wrapper
#

    def updatePayloadsBelow(self, func, *args, depth=0, **kwargs):
        """updatePayloadsBelow

        Utility function used as a closure on updatePayloads() to
        change all the payloads in fibers at "depth" in the tree by
        applying "func" with parameters *args and **kwargs to the
        payloads.

        """

        update_lambda = lambda p: func(p, *args, **kwargs)
        return self.updatePayloads(update_lambda, depth=depth)


    splitUniformBelow = partialmethod(updatePayloadsBelow,
                                      splitUniform)

    splitNonUniformBelow = partialmethod(updatePayloadsBelow,
                                         splitNonUniform)

    splitEqualBelow = partialmethod(updatePayloadsBelow,
                                    splitEqual)

    splitUnEqualBelow = partialmethod(updatePayloadsBelow,
                                      splitUnEqual)

    swapRanksBelow = partialmethod(updatePayloadsBelow,
                                      swapRanks)

    flattenRanksBelow = partialmethod(updatePayloadsBelow,
                                      flattenRanks)

    unflattenRanksBelow = partialmethod(updatePayloadsBelow,
                                        unflattenRanks)


#
#  Comparison operations
#

    def __eq__(self, other):
        """__eq__ - Equality check for Fibers

        Note: explict zeros do not result in inequality
        """

        for c, (mask, ps, po) in self | other:
            if mask == "A" and not Payload.isEmpty(ps):
                return False

            if mask == "B" and not Payload.isEmpty(po):
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

    def __format__(self, format):
        """__format__

        Format a fiber

        Spec:

        [(<coord spec>,<scalar spec>)][n][*]

        where:
                "n" means add newlines
                "*" means do not truncate with elipsis

        """
        import re

        kwargs = {}

        regex0 = '(\(.*,.*\))?(n)?(\*)?'
        match0 = re.search(regex0, format)
        group1 = match0.group(1)

        if group1 is not None:
            regex1 = '\((.*),(.*)\)'
            match1 = re.search(regex1, group1)
            kwargs['coord_fmt'] = match1.group(1)
            kwargs['payload_fmt'] = match1.group(2)

        if match0.group(2) == 'n':
            kwargs['newline'] = True

        if match0.group(3) == '*':
            kwargs['cutoff'] = 10000

        return self.__str__(**kwargs)


    def __str__(self,
                coord_fmt = "d",
                payload_fmt = "d",
                newline=False,
                cutoff=2,
                indent=0):
        """__str__"""

        def format_coord(coord):
            """Return "coord" properly formatted with "coord_fmt" """

            if not isinstance(coord, tuple):
                return f"{coord:{coord_fmt}}"

            return '(' + ', '.join(format_coord(c) for c in coord) + ')'


        def cond_string(string):
            """Return "string" if newline is True"""

            if newline:
                return string

            return ''

        str = ''

        if self._owner is None:
            str += "F/["
        else:
            str += f"F({self._owner.getName()})/["

        coord_indent = 0
        next_indent = 0
        items = len(self.coords)

        if self.payloads and isinstance(self.payloads[0], Fiber):

            for (c, p) in zip(self.coords[0:cutoff], self.payloads[0:cutoff]):
                if coord_indent == 0:
                    coord_indent = indent + len(str)
                    str += f"( {format_coord(c)} -> "
                    if newline:
                        next_indent = indent + len(str)
                else:
                    str += cond_string('\n' + coord_indent* ' ')
                    str += f"( {format_coord(c)} -> "

                str += p.__str__(coord_fmt=coord_fmt,
                                 payload_fmt=payload_fmt,
                                 newline=newline,
                                 cutoff=cutoff,
                                 indent=next_indent)
                str += ')'

            if items > cutoff:
                str += cond_string('\n')
                str += next_indent*' ' + "..."
                str += cond_string('\n')
                str += next_indent*' ' + "..."

            return str

        if newline:
            next_indent = indent + len(str)

        for i in range(min(items, cutoff)):
            if coord_indent != 0:
                str += cond_string('\n')

            str += cond_string(coord_indent*' ')
            str += f"({format_coord(self.coords[i])} -> "
            str += f"{self.payloads[i]:{payload_fmt}}) "
            coord_indent = next_indent

        if items > cutoff:
            str += cond_string('\n'+next_indent*' ')
            str += " ... "
            str += cond_string('\n'+next_indent*' ')
            str += " ... "

        str += "]"
        return str

    def __repr__(self):
        """__repr__"""

        # TBD: Owner is not properly reflected in representation

        str = f"Fiber({self.coords!r}, {self.payloads!r}"

        if self._owner:
            str += f", owner={self._owner.getName()}"

        str += ")"

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

#
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
                'payloads' : [ Payload.payload2dict(p) for p in self.payloads ]
              }
        }

        return f

#
# Utility functions
#

    def _maybe_box(self, value):
        """_maybe_box"""

        if isinstance(value, (bool, float, int, str, tuple, frozenset)):
            return Payload(value)

        return value


    @staticmethod
    def _deprecated(message):
        import warnings

        warnings.warn(message, FutureWarning, stacklevel=3)




if __name__ == "__main__":

    a = Fiber([2, 4, 6], [3, 5, 7])

    print("Simple print")
    a.print()
    print("----\n\n")

