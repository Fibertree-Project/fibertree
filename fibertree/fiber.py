"""Fiber"""

from fibertree.payload import Payload

class Fiber:
    """Fiber class"""

    def __init__(self, coords=None, payloads=None, default=0):
        """__init__"""

        if coords is None:
            coords = []
        if payloads is None:
            payloads = []

        assert (len(coords) == len(payloads)), "Coordinates and payloads must be same length"

        self.coords = coords
        self.payloads = [self._maybe_box(p) for p in payloads]

        # Owner rank... set on append to rank
        self.owner = None

        # Default value assigned to new coordinates
        self.set_default(default)

#
# Accessor methods
#
    def set_default(self, default):
        """set_default"""

        self.default = default

    def set_owner(self, owner):
        """set_owner"""

        self.owner = owner

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

#
#  Merge operatons
#
    def __and__(self, other):
        """__and__"""

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
        """__or__"""

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
        """__lshift__"""

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

        print("FIBER comparison")

        if (self.coords != other.coords):
            print("Coords different")
            return False

        if (self.payloads != other.payloads)
            print("Payloads different")
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
            str = "F(%s)/[" % self.owner.get_name()

        for i in range(len(self.coords)):
            str += "(%s -> %s) " % (self.coords[i], self.payloads[i])

        str += "]"
        return str

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

