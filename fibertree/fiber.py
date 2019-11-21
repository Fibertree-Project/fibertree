from fibertree.payload import Payload

class Fiber:

    def __init__(self, coords=[], payloads=[], default=0):
        assert (len(coords) == len(payloads)),"Coordinates and payloads must be same length"
        
        self.coords = coords
        self.payloads = [self._maybe_box(p) for p in payloads]
        self.default = default

    def __iter__(self):
        for i in range(len(self.coords)):
            yield (self.coords[i], self.payloads[i])


    def payload(self, coord):
        try:
            index = self.coords.index(coord)
            return self.payloads[index]
        except:
            return None

    def insert(self, coord, value):
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

        def get_next(iter):
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

        while ( not (a_coord is None or b_coord is None)):
            if (a_coord == b_coord):
                z_coords.append(a_coord)
                z_payloads.append((a_payload, b_payload))

                a_coord, a_payload = get_next(a)
                b_coord, b_payload = get_next(b)
                continue
            
            if (a_coord < b_coord):
                a_coord, a_payload = get_next(a)
                continue

            if (a_coord > b_coord):
                b_coord, b_payload = get_next(b)
                continue

        return Fiber(z_coords, z_payloads)


    def __or__(self, other):

        def get_next(iter):
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

        while ( not (a_coord is None or b_coord is None)):
            if (a_coord == b_coord):
                z_coords.append(a_coord)
                z_payloads.append(("AB", a_payload, b_payload))

                a_coord, a_payload = get_next(a)
                b_coord, b_payload = get_next(b)
                continue
            
            if (a_coord < b_coord):
                z_coords.append(a_coord)
                z_payloads.append(("A", a_payload, 0))

                a_coord, a_payload = get_next(a)
                continue

            if (a_coord > b_coord):
                z_coords.append(b_coord)
                z_payloads.append(("B", 0, b_payload))

                b_coord, b_payload = get_next(b)
                continue

        while  not (a_coord is None):
            z_coords.append(a_coord)
            z_payloads.append(("A", a_payload, 0))

            a_coord, a_payload = get_next(a)

        while  not (b_coord is None):
            z_coords.append(b_coord)
            z_payloads.append(("B", 0, b_payload))

            b_coord, b_payload = get_next(b)

        return Fiber(z_coords, z_payloads)



    def __lshift__(self, other):
        def get_next(iter):
            try:
                coord, payload = next(iter)
            except StopIteration:
                return (None, None)
            return (coord, payload)
            
        b = other.__iter__()

        z_coords = []
        z_payloads = []
    
        b_coord, b_payload = get_next(b)

        while (not b_coord is None):
            z_coords.append(b_coord)

            a_payload = self.payload(b_coord)
            if (a_payload is None):
                # Iemporary value (should be None)
                if callable(self.default):
                    value = self.default()
                else:
                    value = self.default
                self.insert(b_coord, value)
                # Inefficient: another search
                a_payload = self.payload(b_coord)

            z_payloads.append( (a_payload, b_payload) )
            b_coord, b_payload = get_next(b)

        return Fiber(z_coords, z_payloads)

#
#  String operatons
#        
    def __str__(self):
        str = "["
        for i in range(len(self.coords)):
            str += "(%s -> %s) " % (self.coords[i], self.payloads[i])
        str += "]"
        return str

    def print(self, title=None):
        if not title is None:
            print("%s" % title)

        print("%s" % self)
        print("")

#
# Utility functions
#

    def _maybe_box(self, value):
        if isinstance(value, int) or isinstance(value, float):
            return Payload(value)
        else:
            return value


if __name__ == "__main__":

    a = Fiber([2, 4, 6], [3, 5, 7])

    print("Simple print")
    a.print()
    print("----\n\n")
    

    print("Find payload of 0,4,6,3")
    for i in [0, 4, 6, 3]:
        print("Payload of %s = %s" % (i, a.payload(i)))
    print("----\n\n")

    print("Insert payload at coordinates 0, 3, 7")
    for i in [0, 3, 7]:
        p = i*i+1
        a.insert(i, p)
        a.print("List with (%s, %s) inserted" % (i, p))
        print("")
    print("----\n\n")


    print("For loop print")
    for p in a:
        print(p)
    print("----\n\n")


    print("Iterator Print")
    i = a.__iter__()

    while True:
        try:
            coord, payload = next(i)
            print("(%s, %s)" % (coord, payload))
        except StopIteration:
            print("End")
            break
    print("----\n\n")


    print("Intersection")
    
    b = Fiber([2, 6, 8], [4 , 8, 10])

    a.print()
    b.print()

    ab = a & b
    ab.print()
    print("----\n\n")

    print("For Intersection")

    for coord, payload in ab:
        print("(%s, %s)" % (coord, payload))
    
    print("----\n\n")

    print("Union")

    a.print()
    b.print()

    ab = a | b
    ab.print()
    print("----\n\n")

    print("Assignment")
    
    z = Fiber()
    a = Fiber([2, 6, 8], [4 , 8, 10])

    z.print("Z Fiber")
    a.print("A Fiber")

    za = z << a
    za.print("Z << A Fiber")
    print("----\n\n")
