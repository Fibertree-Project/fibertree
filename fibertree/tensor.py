from fibertree.rank  import Rank
from fibertree.fiber import Fiber

""" Tensor """

class Tensor:
    """ Tensor Class """

    def __init__(self,rank_ids=["X"], n=0):
        """__init__"""

        self.rank_ids = rank_ids

        #
        # Create a linked list of ranks
        #
        self.ranks = []
        old_rank = None
        for id in rank_ids:
            new_rank = Rank(name=id)
            if not old_rank is None: old_rank.set_next(new_rank)
            old_rank = new_rank
            self.ranks.append(new_rank)
            
        if (n == 0):
            root_fiber = Fiber()
            
            if len(rank_ids) != 1:
                root_fiber.set_default(Fiber)
            else:
                root_fiber.set_default(0)

            self.ranks[0].append(root_fiber)
            return

        if (n > 0):
            self._builtin_value(n)


    def root(self):
        """root"""

        return self.ranks[0].fibers[0]

#
# String methods
#
    def print(self, title=None):
        """print"""

        if not title is None:
            print("%s" % title)

        print("%s" % self)
        print("")

    def __repr__(self):
        """__repr__"""

        str = "T(%s)/[" % ",".join(self.rank_ids) + "\n"
        for r in self.ranks:
            str += "  " + r.__repr__() + "\n"
        str += "]"
        return str
#
#
# Temporary tensor initialization values
#
    def _builtin_value(self, n):
        """_builtin_value"""

        if (n == 1):

            rank0 = self.ranks[0]            
            rank1 = self.ranks[1]
            
            a_k = Fiber([0, 2, 4, 6],            # Coordinates
                        [1, 3, 5, 7])            # Payloads

            rank1.append(a_k)

            a_m = Fiber([0], [a_k])

            rank0.append(a_m)
        elif n == 2:
            b_k = Fiber([2, 6, 8],                # Coordinates
                        [4 , 8, 10])              # Payloads

            self.ranks[1].append(b_k)

            b_m = Fiber([0], [b_k])

            self.ranks[0].append(b_m)
        elif n == 3:
            a_m = Fiber([0, 2, 4, 6],             # Coordinates
                        [1 , 3, 5, 7])            # Payloads

            self.ranks[0].append(a_m)
        elif n == 4:
            a_m = Fiber([0, 2, 4, 6],             # Coordinates
                        [1 , 3, 5, 7])            # Payloads

            self.ranks[0].append(a_m)
        elif n == 5:
            b_m = Fiber([2, 4],
                        [2, 4])

            self.ranks[0].append(b_m)
        elif n == 6:
            a_k0 = Fiber([0, 2],
                         [1, 3])

            self.ranks[1].append(a_k0)

            a_k1 = Fiber([0, 3],
                         [1, 4])

            self.ranks[1].append(a_k1)

            a_k3 = Fiber([2, 3],
                         [3, 4])

            self.ranks[1].append(a_k3)

            a_m0 = Fiber([0,  1,  3],
                      [a_k0, a_k1, a_k3])

            self.ranks[0].append(a_m0)
            
        else:
            assert(False)

        
