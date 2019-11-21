from fibertree.rank  import Rank
from fibertree.fiber import Fiber


class Tensor:

    def __init__(self,rank_ids=["X"], n=0):

        self.ranks = []
        for id in rank_ids:
            self.ranks.append(Rank(name=id))

        print("Ranks = %s" % self.ranks)
        
        if (n == 0):
            self.ranks[0].append(Fiber(default = 0))
            return
        
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



    def root(self):
        print("Root name: %s" % self.ranks[0].name)
        return self.ranks[0].fibers[0]

