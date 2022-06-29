#cython: language_level=3
"""Traffic

A class for computing the memory traffic incurred by a tensor
"""
import pandas as pd

from fibertree import Tensor

class Traffic:
    """Class for computing the memory traffic of a tensor"""
    @staticmethod
    def buffetTraffic(prefix, tensor, rank, format_):
        """Compute the buffet traffic for a given tensor and rank

        Parameters
        ----------

        prefix: str
            The file prefix where the data for this loopnest was collected

        tensor: Tensor
            The tensor whose buffer traffic to compute

        rank: str
            The name of the buffered rank

        format_: Format
            The format of the tensor

        Returns
        -------

        bits: int
            The number of bits loaded from off-chip memory into the buffet
        """
        uses = Traffic._getAllUses(prefix, tensor, rank)
        use_data = {}
        for use in uses:
            if use not in use_data.keys():
                use_data[use] = [format_.getSubTree(*use), 0]

            use_data[use][1] += 1

        return sum(data[0] * data[1] for data in use_data.values())

    @staticmethod
    def cacheTraffic(prefix, tensor, rank, format_, capacity):
        """Compute the cache traffic for given tensor and rank

        Parameters
        ----------

        prefix: str
            The file prefix where the data for this loopnest was collected

        tensor: Tensor
            The tensor whose buffer traffic to compute

        rank: str
            The name of the buffered rank

        format_: Format
            The format of the tensor

        capacity: int
            The capacity of the cache in bits

        Returns
        -------

        bits: int
            The number of bits loaded from off-chip memory into the cache
        """
        uses = list(Traffic._getAllUses(prefix, tensor, rank))

        # Save some state about the uses
        use_data = {}
        for i, use in enumerate(reversed(uses)):
            if use not in use_data:
                use_data[use] = [format_.getSubTree(*use), []]
            use_data[use][1].append(len(uses) - i - 1)

        # Model the cache
        objs = set()

        occupancy = 0
        bits_loaded = 0

        for i, use in enumerate(uses):
            # If it is already in the cache, we incur no traffic
            if use in objs:
                use_data[use][1].pop()
                if len(use_data[use][1]) == 0:
                    objs.remove(use)
                    occupancy -= use_data[use][0]
                continue

            # Data + metadata stored as 32 bit values
            size = use_data[use][0]

            # Evict until there is space in the cache
            while occupancy + size > capacity:
                obj = Traffic._optimalEvict(use_data, objs)
                objs.remove(obj)
                occupancy -= use_data[obj][0]

            # Now add in the new fiber
            bits_loaded += size

            # Immediately evict objects that will never be used again
            use_data[use][1].pop()
            if len(use_data[use][1]) > 0:
                objs.add(use)
                occupancy += size

        return bits_loaded

    @staticmethod
    def streamTraffic(prefix, tensor, rank, format_):
        """Compute the traffic for streaming over a given tensor and rank

        WARNING: Should not be used for tensors iterated over with
        intersection

        Parameters
        ----------

        prefix: str
            The file prefix where the data for this loopnest was collected

        tensor: Tensor
            The tensor whose buffer traffic to compute

        rank: str
            The name of the buffered rank

        format_: Format
            The format of the tensor

        Returns
        -------

        bits: int
            The number of bits loaded from off-chip memory onto the chip
        """
        uses = Traffic._getAllUses(prefix, tensor, rank)
        curr_fiber = None
        bits = format_.getRHBits(rank)
        fheader = format_.getFHBits(rank)
        elem = format_.getCBits(rank) + format_.getPBits(rank)

        for use in uses:
            fiber = use[:-1]
            if fiber != curr_fiber:
                bits += fheader
                curr_fiber = fiber
            bits += elem

        return bits

    @staticmethod
    def _getAllUses(prefix, tensor, rank):
        """
        Get an iterable of uses ordered by iteration stamp
        """
        df = pd.read_csv(prefix + "-" + rank + ".csv")
        cols = df.columns[(df.shape[1] // 2):]
        ranks = [col for col in cols if col in tensor.getRankIds()]
        records = df[ranks].to_records(index=False)
        return map(tuple, records)

    @staticmethod
    def _optimalEvict(use_data, objs):
        """
        Get the index of the optimal object to evict
        """
        last = -1
        evict = None

        for obj in objs:
            if use_data[obj][1][-1] > last:
                evict = obj
                last = use_data[obj][1][-1]

        return evict
