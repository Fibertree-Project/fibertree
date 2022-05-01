#cython: language_level=3
"""Traffic

A class for computing the memory traffic incurred by a tensor
"""

from fibertree import Tensor

class Traffic:
    """Class for computing the memory traffic of a tensor"""
    @staticmethod
    def buffetTraffic(tensor, rank, format_):
        """Compute the buffet traffic for a given tensor and rank"""
        uses = Traffic._getAllUses(tensor.getUseStats()["Rank " + rank])
        use_data = {}
        for _, use in uses:
            if use not in use_data.keys():
                use_data[use] = [format_.getSubTree(*use), 0]

            use_data[use][1] += 1

        return sum(data[0] * data[1] for data in use_data.values())

    @staticmethod
    def cacheTraffic(tensor, rank, format_, capacity):
        """Compute the cache traffic for given tensor and rank"""
        uses = Traffic._getAllUses(tensor.getUseStats()["Rank " + rank])
        uses.sort()

        # Remove the timestamps
        uses = [use[1] for use in uses]

        # Save some state about the uses
        use_data = {}
        for i, use in enumerate(reversed(uses)):
            if use not in use_data:
                use_data[use] = [format_.getSubTree(*use), []]
            use_data[use][1].append(len(uses) - i - 1)

        # Model the cache
        objs = set()

        occupancy = 0
        bytes_loaded = 0

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
            bytes_loaded += size

            # Immediately evict objects that will never be used again
            use_data[use][1].pop()
            if len(use_data[use][1]) > 0:
                objs.add(use)
                occupancy += size

        return bytes_loaded


    @staticmethod
    def _getAllUses(raw_reuses):
        """
        Get a list of tuples of the form (time point, use)
        """
        # Flatten the reuse list
        reuses = {point + (coord,): stat
                  for point, payloads in raw_reuses.items()
                  for coord, stat in payloads.items()}

        # Get the uses in the order that they occur
        uses = []
        for point in reuses.keys():
            first = reuses[point][0]
            uses.append((first, point))

            for reuse in reuses[point][1]:
                curr = Traffic._getUse(first, reuse)
                uses.append((curr, point))

        return uses

    @staticmethod
    def _getUse(last, dist):
        """
        Combine the last use and reuse distance to get the time point of the
        current use
        """
        return tuple(l + d for l, d in zip(last, dist))


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
