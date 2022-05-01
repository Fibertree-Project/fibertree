#cython: language_level=3
"""
YS
"""

from fibertree import Tensor

class Compute:
    """YS
    """
    def __init__(self):
        """__init__"""
        pass

    @staticmethod
    def opCount(dump, op):
        """Compute the number of operations executed by this kernel
        YS

        Compute.opCount(Metrics.dump(), "mul") -> return # of multiplications
        """
        metric = "payload_" + op
        if(metric in dump["Compute"].keys()):
            return dump["Compute"][metric]
        else:
            return 0

    def lfCount(dump, rank, leader):
        """Compute the number of intersection attempts 
        
        leader is 0 or 1 depending on which tensor the leader is.
        """
        
        line = "Rank " + rank
        l = "tensor" + str(leader)
        metric = "unsuccessful_intersect_" + l
        return dump[line]["successful_intersect"] + dump[line][metric]
        
    def skipCount(dump, rank):
    
        line = "Rank " + rank
        total = dump[line]["successful_intersect"] + dump[line]["unsuccessful_intersect_tensor0"] + dump[line]["unsuccessful_intersect_tensor1"]
        skipped = dump[line]["skipped_intersect"]
        return total - skipped