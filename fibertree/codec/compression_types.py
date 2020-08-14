from .formats.uncompressed import Uncompressed
from .formats.coord_list import CoordinateList
from .formats.bitvector import Bitvector
from .formats.hashtable import HashTable
from .formats.balanced_tree import RBTree
from .formats.rle import RunLengthEncoding
"""
# U = uncompressed
    # size of vector = shape of fiber
    # contents = 0 if nothing in position, payload otherwise
    # fibers serialized in position order
uncompressed = "U"

# Bu = untruncated bit vector
    # size of vector = shape of this fiber
    # contents = 0 in position if empty, 1 if not
    # when each rank is serialized, fibers are serialized in order
untruncated_bitvector = "Bu"

# Bt = truncated bit vector
    # cut off bit vector at last 1, store number of bits in previous rank's payloads
    # size of vector <= shape of fiber
    # when each rank is serialized, fibers are serialized in order
truncated_bitvector = "Bt"

# C = coordinate list
    # size of vector = occupancy of this fiber
    # contents = sorted / deduplicated coordinates in this fiber
    # when each rank is serialized, fibers are serialized in order
coord_list = "C"

# list of all valid formats
valid_formats =  [uncompressed, coord_list, untruncated_bitvector, truncated_bitvector] 
# ["U", "C", "R", "A", "B", "D", "Hf", "Hr"]

# types of bitvectors
bitvectors = [untruncated_bitvector, truncated_bitvector]

# TO BE IMPLEMENTED
# D = delta compressed
    # num elements in vector = occupancy of fiber
    # contents = delta-compressed coordinate list
    # serialize according to position order
# Hf = hash table per fiber
    # TODO
"""
# mapping descriptors to formats
"""
uncompressed = Uncompressed().getName()
coord_list = CoordinateList.getName()
descriptor_to_fmt = {uncompressed : Uncompressed, coord_list: CoordinateList}
"""

# TODO: figure out how to register yourself

descriptor_to_fmt = {"U" : Uncompressed, "C":CoordinateList, "B": Bitvector, "T": RBTree, "H":HashTable }

# , "C": CoordinateList, "B": Bitvector, "Hf" : HashTable(), "T": RBTree, "R": RunLengthEncoding }# , "UB": UncompressedBitvector}
