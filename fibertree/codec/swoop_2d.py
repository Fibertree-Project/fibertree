from fibertree import Fiber
from fibertree import Tensor
from fibertree import Codec
import yaml
#
# Tensor
#
# This is a format-independent abstraction for a Tensor that contains abstract
# "Rank"s that can be later set to have a concrete format and/or data.
#


class SwoopTensor:
  def __init__(self, name, rank_ids):
    assert(len(rank_ids) > 0)
    self.ranks = {}
    for r in rank_ids:
      self.ranks[r] = Rank(name + "_" + r)
    self.name = name
    self.rank_ids = rank_ids
  
  def __getitem__(self, rank_id):
    return self.ranks[rank_id]

  def setImplementations(self, rank_id, imps):
    self.ranks[rank_id].setImplementations(imps)

#
# Rank
#
# This is a format-independent abstraction for a Rank that dispatches
# method calls to a format-specific version. The reason this class
# exists is that it lets us define swoop programs first, then fill in
# the formats as a later step (using setImplementation()).
#


class Rank:
  def __init__(self, name):
    self.implementations = []
    self.name = name
    self.current_fiber = None
  
  def getRootFiberHandle(self):
    return 0
  
  def setImplementations(self, imps):
    self.implementations = imps
    self.current_fiber = 0

  def setCurrent(self, fiber_idx):
    self.current_fiber = fiber_idx
  
  def nextFiber(self):
    self.current_fiber += 1

  def setupSlice(self, base_coord = 0, bound_coord = None, max_num = None):
    self.implementations[self.current_fiber].setupSlice(base_coord, bound_coord, max_num)
    return self.current_fiber

  def nextInSlice(self):
    return self.implementations[self.current_fiber].nextInSlice()
  
  def handleToCoord(self, handle):
    return self.implementations[self.current_fiber].handleToCoord(handle)
  
  def handleToPayload(self, handle):
    return self.implementations[self.current_fiber].handleToPayload(handle)
  
  def payloadToFiberHandle(self, payload):
    return self.implementations[self.current_fiber].payloadToFiberHandle(payload)

  def payloadToValue(self, payload):
    return self.implementations[self.current_fiber].payloadToValue(payload)
    
  def coordToHandle(self, coord):
    return self.implementations[self.current_fiber].coordToHandle(coord)
  
  def insertElement(self, coord):
    return self.implementations[self.current_fiber].insertElement(coord)
  
  def updatePayload(self, handle, payload):
    return self.implementations[self.current_fiber].updatePayload(handle, payload)

  def dumpStats(self, stats_dict):
    for impl in self.implementations:
      impl.dumpStats(stats_dict)

#
# AST
#
# Base Class for all AST nodes
# Most have a reference to a Rank. Additionally, track the "fanout" number
# which is the number of receivers. If this remains 0, this Node is unconnected
# and can be eliminated as dead code.
#
class AST:

  def __init__(self, rank = None, num_fields = 1):
    self.rank = rank
    self.num_fields = num_fields
    self.num_fanout = 0
    self.fanout_mask = {}
    self.cur_result = None
    self.producers = []
    self.initialized = False
    self.finalized = False
  
  def _addProducer(self, other):
    self.producers.append(other)
  
  def connect(self, other):
    other._addProducer(self)
    self.fanout_mask[other] = False
    self.num_fanout += 1

  def initialize(self):
    self.initialized = True
    for prod in self.producers:
      if not prod.initialized:
        prod.initialize()
    
  def evaluate(self):
    self.trace("Unimplemented Evaluate")
    return None
  
  def finalize(self, stats_dict):
    if (hasattr(self, "rank") and self.rank != None):
      self.rank.dumpStats(stats_dict)
    self.finalized = True
    for prod in self.producers:
      if not prod.finalized:
        prod.finalize(stats_dict)
    
  def needEvaluation(self, other):
    # If no one has asked yet, advance.
    if not any(self.fanout_mask.values()):
      return True
    # If the same receiver asks twice, advance.
    if self.fanout_mask[other]:
      # Reset the mask.
      for receiver in self.fanout_mask.keys():
        self.fanout_mask[receiver] = False
      return True
    return False
  
  def nextValue(self, other):
    # Call evaluate, but only once and fan out the result to later callers.
    if self.needEvaluation(other):
      self.cur_result = self.evaluate()
    self.fanout_mask[other] = True
    # If everyone got the value, reset the mask.
    if all(self.fanout_mask.values()):
      for receiver in self.fanout_mask.keys():
        self.fanout_mask[receiver] = False
    return self.cur_result
  
  def trace(self, args):
    if (hasattr(self, "rank") and self.rank != None):
      print(self.rank.name + ":", args)
    else:
      print(args)

  # These are for Nodes that want to return a tuple from evaluate.
  def __iter__(self):
    self.cur_field = 0
    return self
  
  def __next__(self):
    if self.cur_field > (self.num_fields - 1):
       raise StopIteration
    sp = Splitter(self, self.cur_field)
    self.cur_field += 1
    return sp

#
# Slice
#
# Given an AST::Rank, and a slice spec returns a rank-1 stream of all handles
# to elements in that slice.
# 

class Slice (AST):
  def __init__(self, rank, base = 0, bound  = None, max_num = None):
    super().__init__(rank)
    self.base = base
    self.bound = bound
    self.max_num = max_num
  
  
  def initialize(self):
    self.rank.setupSlice(self.base, self.bound, self.max_num)
    super().initialize()
  
  def evaluate(self):
    res = self.rank.nextInSlice()
    self.trace(f"NextInSlice: {res}")
    return res
  
#
# Scan
#
# Given a Rank, and a 1-stream of fiber_handles,
#  returns a 2-stream of all handles to elements in those fibers.
# 

class Scan (AST):
  def __init__(self, rank, fiber_handles):
    super().__init__(rank)
    self.fiber_handles = fiber_handles
    fiber_handles.connect(self)
    self.active = False
  
  def evaluate(self):
    if not self.active:
      fiber_handle = self.fiber_handles.nextValue(self)
      if fiber_handle is None:
        self.trace(f"Scan: Done.")
        return None
      self.trace(f"Scan: Start {fiber_handle}")
      self.rank.setCurrent(fiber_handle)
      self.rank.setupSlice() # TODO use base/bound/max_num here somehow
      self.active = True

    res = self.rank.nextInSlice()
    if res is None:
      self.active = False
      self.trace(f"Scan: Inner Done.")
      return None
    self.trace(f"ScanNext: {res}")
    return res
#
# Iterate
#
# Simple convenience alias for iterating over an entire fiber
#

def Iterate(rank):
  return Slice(rank)

#
# HandlesToCoords
#
# Given a reference to an AST Rank, and an AST Node that 
# that produces a N-stream of handles, produces a N-stream of coordinates
#
class HandlesToCoords (AST):

  def __init__(self, rank, handles):
    super().__init__(rank)
    self.handles = handles
    handles.connect(self)

  def evaluate(self):
    handle = self.handles.nextValue(self)
    if handle is None:
      self.trace(f"HandleToCoord: None")
      return None
    coord = self.rank.handleToCoord(handle)
    self.trace(f"HandleToCoord: {handle}, {coord}")
    return coord
    
#
# HandlesToPayloads
#
# Given a reference to an AST Rank, and an AST Node that 
# that produces a N-stream of handles, produces a N-stream of payloads
#
class HandlesToPayloads (AST):

  def __init__(self, rank, handles):
    super().__init__(rank)
    self.handles = handles
    handles.connect(self)

  def evaluate(self):
    handle = self.handles.nextValue(self)
    if handle is None:
      self.trace(f"HandleToPayload: None")
      return None
    payload = self.rank.handleToPayload(handle)
    self.trace(f"HandleToPayload: {handle}, {payload}")
    return payload


#
# HandlesToCoordsAndPayloads
#
# Simple convenience alias for concise code
#
#def HandlesToCoordsAndPayloads(rank, handles):
#  return (HandlesToCoords(rank, handles2), HandlesToPayloads(rank, handles2))
  

#
# PayloadsToFiberHandles
#
# Given a reference to an AST Rank, and an AST Node that 
# that produces a N-stream of payloads, produces a N-stream of Fiber Handles
#
class PayloadsToFiberHandles (AST):

  def __init__(self, rank, payloads):
    super().__init__(rank)
    self.payloads = payloads
    payloads.connect(self)

  def evaluate(self):
    payload = self.payloads.nextValue(self)
    if payload is None:
      self.trace(f"PayloadToFiberHandle: None")
      return None
    fiber_handle = self.rank.payloadToFiberHandle(payload)
    self.trace(f"PayloadToFiberHandle: {payload}, {fiber_handle}")
    return fiber_handle

#
# PayloadsToValues
#
# Given a reference to an AST Rank, and an AST Node that 
# that produces a N-stream of payloads, produces a N-stream of Values
#
class PayloadsToValues (AST):

  def __init__(self, rank, payloads):
    super().__init__(rank)
    self.payloads = payloads
    payloads.connect(self)

  def evaluate(self):
    payload = self.payloads.nextValue(self)
    if payload is None:
      self.trace(f"PayloadToValue: None")
      return None
    value = self.rank.payloadToValue(payload)
    self.trace(f"PayloadToValue: {payload}, {value}")
    return value



#
# CoordsToHandles
#
# Given a reference to an AST Rank, and an AST Node that 
# that produces a N-stream of coords, produces a N-stream of handles
# (NOTE: EXPENSIVE FOR MOST FORMATS)
# TODO: Add starting position.
#
class CoordsToHandles (AST):

  def __init__(self, rank, coords):
    super().__init__(rank)
    self.coords = coords
    coords.connect(self)

  def evaluate(self):
    coord = self.coords.nextValue(self)
    if coord is None:
      self.trace(f"CoordToHandle: None")
      return None
    handle = self.rank.coordToHandle(coord)
    self.trace(f"CoordToHandle: {coord}, {handle}")
    return handle

#
# InsertElement
#
# Given a reference to an AST Rank, and an AST Node that 
# that produces a N-stream of coords, produces a N-stream of handles
# after creating that (coord, payload) element and initializing coord
# (NOTE: EXPENSIVE FOR MOST FORMATS)
# TODO: Add starting position.
#

class InsertElement (AST):

  def __init__(self, rank, coords):
    super().__init__(rank)
    self.coords = coords
    coords.connect(self)

  def evaluate(self):
    coord = self.coords.nextValue(self)
    if coord is None:
      self.trace(f"InsertElement: None")
      return None
    handle = self.rank.insertElement(coord)
    self.trace(f"InsertElement: {coord}, {handle}")
    return handle

#
# UpdatePayloads
#
# Given a reference to an AST Rank, and an AST Node that produces
# a N-stream of handles, and an AST Node that produces a N-stream
# of payloads, updates each element (coord, payload) to the new payload.
#

class UpdatePayloads (AST):

  def __init__(self, rank, handles, payloads):
    super().__init__(rank)
    self.handles = handles
    handles.connect(self)
    self.payloads = payloads
    payloads.connect(self)

  def evaluate(self):
    handle = self.handles.nextValue(self)
    payload = self.payloads.nextValue(self)
    if handle is None or payload is None:
      assert (handle is None and payload is None)
      self.trace(f"UpdatePayloads: None")
      return None
    self.trace(f"UpdatePayloads: {handle}, {payload}")
    return self.rank.updatePayload(handle, payload)


# 
# Intersect
# 
#

class Intersect (AST):
  def __init__(self, a_coords, a_handles, b_coords, b_handles):
    # Note: we return a 3-tuple, so tell the super-class that.
    super().__init__(num_fields = 3)
    self.a_coords = a_coords
    a_coords.connect(self)
    self.a_handles = a_handles
    a_handles.connect(self)
    self.b_coords = b_coords
    b_coords.connect(self)
    self.b_handles = b_handles
    b_handles.connect(self)

  def evaluate(self):
    #a_coord = self.a_coords.nextValue()
    #a_handle = self.a_handles.nextValue()
    #b_coord = self.b_coords.nextValue()
    #b_handle = self.b_handles.nextValue()
    a_coord = -2
    b_coord = -1
    a_handle = None
    b_handle = None
    while a_coord != None and b_coord != None:
      if a_coord == b_coord:
        self.trace(f"Intersection found at: {a_coord}: ({a_handle}, {b_handle})")
        return (a_coord, a_handle, b_handle)
      while a_coord != None and b_coord != None and a_coord < b_coord:
        a_coord = self.a_coords.nextValue(self)
        a_handle = self.a_handles.nextValue(self)
        self.trace(f"Intersection advancing A: {a_coord}, {b_coord} ({a_handle}, {b_handle})")        
      while b_coord != None and a_coord != None and b_coord < a_coord:
        b_coord = self.b_coords.nextValue(self)
        b_handle = self.b_handles.nextValue(self)
        self.trace(f"Intersection advancing B: {a_coord}, {b_coord} ({a_handle}, {b_handle})")
      # If one ended, drain the other
      if a_coord is None:
        while b_coord is not None:
          b_coord = self.b_coords.nextValue(self)
          b_handle = self.b_handles.nextValue(self)
          self.trace(f"Intersection draining B: {b_coord} ({b_handle})")
        return (None, None, None)
      elif b_coord is None:
        while a_coord is not None:
          a_coord = self.a_coords.nextValue(self)
          a_handle = self.a_handles.nextValue(self)
          self.trace(f"Intersection draining A: {a_coord} ({a_handle})")
        return (None, None, None)

    self.trace("Intersection done.")
    return (None, None, None)
  

#
# Splitter
#
# Helper module for splitting tuple streams.
# Note: explicitly over-rides default fanout behavior.
#

class Splitter (AST):
  def __init__(self, stream, num):
    super().__init__()
    self.stream = stream
    stream.connect(self)
    self.num = num

  def evaluate(self):
    return self.stream.nextValue(self)

  def nextValue(self, other):
    cur_result = super().nextValue(other)
    self.trace(f"Splitter[{self.num}]: {self.cur_result[self.num]}")
    return self.cur_result[self.num]

#
# Compute
#
# Given an N-argument function and a list of N AST nodes that produce
# N-streams of values, apply the function to the values to produce an N-stream
# of outputs
#

class Compute (AST):
  def __init__(self, function, *streams):
    super().__init__()
    self.streams = streams
    for stream in streams:
      stream.connect(self)
    self.function = function

  def evaluate(self):
    args = [None] * len(self.streams)
    for x, stream in enumerate(self.streams):
      args[x] = stream.nextValue(self)
    # If one arg is None, they all should be None (in which case, skip the func)
    any_is_none = False
    all_are_none = True
    for arg in args:
      is_none = arg is None
      any_is_none |= is_none
      all_are_none &= is_none
    
    assert(not any_is_none or all_are_none)
    
    if all_are_none:
      self.trace(f"Compute: None")
      return None
      
    result = self.function(*args)
    self.trace(f"Compute({args}) => {result}")
    return result

#
# Amplify
#
# Given an AST node that produces an N-stream, and a Node that produces an
# N+1-stream, replicate each element from the N-stream, so that the output
# is an N+1-stream.
#

class Amplify (AST):
  def __init__(self, smaller, bigger):
    super().__init__()
    self.smaller = smaller
    smaller.connect(self)
    self.bigger = bigger
    bigger.connect(self)
    self.current_value = None
  
  def evaluate(self):
    if self.current_value is None: # Is this initialization condition correct?  
      self.current_value = self.smaller.nextValue(self)
      self.trace(f"Amplify Init: {self.current_value}")
      
    next = self.bigger.nextValue(self)
    if next is None:
      self.current_value = None
      self.trace(f"Amplify: Shrink")

    self.trace(f"Amplify: {next} => {self.current_value}")
    return self.current_value

#
# Reduce
#
# Given an AST node that produces an N-stream, and a Node that produces an
# N-1-stream, reduce each element from the N-stream, so that the output
# is an N-1-stream.
#

class Reduce (AST):
  def __init__(self, smaller, bigger):
    super().__init__()
    self.smaller = smaller
    smaller.connect(self)
    self.bigger = bigger
    bigger.connect(self)
  
  def evaluate(self):

    current_value = self.smaller.nextValue(self)
    if current_value is None:
      next = self.bigger.nextValue(self)
      assert(next is None)
      self.trace(f"Reduce Init: None")
      return None
    self.trace(f"Reduce Init: {current_value}")
    
    next = 0
    while next is not None:
      next = self.bigger.nextValue(self)
      if next is not None:
        current_value += next
        self.trace(f"Reduce: {current_value}")
    self.trace(f"Reduce: Done")
    return current_value

#
# Stream0
#
# Turn a scalar into a 0-stream that transmits exactly that scalar.
#


class Stream0 (AST):
  def __init__(self, val):
    super().__init__()
    self.val = val
    self.done = False
  
  
  def evaluate(self):
    assert(not self.done)
    self.done = True
    return self.val


#
# BasicIntermediateRankImplementation
#
# Rank implementation JUST to test out the program below.

class BasicIntermediateRankImplementation:
  def __init__(self, shape, shape_of_next_rank):
    self.shape = shape
    self.shape_of_next_rank = shape_of_next_rank
  
  def setupSlice(self, base, bound, max_num):
    # ignore base/bound/max num because this class is BASIC.
    self.max_num = self.shape
    self.cur_num = 0
  
  def nextInSlice(self):
    if self.cur_num >= self.max_num:
      return None
    num = self.cur_num
    self.cur_num += 1
    return num
  
  def handleToCoord(self, handle):
    return handle
  
  def handleToPayload(self, handle):
    return (handle * self.shape_of_next_rank, (handle+1) * self.shape_of_next_rank)
  
  def payloadToFiberHandle(self, payload):
    return payload[0] // self.shape_of_next_rank
  
  def payloadToValue(self, payload):
    assert(False)
  
  def coordToHandle(self, coord):
    return coord
  
  def insertElement(self, coord):
    return coord
  
  def updatePayload(self, handle, payload):
    return handle

  def dumpStats(self, stats_dict):
    pass

#
# BasicFiberImplementation
#
# Rank implementation JUST to test out the programs below.

class BasicFiberImplementation:
  def __init__(self, vals):
    self.vals = vals
  
  def setupSlice(self, base, bound, max_num):
    # ignore base/bound/max num because this class is BASIC.
    self.max_num = len(self.vals)
    self.cur_num = 0
  
  def nextInSlice(self):
    if self.cur_num >= self.max_num:
      return None
    num = self.cur_num
    self.cur_num += 1
    return num
  
  def handleToCoord(self, handle):
    return handle
  
  def handleToPayload(self, handle):
    return self.vals[handle]
  
  def payloadToFiberHandle(self, payload):
    assert(False)
  
  def payloadToValue(self, payload):
    return payload
  
  def coordToHandle(self, coord):
    return coord
  
  def insertElement(self, coord):
    return coord
  
  def updatePayload(self, handle, payload):
    assert(handle is not None)
    self.vals[handle] = payload
    return handle

  def dumpStats(self, stats_dict):
    pass

#
# evaluate
#
# Run the given node (and all nodes connected to it) until it returns None
# N times in a row.
#


def evaluate(node, n = 1, stats_dict = {}):
  assert(n > 0)
  node.initialize()
  consecutive_dones = 0
  while (consecutive_dones != n):
    res = node.evaluate()
    print(f"+++++++++")
    print(f"Evaluate: {res}")
    print(f"+++++++++")
    if res is None:
      consecutive_dones += 1
    else:
      consecutive_dones = 0
  node.finalize(stats_dict)

def encodeTensorInFormat(tensor, descriptor):
    codec = Codec(tuple(descriptor), [True]*len(descriptor))

    # get output dict based on rank names
    rank_names = tensor.getRankIds()
    # print("encode tensor: rank names {}, descriptor {}".format(rank_names, descriptor))
    # TODO: move output dict generation into codec
    output = codec.get_output_dict(rank_names)
    # print("output dict {}".format(output))
    output_tensor = []
    for i in range(0, len(descriptor)):
            output_tensor.append(list())

    # print("encode, output {}".format(output_tensor))
    codec.encode(-1, tensor.getRoot(), tensor.getRankIds(), output, output_tensor)
    # print(output_tensor)

    # name the fibers in order from left to right per-rank    
    rank_idx = 0
    for rank in output_tensor:
        fiber_idx = 0
        for fiber in rank:
            fiber_name = "_".join([tensor.getName(), rank_names[rank_idx], str(fiber_idx)])
            fiber.setName(fiber_name)
            fiber_idx += 1
        rank_idx += 1
    return output_tensor

def dumpAllStatsFromTensor(tensor, output):
    # print(tensor)
    for rank in tensor:
        for fiber in rank:
            # fiber.printFiber()
            fiber.dumpStats(output)

## Test program: Element-wise multiplication
# a_k = A.getRoot()
# b_k = B.getRoot()
# for k, (z, (a, b)) in z_k << (a_k & b_k):
#   z_ref <<= a_k * b_k

# Define the tensors
a = SwoopTensor(name = "A", rank_ids = ["K"])
b = SwoopTensor(name = "B", rank_ids = ["K"])
z = SwoopTensor(name = "Z", rank_ids = ["K"])

# Convenience shortcuts
a_k = a["K"]
b_k = b["K"]
z_k = z["K"]

# Iterate the root ranks and get handles to their contents
a_handles = Iterate(a_k)
b_handles = Iterate(b_k)
# Convert handles to coordinates
a_coords = HandlesToCoords(a_k, a_handles)
b_coords = HandlesToCoords(b_k, b_handles)
# Intersect the K rank
(ab_coords, ab_a_handles, ab_b_handles) = Intersect(a_coords, a_handles, b_coords, b_handles)
# Only insert elements that survive intersection
z_handles = InsertElement(z_k, ab_coords)
# Only retrieve the values that survive intersection
a_payloads = HandlesToPayloads(a_k, ab_a_handles)
b_payloads = HandlesToPayloads(b_k, ab_b_handles)
a_values = PayloadsToValues(a_k, a_payloads)
b_values = PayloadsToValues(b_k, b_payloads)
# Calculate the loop body
results = Compute(lambda a, b: a*b, a_values, b_values)
# Final writeback
z_k_update_acks = UpdatePayloads(z_k, z_handles, results)

# Create some example implmentations
# myA_K = BasicFiberImplementation([1, 2, 3])
# myB_K = BasicFiberImplementation([4, 5, 6])
# myZ_K = BasicFiberImplementation([None, None, None])

descriptor = ["C"]
ranks = ["K"]

# base data
A_data = [1, 2, 3]
B_data = [4, 5, 6]
Z_data = [0, 0, 0] # this used to be None but 0 makes it a bit easier from HFA

# convert to HFA
A_HFA = Tensor.fromUncompressed(ranks, A_data, name = "A")
B_HFA = Tensor.fromUncompressed(ranks, B_data, name = "B")
Z_HFA = Tensor.fromUncompressed(ranks, Z_data, name = "Z")

# get fibers from tensor encodings
myA = encodeTensorInFormat(A_HFA, descriptor)
myB = encodeTensorInFormat(B_HFA, descriptor)
myZ = encodeTensorInFormat(Z_HFA, descriptor)

# Use those implementations in practice
a.setImplementations("K", myA[0])
b.setImplementations("K", myB[0])
z.setImplementations("K", myZ[0])

# Run the program and check and print the result
evaluate(z_k_update_acks)

# NOTE: to get the payloads for comparison requires some knowledge of the output format.
# for example, in C, the outputs will be compressed, while they will not be in U
output_fiber = myZ[0][0]
# dump stats
stats_dict = dict()
dumpAllStatsFromTensor(myA, stats_dict)
dumpAllStatsFromTensor(myB, stats_dict)
dumpAllStatsFromTensor(myZ, stats_dict)
print("\nElt-wise multiply: A = <C>, B = <C>, Z = <C>")
print(yaml.dump(stats_dict))


print("===========================")
print(f"Final element-wise result: {output_fiber.payloads}")
print("===========================")
assert(output_fiber.payloads == [4, 10, 18])

## Test program: A-Stationary vector-matrix multiplication
#for k, (a, b_n) in a_k & b_k:
#  for n, (z, b) in z_n << b_n:
#    z += a * b

print("\n *** A-stationary vector-matrix ***")
a = SwoopTensor(name="A", rank_ids=["K"])
b = SwoopTensor(name="B", rank_ids=["K", "N"])
z = SwoopTensor(name="Z", rank_ids=["N"])

a_k = a["K"]
b_k = b["K"]
b_n = b["N"]
z_n = z["N"]

# a_k & b_k
a_k_handles = Iterate(a_k)
b_k_handles = Iterate(b_k)
a_k_coords = HandlesToCoords(a_k, a_k_handles)
b_k_coords = HandlesToCoords(b_k, b_k_handles)
(ab_k_coords, ab_a_k_handles, ab_b_k_handles) = Intersect(a_k_coords, a_k_handles, b_k_coords, b_k_handles)
ab_a_k_payloads = HandlesToPayloads(a_k, ab_a_k_handles)
ab_b_k_payloads = HandlesToPayloads(b_k, ab_b_k_handles)

# z_n << b_n
ab_b_n_fiber_handles = PayloadsToFiberHandles(b_k, ab_b_k_payloads)
b_n_handles = Scan(b_n, ab_b_n_fiber_handles)
b_n_coords = HandlesToCoords(b_n, b_n_handles)
b_n_payloads = HandlesToPayloads(b_n, b_n_handles)
z_n_coords = b_n_coords
z_n_handles = InsertElement(z_n, z_n_coords)
z_n_payloads = HandlesToPayloads(z_n, z_n_handles)

# z_ref += a_val * b_val
a_values = PayloadsToValues(a_k, ab_a_k_payloads)
b_values = PayloadsToValues(b_n, b_n_payloads)
z_values = PayloadsToValues(z_n, z_n_payloads)
# We need to repeat A value across every Z
a_values_amplified = Amplify(a_values, z_n_coords)
# Actual computation
body_func = lambda a_val, b_val, z_ref: z_ref + a_val * b_val
z_new_values = Compute(body_func, a_values_amplified, b_values, z_values)
# Final write-back
z_n_update_acks = UpdatePayloads(z_n, z_n_handles, z_new_values)

K=3
N=3

A_data = [1, 2, 3]
B_data = [[4, 5, 6], [5, 6, 7], [6, 7, 8]]
Z_data = [0, 0, 0]

# convert to HFA
A_HFA = Tensor.fromUncompressed(["K"], A_data, name = "A")
B_HFA = Tensor.fromUncompressed(["K", "N"], B_data, name = "B")
Z_HFA = Tensor.fromUncompressed(["N"], Z_data, name = "Z")

A_tensor = encodeTensorInFormat(A_HFA, ["C"])
B_tensor = encodeTensorInFormat(B_HFA, ["C", "C"])
Z_tensor = encodeTensorInFormat(Z_HFA, ["C"])

a.setImplementations("K", A_tensor[0])
b.setImplementations("K", B_tensor[0])
b.setImplementations("N", B_tensor[1])
z.setImplementations("N", Z_tensor[0])

"""
my_a_k = BasicFiberImplementation([1, 2, 3])
my_b_k = BasicIntermediateRankImplementation(K, N)
my_b_n = [
           BasicFiberImplementation([4, 5, 6]), 
           BasicFiberImplementation([5, 6, 7]), 
           BasicFiberImplementation([6, 7, 8])
         ]
my_z_n = BasicFiberImplementation([0, 0, 0])
a.setImplementations("K", [my_a_k])
b.setImplementations("K", [my_b_k])
b.setImplementations("N", my_b_n)
z.setImplementations("N", [my_z_n])

"""
z_fiber = Z_tensor[0][0]
# print(z_fiber.printFiber())
evaluate(z_n_update_acks, 2)
print("==========================")
print(f"Final A-Stationary result: {z_fiber.payloads}")
print("==========================")
assert(Z_tensor[0][0].payloads == [32, 38, 44])

# dump stats
stats_dict = dict()
dumpAllStatsFromTensor(A_tensor, stats_dict)
dumpAllStatsFromTensor(B_tensor, stats_dict)
dumpAllStatsFromTensor(Z_tensor, stats_dict)
print("\nA-stationary vector-matrix: A = <C>, B = <C, C>, Z = <C>")
print(yaml.dump(stats_dict))
print()

## Test program: Z-Stationary vector-matrix multiplication
#for n, (z, b_k) in z_n << b_n:
#  for k, (a, b) in a_k & b_k:
#    z += a * b
print("\n*** Z-stationary vector-matrix ***")
a = SwoopTensor(name="A", rank_ids=["K"])
b = SwoopTensor(name="B", rank_ids=["N", "K"])
z = SwoopTensor(name="Z", rank_ids=["N"])

a_k = a["K"]
b_n = b["N"]
b_k = b["K"]
z_n = z["N"]

# z_n << b_n
b_n_handles = Iterate(b_n)
b_n_coords = HandlesToCoords(b_n, b_n_handles)
b_n_payloads = HandlesToPayloads(b_n, b_n_handles)
z_n_coords = b_n_coords
z_n_handles = InsertElement(z_n, z_n_coords)
z_n_payloads = HandlesToPayloads(z_n, z_n_handles)
b_k_n_fiber_handles = PayloadsToFiberHandles(b_n, b_n_payloads)


# a_k & b_k
b_k_handles = Scan(b_k, b_k_n_fiber_handles)
# Repeat a_k iteration for each b_k
a_k_fiber_handle = Stream0(a_k.getRootFiberHandle())
a_k_fiber_handles = Amplify(a_k_fiber_handle, b_k_n_fiber_handles)
a_k_handles = Scan(a_k, a_k_fiber_handles) 
a_k_coords = HandlesToCoords(a_k, a_k_handles)
b_k_coords = HandlesToCoords(b_k, b_k_handles)
(ab_k_coords, ab_a_k_handles, ab_b_k_handles) = Intersect(a_k_coords, a_k_handles, b_k_coords, b_k_handles)
ab_a_k_payloads = HandlesToPayloads(a_k, ab_a_k_handles)
ab_b_k_payloads = HandlesToPayloads(b_k, ab_b_k_handles)


# z_ref += a_val * b_val
a_values = PayloadsToValues(a_k, ab_a_k_payloads)
b_values = PayloadsToValues(b_k, ab_b_k_payloads)
# NOTE: MUL and ADD broken out for efficiency
body_func = lambda a_val, b_val: a_val * b_val
partial_products = Compute(body_func, a_values, b_values)
z_values = z_n_payloads
# Reduce into the same value until end of rank
z_new_values = Reduce(z_values, partial_products)
z_n_update_acks = UpdatePayloads(z_n, z_n_handles, z_new_values)

# convert to HFA
A_HFA = Tensor.fromUncompressed(["K"], A_data, name = "A")
B_HFA = Tensor.fromUncompressed(["N", "K"], B_data, name = "B")
Z_HFA = Tensor.fromUncompressed(["N"], Z_data, name = "Z")

A_tensor = encodeTensorInFormat(A_HFA, ["C"])
B_tensor = encodeTensorInFormat(B_HFA, ["C", "C"])
Z_tensor = encodeTensorInFormat(Z_HFA, ["C"])

a.setImplementations("K", A_tensor[0])
b.setImplementations("N", B_tensor[0])
b.setImplementations("K", B_tensor[1])
z.setImplementations("N", Z_tensor[0])
"""
my_a_k = BasicFiberImplementation([1, 2, 3])
my_b_n = BasicIntermediateRankImplementation(N, K)
my_b_k = [BasicFiberImplementation([4, 5, 6]), 
          BasicFiberImplementation([5, 6, 7]), 
          BasicFiberImplementation([6, 7, 8])]
my_z_n = BasicFiberImplementation([0, 0, 0])


a.setImplementations("K", [my_a_k])
b.setImplementations("N", [my_b_n])
b.setImplementations("K", my_b_k)
z.setImplementations("N", [my_z_n])
"""

z_fiber = Z_tensor[0][0]
# print(z_fiber.printFiber())
evaluate(z_n_update_acks)

# dump stats
stats_dict = dict()
dumpAllStatsFromTensor(A_tensor, stats_dict)
dumpAllStatsFromTensor(B_tensor, stats_dict)
dumpAllStatsFromTensor(Z_tensor, stats_dict)

print("\nZ-stationary vector-matrix: A = <C>, B = <C, C>, Z = <C>")
print(yaml.dump(stats_dict))
print()
print("==========================")
print(f"Final Z-Stationary result: {z_fiber.payloads}")
print("==========================")
assert(z_fiber.payloads == [32, 38, 44])

## Test program: Z-Stationary vector-matrix multiplication
#for n, (z, b_k) in z_n << b_n:
#  for k, (a, b) in a_k & b_k:
#    z += a * b
print("\n*** Z-stationary vector-matrix ***")
a = SwoopTensor(name="A", rank_ids=["K"])
b = SwoopTensor(name="B", rank_ids=["N", "K"])
z = SwoopTensor(name="Z", rank_ids=["N"])

a_k = a["K"]
b_n = b["N"]
b_k = b["K"]
z_n = z["N"]

# z_n << b_n
b_n_handles = Iterate(b_n)
b_n_coords = HandlesToCoords(b_n, b_n_handles)
b_n_payloads = HandlesToPayloads(b_n, b_n_handles)
z_n_coords = b_n_coords
z_n_handles = InsertElement(z_n, z_n_coords)
z_n_payloads = HandlesToPayloads(z_n, z_n_handles)
b_k_n_fiber_handles = PayloadsToFiberHandles(b_n, b_n_payloads)


# a_k & b_k
b_k_handles = Scan(b_k, b_k_n_fiber_handles)
# Repeat a_k iteration for each b_k
a_k_fiber_handle = Stream0(a_k.getRootFiberHandle())
a_k_fiber_handles = Amplify(a_k_fiber_handle, b_k_n_fiber_handles)
a_k_handles = Scan(a_k, a_k_fiber_handles) 
a_k_coords = HandlesToCoords(a_k, a_k_handles)
b_k_coords = HandlesToCoords(b_k, b_k_handles)
(ab_k_coords, ab_a_k_handles, ab_b_k_handles) = Intersect(a_k_coords, a_k_handles, b_k_coords, b_k_handles)
ab_a_k_payloads = HandlesToPayloads(a_k, ab_a_k_handles)
ab_b_k_payloads = HandlesToPayloads(b_k, ab_b_k_handles)


# z_ref += a_val * b_val
a_values = PayloadsToValues(a_k, ab_a_k_payloads)
b_values = PayloadsToValues(b_k, ab_b_k_payloads)
# NOTE: MUL and ADD broken out for efficiency
body_func = lambda a_val, b_val: a_val * b_val
partial_products = Compute(body_func, a_values, b_values)
z_values = z_n_payloads
# Reduce into the same value until end of rank
z_new_values = Reduce(z_values, partial_products)
z_n_update_acks = UpdatePayloads(z_n, z_n_handles, z_new_values)

# convert to HFA
A_HFA = Tensor.fromUncompressed(["K"], A_data, name = "A")
B_HFA = Tensor.fromUncompressed(["N", "K"], B_data, name = "B")
Z_HFA = Tensor.fromUncompressed(["N"], Z_data, name = "Z")

A_tensor = encodeTensorInFormat(A_HFA, ["C"])
B_tensor = encodeTensorInFormat(B_HFA, ["U", "C"])
Z_tensor = encodeTensorInFormat(Z_HFA, ["U"])
# TODO: name the fibers (automatically in encode tensor?)

a.setImplementations("K", A_tensor[0])
b.setImplementations("N", B_tensor[0])
b.setImplementations("K", B_tensor[1])
z.setImplementations("N", Z_tensor[0])

z_fiber = Z_tensor[0][0]
# print(z_fiber.printFiber())
evaluate(z_n_update_acks)

# dump stats
stats_dict = dict()
dumpAllStatsFromTensor(A_tensor, stats_dict)
dumpAllStatsFromTensor(B_tensor, stats_dict)
dumpAllStatsFromTensor(Z_tensor, stats_dict)
print("\nZ-stationary vector-matrix: A = <C>, B = <U, C>, Z = <U>")
print(yaml.dump(stats_dict))
print()
print("==========================")
print(f"Final Z-Stationary result: {z_fiber.payloads}")
print("==========================")
assert(z_fiber.payloads == [32, 38, 44])

