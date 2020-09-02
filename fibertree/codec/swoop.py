
DEFAULT_TRACE_LEVEL = 0 # 3
# 
# NoTransmit
#
# An alternative to None that indicates no transmission on a field in a struct.

class NoTransmit:
  pass


#
# SwoopTensor
#
# This is a format-independent abstraction for a SwoopTensor that contains abstract
# "Rank"s that can be later set to have a concrete format and/or data.
#


class SwoopTensor:
  def __init__(self, name, rank_ids):
    assert(len(rank_ids) >= 0)
    # All tensors have a root rank.
    my_rank_ids = rank_ids[:]
    my_rank_ids.insert(0, "root")
    self.ranks = {}
    for (n, r) in enumerate(my_rank_ids):
      self.ranks[r] = Rank(r, self, n)
    self.name = name
    self.rank_ids = my_rank_ids
  
  def __getitem__(self, rank_id):
    return self.ranks[rank_id]

  def setImplementations(self, rank_id, imps):
    self.ranks[rank_id].setImplementations(imps)
  
  def getRootHandle(self):
    return Stream0(FiberHandle(self.ranks["root"], 0), instance_name=self.name + "_root")
  
  def getStartHandle(self):
    r_1 = self.getRankByIndex(1)
    return Stream0(FiberHandle(r_1, 0), instance_name=self.name + "_" + r_1.name)
  
  def getRankByIndex(self, idx):
    return self.ranks[self.rank_ids[idx]]

#
# Rank
#
# This is a format-independent abstraction for a Rank that dispatches
# method calls to a format-specific version. The reason this class
# exists is that it lets us define swoop programs first, then fill in
# the formats as a later step (using setImplementation()).
#


class Rank:
  def __init__(self, name, tensor, rank_index):
    self.implementations = []
    self.name = name
    self.tensor = tensor
    self.rank_index = rank_index

  def setImplementations(self, imps):
    self.implementations = imps

  def setupSlice(self, fiber_idx, base_coord = 0, bound_coord = None, max_num = None):
    # print("{}_{}:: implementations {}, fiber idx {}".format(self.tensor.name, self.name, self.implementations, fiber_idx))
    self.implementations[fiber_idx].setupSlice(base_coord, bound_coord, max_num)
    return fiber_idx

  def nextInSlice(self, fiber_idx):
    return self.implementations[fiber_idx].nextInSlice()
  
  def handleToCoord(self, fiber_idx, handle):
    return self.implementations[fiber_idx].handleToCoord(handle)
  
  def handleToPayload(self, fiber_idx, handle):
    return self.implementations[fiber_idx].handleToPayload(handle)
  
  def payloadToFiberHandle(self, fiber_idx, payload):
    next_rank_index = self.tensor.getRankByIndex(self.rank_index + 1)
    return FiberHandle(next_rank_index, self.implementations[fiber_idx].payloadToFiberHandle(payload))

  def payloadToValue(self, fiber_idx, payload):
    return self.implementations[fiber_idx].payloadToValue(payload)
    
  def coordToHandle(self, fiber_idx, coord):
    return self.implementations[fiber_idx].coordToHandle(coord)
  
  def insertElement(self, fiber_idx, coord):
    print(self.implementations)
    return self.implementations[fiber_idx].insertElement(coord)
  
  def updatePayload(self, fiber_idx, handle, payload):
    return self.implementations[fiber_idx].updatePayload(handle, payload)
  
  def getUpdatedFiberHandle(self, fiber_idx):
    return self.implementations[fiber_idx].getUpdatedFiberHandle()

  def fiberHandleToPayload(self, fiber_idx):
    return self.implementations[fiber_idx].fiberHandleToPayload()

  def valueToPayload(self, fiber_idx):
    return self.implementations[fiber_idx].valueToPayload()

  def dumpStats(self, stats_dict):
    for impl in self.implementations:
      impl.dumpStats(stats_dict)
      
  def __str__(self):
    return self.tensor.name + "_" + self.name

#
# FiberHandle
#
# A convenience class for a fiber index that also has a reference to the rank
# that the fiber comes from. Just dispatches calls into the rank.

class FiberHandle:
  def __init__(self, rank, pos):
    self.rank = rank
    self.position = pos
    
  def __str__(self):
    return str(self.rank) + "[" + str(self.position) + "]"
    
  def setupSlice(self, base_coord = 0, bound_coord = None, max_num = None):
    return self.rank.setupSlice(self.position, base_coord, bound_coord, max_num)

  def nextInSlice(self):
    return self.rank.nextInSlice(self.position)
  
  def handleToCoord(self, handle):
    return self.rank.handleToCoord(self.position, handle)
  
  def handleToPayload(self, handle):
    return self.rank.handleToPayload(self.position, handle)
  
  def payloadToFiberHandle(self, payload):
    return self.rank.payloadToFiberHandle(self.position, payload)

  def payloadToValue(self, payload):
    return self.rank.payloadToValue(self.position, payload)
    
  def coordToHandle(self, coord):
    return self.rank.coordToHandle(self.position, coord)
  
  def insertElement(self, coord):
    return self.rank.insertElement(self.position, coord)
  
  def updatePayload(self, handle, payload):
    return self.rank.updatePayload(self.position, handle, payload)
  
  def getUpdatedFiberHandle(self):
    return self.rank.getUpdatedFiberHandle(self.position)

  def fiberHandleToPayload(self, fiber_handle):
    return self.rank.fiberHandleToPayload(self.position, fiber_handle)

  def valueToPayload(self, value):
    return self.rank.valueToPayload(self.position, value)

  def dumpStats(self, stats_dict):
    return self.rank.dumpStats(stats_dict)

#
# AST
#
# Base Class for all AST nodes
# Most have a reference to a stream of Fiber Handles. 
# Additionally, track the "fanout" number which is the number of receivers. 
# If this remains 0, this Node is unconnected 
# and can be eliminated as dead code.
#
class AST:

  def __init__(self, class_name, fiber_handles = None, num_fields = 1):
    self.class_name = class_name
    self.num_fields = num_fields
    self.num_fanout = 0
    self.cur_results = {} # dictionary of FIFOs, only used if num_fanout > 1
    self.producers = []
    self.initialized = False
    self.finalized = False
    self.trace_level = DEFAULT_TRACE_LEVEL
    self.current_fiber = None
    if fiber_handles is not None:
      self.fiber_handles = fiber_handles
      fiber_handles.connect(self)
  
  def _addProducer(self, other):
    self.producers.append(other)
  
  def connect(self, other):
    other._addProducer(self)
    self.cur_results[other] = []
    self.num_fanout += 1

  def initialize(self):
    self.initialized = True
    for prod in self.producers:
      if not prod.initialized:
        prod.initialize()
    # Get the initial fiber handle in place for evaulation.
    if (hasattr(self, "fiber_handles") and self.current_fiber is None):
      self.nextFiber()
    
  def evaluate(self):
    self.trace("Unimplemented Evaluate")
    return None
  
  def finalize(self, stats_dict):
    self.dumpStats(stats_dict)
    self.finalized = True
    for prod in self.producers:
      # print("producer {}, finalized {}".format(prod, prod.finalized))
      # if not prod.finalized:
      prod.finalize(stats_dict)
  
  def nextFiber(self):
    assert(hasattr(self, "fiber_handles") and self.fiber_handles is not None)
    prev_fiber = self.current_fiber
    self.current_fiber = self.fiber_handles.nextValue(self)
    
    self.trace(3, "Prev fiber handle {}, New fiber handle:{}".format(prev_fiber, self.current_fiber))

  
  def nextValue(self, other):
    if other is not None:
      res_q = self.cur_results[other]
      # If we have queue'd up a value because of fanout, just use it.
      if len(res_q) > 0:
        self.trace(4, f"Fanout: {other} => {res_q[0]}")
        return res_q.pop(0) # Important: We need 0 here, otherwise it will skip None

    # Proceed with normal evaluation.
    # If it has a current_fiber, if that is None, then we advance it.
    if (hasattr(self, "fiber_handles") and self.current_fiber is None):
      # Just pop all producers (should also be None) and return None.
      self.trace(3, f"Eval: Passthrough empty fiber handle.")
      for prod in self.producers:
        if prod != self.fiber_handles:
          res = prod.nextValue(self)
          if (res is not None):
            self.trace(0, f"Expected None from producer: {prod.getName()} => {res}")
          # print("\tcurrent fiber {}, res: {}".format(self.current_fiber, res))
          assert(res is None)
      # Return none and try again next time.
      for (caller, q) in self.cur_results.items():
        if caller != other:
            q.append(None)
      self.nextFiber()
      return None
    # At this point, child can expect current_fiber is not None (if used)
    # Call evaluate, but only once and fan out the result to later callers.
    self.trace(4, f"Eval {other}")
    res = self.evaluate()
    for (caller, q) in self.cur_results.items():
      if caller != other:
          q.append(res)
    self.trace(4, f"Eval: {other} => {res}")
    return res

  def trace(self, level, args):
    if (level > self.trace_level):
      return
    print(self.getName() + ":", args)

  def setTraceLevel(self, level):
    self.trace_level = level

  def getName(self):
    if (hasattr(self, "fiber_handles")):
      return self.class_name + ":" + str(self.current_fiber)
    else:
      return self.class_name

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
    
  def __getitem__(self, n):
    assert(self.num_fields != 0)
    assert(n < self.num_fields)
    sp = Splitter(self, n)
    return sp
  
  def dumpStats(self, stats_dict):
    # print("dumpStats2 {}".format(self.class_name))
    if (hasattr(self, "accesses")): # and self.current_fiber is None):
      stats_dict[self.class_name] = self.accesses
    # pass

#
# Slice
#
# Given a fiber handle, and a slice spec returns a rank-1 stream of all handles
# to elements in that slice.
# 

class Slice (AST):
  def __init__(self, fiber_handle, base = 0, bound  = None, max_num = None):
    super().__init__("Slice", rank)
    self.fiber_handle = fiber_handle
    self.base = base
    self.bound = bound
    self.max_num = max_num
  
  
  def initialize(self):
    self.trace(3, f"SetupSlice: {self.base}, {self.bound}, {self.max_num}")
    self.fiber_handle.setupSlice(self.base, self.bound, self.max_num)
    super().initialize()
  
  def evaluate(self):
    res = fiber_handle.nextInSlice()
    self.trace(2, f"NextInSlice: {res}")
    return res
  
#
# Scan
#
# Given a N-stream of fiber_handles,
#  returns a N+1-stream of all handles to elements in those fibers.
# 

class Scan (AST):
  def __init__(self, fiber_handles):
    super().__init__("Scan", fiber_handles)
    self.active = False
  
  def evaluate(self):
    
    if not self.active:
      self.trace(3, "Fiber Slice Setup {}".format(self.current_fiber))
      self.current_fiber.setupSlice()
    
    res = self.current_fiber.nextInSlice()
    if res is None and self.active:
      self.trace(3, "Fiber Done.")
      self.nextFiber()
      self.active = False
      return None
    self.active = True
    self.trace(2, f"Next: {res}")
    # if res is None:
    if res is None:
      self.trace(3, "after next, Fiber Done.")
      self.nextFiber()
      self.active = False
      return None

    return res


#
# InsertionScan
#
# Given an N-stream of fiber_handles, and a N+1-stream of 
# coords, returns a N+1-stream of handles to elements of those coords in 
# those fibers. Also returns a N-stream of updated fiber handles.
# 

class InsertionScan (AST):
  def __init__(self, fiber_handles, coords):
    super().__init__("InsertionScan", fiber_handles, num_fields=2)
    self.coords = coords
    coords.connect(self)
    self.active = False
  
  def evaluate(self):
#    if not self.active:
#      self.updateCurrentFiber()
#      if self.current_fiber is None:     
#        coord = self.coords.nextValue(self)
#        assert(coord is None)
#        self.trace(3, "Done.")
#        return (None, None)
#      self.trace(2, f"Start Fiber {self.current_fiber.position}")
#      self.active = True

    coord = self.coords.nextValue(self)
    print("\tInsertionScan: coord {}, active {}, current fiber {}".format(coord, self.active, self.current_fiber))
    if coord is None:
      if self.active:
        new_handle = self.current_fiber.getUpdatedFiberHandle()
        self.trace(3, f"Fiber Done. New Handle: {new_handle}")
      else:
        new_handle = None
      self.active = False
      self.nextFiber()
      assert (not new_handle is None or coord is None)
      
      print("\t\tcoord {}, newHandle: {}".format(coord, new_handle))
      return (None, new_handle)
    self.active = True
    self.trace(2, f"Inserting: {coord}")
    handle = self.current_fiber.insertElement(coord)
    return (handle, NoTransmit)



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
# Given a reference to a N-1 stream of fiber handles, and an AST Node that 
# that produces a N-stream of handles, produces a N-stream of coordinates
#
class HandlesToCoords (AST):

  def __init__(self, fiber_handles, handles):
    super().__init__("HandlesToCoords", fiber_handles)
    self.handles = handles
    handles.connect(self)

  def evaluate(self):
    handle = self.handles.nextValue(self)
    if handle is None:
      self.trace(3, "None")
      self.nextFiber()
      return None
    coord = self.current_fiber.handleToCoord(handle)
    self.trace(2, f"{handle} => {coord}")
    return coord
    
#
# HandlesToPayloads
#
# Given a reference to an AST Rank, and an AST Node that 
# that produces a N-stream of handles, produces a N-stream of payloads
#
class HandlesToPayloads (AST):

  def __init__(self, fiber_handles, handles):
    super().__init__("HandlesToPayloads", fiber_handles)
    self.handles = handles
    handles.connect(self)

  def evaluate(self):
    handle = self.handles.nextValue(self)
    if handle is None:
      self.trace(3, "None")
      self.nextFiber()
      return None
    payload = self.current_fiber.handleToPayload(handle)
    self.trace(2, f"{handle} => {payload}")
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
# Given a reference to an N-1 stream of fiber handles, and an AST Node that 
# that produces a N-stream of payloads, produces a N-stream of Fiber Handles
#
class PayloadsToFiberHandles (AST):

  def __init__(self, fiber_handles, payloads):
    super().__init__("PayloadsToFiberHandles", fiber_handles)
    self.payloads = payloads
    payloads.connect(self)

  def evaluate(self):
    payload = self.payloads.nextValue(self)
    print("\tcurrent fiber in PayloadToFiberHandle {}".format(self.current_fiber))
    if payload is None:
      self.trace(3, "None")
      self.nextFiber()
      return None
    fiber_handle = self.current_fiber.payloadToFiberHandle(payload)
    self.trace(2, f"{payload} => {fiber_handle}")
    return fiber_handle

#
# PayloadsToValues
#
# Given a reference to an N-1 stream of fiber handles, and an AST Node that 
# that produces a N-stream of payloads, produces a N-stream of Values
#
class PayloadsToValues (AST):

  def __init__(self, fiber_handles, payloads):
    super().__init__("PayloadsToValues", fiber_handles)
    self.payloads = payloads
    payloads.connect(self)

  def evaluate(self):
    payload = self.payloads.nextValue(self)
    if payload is None:
      self.trace(3, "None")
      self.nextFiber()
      return None
    value = self.current_fiber.payloadToValue(payload)
    self.trace(2, f"{payload} => {value}")
    return value



#
# CoordsToHandles
#
# Given a reference to an N-1 stream of fiber handles, and an AST Node that 
# that produces a N-stream of coords, produces a N-stream of handles
# (NOTE: EXPENSIVE FOR MOST FORMATS)
# TODO: Add starting position.
#
class CoordsToHandles (AST):

  def __init__(self, fiber_handles, coords):
    super().__init__("CoordsToHandles", fiber_handles)
    self.coords = coords
    coords.connect(self)

  def evaluate(self):
    coord = self.coords.nextValue(self)
    if coord is None:
      self.trace(3, "None")
      self.nextFiber()
      return None
    handle = self.current_fiber.coordToHandle(coord)
    self.trace(2, f"{coord} => {handle}")
    return handle

#
# InsertElements
#
# Given a reference to an N-1 stream of fiber handles, and an AST Node that 
# that produces a N-stream of coords, produces a N-stream of handles
# after creating that (coord, payload) element and initializing coord
# (NOTE: EXPENSIVE FOR MOST FORMATS)
# TODO: Add starting position.
#

class InsertElements (AST):

  def __init__(self, fiber_handles, coords):
    super().__init__("InsertElements", fiber_handles, num_fields=2)
    self.coords = coords
    coords.connect(self)

  def evaluate(self):
    coord = self.coords.nextValue(self)
    if coord is None:
      new_handle = self.current_fiber.getUpdatedFiberHandle()
      self.trace(3, f"Fiber done. New Handle: {new_handle}")
      self.nextFiber()
      return (None, new_handle)
    handle = self.current_fiber.insertElement(coord)
    self.trace(2, f"{coord} => {handle}")
    return (handle, NoTransmit)

#
# UpdatePayloads
#
# Given a reference to an N-1 stream of fiber handles, and an AST Node that produces
# a N-stream of handles, and an AST Node that produces a N-stream
# of payloads, updates each element (coord, payload) to the new payload.
#

class UpdatePayloads (AST):

  def __init__(self, fiber_handles, handles, payloads):
    super().__init__("UpdatePayloads", fiber_handles)
    self.handles = handles
    handles.connect(self)
    self.payloads = payloads
    payloads.connect(self)

  def evaluate(self):
    handle = self.handles.nextValue(self)
    payload = self.payloads.nextValue(self)
    # if handle is None and payload is None:
    if handle is None or payload is None:
      self.trace(2, f"{handle} => {payload}")
      assert (handle is None and payload is None)
      self.trace(3, "Done.")
      self.nextFiber()
      return None
    self.trace(2, f"{handle} => {payload}")
    return self.current_fiber.updatePayload(handle, payload)


# 
# Intersect
# 
#

class Intersect (AST):
  def __init__(self, a_coords, a_handles, b_coords, b_handles, instance_name=None):
    # Note: we return a 3-tuple, so tell the super-class that.
    name = "Intersect"
    if instance_name is not None:
      name = name + "_" + instance_name
    super().__init__(name, num_fields=3)
    self.a_coords = a_coords
    a_coords.connect(self)
    self.a_handles = a_handles
    a_handles.connect(self)
    self.b_coords = b_coords
    b_coords.connect(self)
    self.b_handles = b_handles
    b_handles.connect(self)

  def evaluate(self):
    a_coord = -2
    b_coord = -1
    a_handle = None
    b_handle = None
    while a_coord != None and b_coord != None:
      if a_coord == b_coord:
        self.trace(2, f"Intersection found at: {a_coord}: ({a_handle}, {b_handle})")
        return (a_coord, a_handle, b_handle)
      while a_coord != None and b_coord != None and a_coord < b_coord:
        a_coord = self.a_coords.nextValue(self)
        a_handle = self.a_handles.nextValue(self)
        self.trace(3, f"Advancing A: {a_coord}, {b_coord} ({a_handle}, {b_handle})")        
      while b_coord != None and a_coord != None and b_coord < a_coord:
        b_coord = self.b_coords.nextValue(self)
        b_handle = self.b_handles.nextValue(self)
        self.trace(3, f"Advancing B: {a_coord}, {b_coord} ({a_handle}, {b_handle})")
      # If one ended, drain the other
      if a_coord is None:
        while b_coord is not None:
          b_coord = self.b_coords.nextValue(self)
          b_handle = self.b_handles.nextValue(self)
          self.trace(3, f"Draining B: {b_coord} ({b_handle})")
        self.trace(3, "Done.")
        return (None, None, None)
      elif b_coord is None:
        while a_coord is not None:
          a_coord = self.a_coords.nextValue(self)
          a_handle = self.a_handles.nextValue(self)
          self.trace(3, f"Draining A: {a_coord} ({a_handle})")
        self.trace(3, "Done.")
        return (None, None, None)

    self.trace(3, "Done.")
    return (None, None, None)
  

#
# Splitter
#
# Helper module for splitting tuple streams.
# Note: explicitly over-rides default fanout behavior.
#

class Splitter (AST):
  def __init__(self, stream, num):
    super().__init__("Splitter(" + stream.class_name + ")[" + str(num) + "]")
    self.stream = stream
    stream.connect(self)
    self.num = num

  def evaluate(self):
    my_field = NoTransmit
    while my_field is NoTransmit:
      res = NoTransmit
      while res is NoTransmit:
        res = self.stream.nextValue(self)
      if res is None:
        my_field = None
      else:
        my_field = res[self.num]
    self.trace(3, f"{self.num} => {my_field}")
    return my_field

#
# Compute
#
# Given an N-argument function and a list of N AST nodes that produce
# N-streams of values, apply the function to the values to produce an N-stream
# of outputs
#

class Compute (AST):
  def __init__(self, function, *streams, instance_name=None):
    name = "Compute"
    if instance_name is not None:
      name += "_" + instance_name
    super().__init__(name)
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
      assert(arg is not NoTransmit)
      is_none = arg is None
      any_is_none |= is_none
      all_are_none &= is_none

    if (any_is_none and not all_are_none):
      for arg in args:
        self.trace(0, f"Inconsistent None: {arg}")
    assert(not any_is_none or all_are_none)
    
    if all_are_none:
      self.trace(3, "None")
      return None
      
    result = self.function(*args)
    self.trace(1, f"({args}) => {result}")
    return result

#
# Amplify
#
# Given an AST node that produces an N-stream, and a Node that produces an
# N+1-stream, replicate each element from the N-stream, so that the output
# is an N+1-stream.
#

class Amplify (AST):
  def __init__(self, smaller, bigger, instance_name = None):
    name = "Amplify"
    if instance_name is not None:
      name += "_" + instance_name
    super().__init__(name)
    self.smaller = smaller
    smaller.connect(self)
    self.bigger = bigger
    bigger.connect(self)
    self.current_value = None
    self.accesses = 0

  def evaluate(self):
    if self.current_value is None: # Is this initialization condition correct?  
      self.current_value = self.smaller.nextValue(self)
      if self.current_value is None:
        res = self.bigger.nextValue(self)
        if res is not None:
          self.trace(0, f"Inconsitent None: {res}")
        assert(res is None)
        self.trace(3, "None")
        return None
      self.trace(3, f"{self.smaller.class_name}: Init: {self.current_value}")
      
    next = self.bigger.nextValue(self)
    if next is None:
      self.current_value = None
      self.trace(3, f"{self.smaller.class_name}: Done")
    else:
        # increment stat for buffer access to smaller
        self.accesses += 1
    self.trace(2, f"{self.smaller.class_name}: {next} => {self.current_value}")
    return self.current_value

    @override
    def dumpStats(self, stats_dict):
      print("dumpStats {}".format(self.class_name))
      stats_dict[self.class_name] = self.accesses

#
# Reduce
#
# Given an AST node that produces an N-stream, and a Node that produces an
# N-1-stream, reduce each element from the N-stream, so that the output
# is an N-1-stream.
#

class Reduce (AST):
  def __init__(self, bigger, smaller = None, instance_name = None):
    name = "Reduce"
    if instance_name is not None:
      name += "_" + instance_name
    super().__init__(name)
    self.smaller = smaller
    if smaller is not None:
      smaller.connect(self)
    self.bigger = bigger
    bigger.connect(self)
    self.accesses = 0
  def evaluate(self):
    if self.smaller is not None:
      current_value = self.smaller.nextValue(self)
      if current_value is None:
        next = self.bigger.nextValue(self)
        if next is not None:
          self.trace(0, f"Inconsitent None: {current_value} => {next}")
        assert(next is None)
        self.trace(3, "Init: None")
        return None
      self.trace(3, f"Init: {current_value}")
    else:
      current_value = None
    
    next = 0
    while next is not None:
      next = self.bigger.nextValue(self)
      if next is not None:
        if current_value is None:
          current_value = 0
          self.trace(3, f"Init: 0")
        self.trace(2, f"{current_value} + {next} => {current_value + next}")
        current_value += next
        self.accesses += 1
        # increment a stat for smaller (thing being reduced into)
    self.trace(3, f"Output: {current_value}")
    return current_value

  def dumpStats(self, stats_dict):
    print("dumpStats {}".format(self.class_name))
    stats_dict[self.class_name] = self.accesses


#
# Stream0
#
# Turn a scalar into a 0-stream that transmits exactly that scalar.
#


class Stream0 (AST):
  def __init__(self, val, instance_name=None):
    name = "Stream0"
    if instance_name is not None:
      name += "_" + instance_name
    super().__init__(name)
    self.val = val
    self.done = False
  
  
  def evaluate(self):
    #assert(not self.done)
    if self.done:
      self.trace(3, "None")
      return None
    self.done = True
    self.trace(3, f"{self.val}")
    return self.val

#
# Distribute
#
# Route a N-stream down one of M routes based on a distribution choice N-stream.
#

class Distribute (AST):
  def __init__(self, N, distribution_choices, stream):
    super().__init__("Distribute", num_fields=N)
    self.N = N
    self.distribution_choices = distribution_choices
    distribution_choices.connect(self)
    self.stream = stream
    stream.connect(self)
  
  def evaluate(self):
    choice = self.distribution_choices.nextValue(self)
    val = self.stream.nextValue(self)
    if choice is None:
      assert(val is None)
      self.trace(3, "None.")
      return [None] * self.N
    assert(choice < self.N)
    res = [NoTransmit] * self.N
    res[choice] = val
    self.trace(3, f"{val} => {choice}")
    return res

#
# Collect
#
# Route one of M N-streams together into an N-Stream based on a 
# distribution choice N-stream. Usually used to undo a Distribute by
# passing the same distribution_choice stream to both.
#

class Collect (AST):
  def __init__(self, N, distribution_choices, stream_array):
    super().__init__("Collect")
    self.N = N
    self.distribution_choices = distribution_choices
    distribution_choices.connect(self)
    self.stream_array = stream_array
    for n in range(self.N):
      stream_array[n].connect(self)
  
  def evaluate(self):
    choice = self.distribution_choices.nextValue(self)
    if choice is None:
      self.trace(3, "None")
      return None
    assert(choice < self.N)
    val = self.stream_array[choice].nextValue(self)
    self.trace(3, f"{choice} => {val}")
    return val

#
# BasicIntermediateRankImplementation
#
# Rank implementation JUST to test out the program below.

class BasicIntermediateRankImplementation:
  def __init__(self, shape, shape_of_next_rank, pos=0):
    self.shape = shape
    self.shape_of_next_rank = shape_of_next_rank
    self.pos = pos
  
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
    assert(handle < self.shape_of_next_rank)
    return (self.pos * self.shape) + handle
  
  def payloadToFiberHandle(self, payload):
    return payload
  
  def payloadToValue(self, payload):
    assert(False)
  
  def coordToHandle(self, coord):
    return coord
  
  def insertElement(self, coord):
    return coord
  
  def updatePayload(self, handle, payload):
    return handle
  
  def getUpdatedFiberHandle(self):
    return self.shape

  def fiberHandleToPayload(self, fiber_handle):
    return fiber_handle

  def valueToPayload(self, value):
    assert(False)


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
    if coord >= len(self.vals):
      self.vals.append(0)
    return coord
  
  def updatePayload(self, handle, payload):
    assert(handle is not None)
    self.vals[handle] = payload
    return handle
  
  def getUpdatedFiberHandle(self):
    return len(self.vals)

  def fiberHandleToPayload(self, fiber_handle):
    assert(False)

  def valueToPayload(self, value):
    return value


  def dumpStats(self, stats_dict):
    pass

#
# evaluate
#
# Run the given node (and all nodes connected to it) until it returns None
# N times in a row.
#


def evaluate(node, n = 1, stats_dict = {}):
  assert(n >= 0)
  node.initialize()
  consecutive_dones = -1
  while (consecutive_dones != n):
    res = node.nextValue(None)
    print(f"+++++++++")
    print(f"Evaluate: {res}")
    print(f"+++++++++")
    if res is None:
      consecutive_dones += 1
    else:
      consecutive_dones = 0
  node.finalize(stats_dict)


if __name__ == "__main__":


  ## Test program: Element-wise multiplication
  #
  #
  # Z_k = A_k * B_k
  #
  #
  # a_k = A.getRoot()
  # b_k = B.getRoot()
  # z_k = Z.getRoot()
  #
  # for k, (z, (a, b)) in z_k << (a_k & b_k):
  #   z <<= a * b

  # Define the tensors
  a = SwoopTensor(name = "A", rank_ids = ["K"])
  b = SwoopTensor(name = "B", rank_ids = ["K"])
  z = SwoopTensor(name = "Z", rank_ids = ["K"])

  # Get handles to the tree start.
  a_k = a.getStartHandle() # GetStartingFiber(a)
  b_k = b.getStartHandle() # GetStartingFiber(b)
  z_root = z.getRootHandle()
  z_k = z.getStartHandle() # GetStartingFiber(z)

  # Iterate the K rank and get handles to contents
  a_handles = Scan(a_k)
  b_handles = Scan(b_k)
  # Convert handles to coordinates
  a_coords = HandlesToCoords(a_k, a_handles)
  b_coords = HandlesToCoords(b_k, b_handles)
  # Intersect the K rank
  (ab_coords, ab_a_handles, ab_b_handles) = Intersect(a_coords, a_handles, b_coords, b_handles)
  # Only insert elements that survive intersection
  (z_handles, z_k_new_fiber_handle) = InsertionScan(z_k, ab_coords)
  # Only retrieve the values that survive intersection
  a_payloads = HandlesToPayloads(a_k, ab_a_handles)
  b_payloads = HandlesToPayloads(b_k, ab_b_handles)
  a_values = PayloadsToValues(a_k, a_payloads)
  b_values = PayloadsToValues(b_k, b_payloads)
  # Calculate the loop body
  results = Compute(lambda a, b: a*b, a_values, b_values)
  # Final writeback
  z_k_update_acks = UpdatePayloads(z_k, z_handles, results)
  
  # Update final occupancies.
  z_root_update_acks = UpdatePayloads(z_root, Stream0(0), z_k_new_fiber_handle)

  # Create some example implmentations
  my_a_root = BasicIntermediateRankImplementation(1, 1)
  my_a_K = BasicFiberImplementation([1, 2, 3])
  my_b_root = BasicIntermediateRankImplementation(1, 1)
  my_b_K = BasicFiberImplementation([4, 5, 6])
  my_z_root = BasicIntermediateRankImplementation(1, 1)
  my_z_K = BasicFiberImplementation([])

  # Use those implementations in practice
  a.setImplementations("root", [my_a_root])
  a.setImplementations("K", [my_a_K])
  b.setImplementations("root", [my_b_root])
  b.setImplementations("K", [my_b_K])
  z.setImplementations("root", [my_z_root])
  z.setImplementations("K", [my_z_K])

  # Run the program and check and print the result
  evaluate(z_k_update_acks)
  evaluate(z_root_update_acks, 0)
  print("===========================")
  print(f"Final element-wise result: {my_z_K.vals}")
  print("===========================")
  assert(my_z_K.vals == [4, 10, 18])



  ## Test program: A-Stationary vector-matrix multiplication
  #
  # Z_n = A_k * B_kn
  #
  #
  # for k, (a, b_n) in a_k & b_k:
  #   for n, (z, b) in z_n << b_n:
  #     z += a * b

  a = SwoopTensor(name="A", rank_ids=["K"])
  b = SwoopTensor(name="B", rank_ids=["K", "N"])
  z = SwoopTensor(name="Z", rank_ids=["N"])

  # Get handles to the tree start.
  a_k = a.getStartHandle()
  b_k = b.getStartHandle()
  z_root = z.getRootHandle()
  z_n = z.getStartHandle()

  # a_k & b_k
  a_k_handles = Scan(a_k)
  b_k_handles = Scan(b_k)
  a_k_coords = HandlesToCoords(a_k, a_k_handles)
  b_k_coords = HandlesToCoords(b_k, b_k_handles)
  (ab_k_coords, ab_a_k_handles, ab_b_k_handles) = Intersect(a_k_coords, a_k_handles, b_k_coords, b_k_handles)
  ab_a_k_payloads = HandlesToPayloads(a_k, ab_a_k_handles)
  ab_b_k_payloads = HandlesToPayloads(b_k, ab_b_k_handles)
  b_ns = PayloadsToFiberHandles(b_k, ab_b_k_payloads)

  # z_n << b_n
  b_n_handless = Scan(b_ns)
  b_n_coordss = HandlesToCoords(b_ns, b_n_handless)
  b_n_payloadss = HandlesToPayloads(b_ns, b_n_handless)
  z_ns = Amplify(z_n, ab_k_coords)
  (z_n_handless, z_n_updated_fiber_handles) = InsertionScan(z_ns, b_n_coordss)
  z_n_payloadss = HandlesToPayloads(z_ns, z_n_handless)

  # z_ref += a_val * b_val
  a_values = PayloadsToValues(a_k, ab_a_k_payloads)
  b_valuess = PayloadsToValues(b_ns, b_n_payloadss)
  z_valuess = PayloadsToValues(z_ns, z_n_payloadss)
  # We need to repeat A value across every Z
  a_valuess = Amplify(a_values, b_n_coordss)
  # Actual computation
  body_func = lambda a_val, b_val, z_ref: z_ref + a_val * b_val
  z_new_valuess = Compute(body_func, a_valuess, b_valuess, z_valuess)
  # Final write-back
  z_n_update_ackss = UpdatePayloads(z_ns, z_n_handless, z_new_valuess)

  # Record occupancy
  z_root_handles = Amplify(Stream0(0), ab_k_coords)
  z_root_acks = UpdatePayloads(z_root, z_root_handles, z_n_updated_fiber_handles)

  K=3
  N=3
  my_a_root = BasicIntermediateRankImplementation(1, 1)
  my_a_k = BasicFiberImplementation([1, 2, 3])
  my_b_root = BasicIntermediateRankImplementation(1, 1)
  my_b_k = BasicIntermediateRankImplementation(K, N)
  my_b_n = [
             BasicFiberImplementation([4, 5, 6]), 
             BasicFiberImplementation([5, 6, 7]), 
             BasicFiberImplementation([6, 7, 8])
           ]
  my_z_root = BasicIntermediateRankImplementation(1, 1)
  my_z_n = BasicFiberImplementation([])

  a.setImplementations("root", [my_a_root])
  a.setImplementations("K", [my_a_k])
  b.setImplementations("root", [my_b_root])
  b.setImplementations("K", [my_b_k])
  b.setImplementations("N", my_b_n)
  z.setImplementations("root", [my_z_root])
  z.setImplementations("N", [my_z_n])

  evaluate(z_n_update_ackss, 2)
  evaluate(z_root_acks, 1)
  print("==========================")
  print(f"Final A-Stationary result: {my_z_n.vals}")
  print("==========================")
  assert(my_z_n.vals == [32, 38, 44])
  



  ## Test program: Z-Stationary vector-matrix multiplication
  #
  # Z_n = A_k * B_kn
  #
  #for n, (z, b_k) in z_n << b_n:
  #  for k, (a, b) in a_k & b_k:
  #    z += a * b

  a = SwoopTensor(name="A", rank_ids=["K"])
  b = SwoopTensor(name="B", rank_ids=["N", "K"])
  z = SwoopTensor(name="Z", rank_ids=["N"])

  a_k = a.getStartHandle()
  b_n = b.getStartHandle()
  z_root = z.getRootHandle()
  z_n = z.getStartHandle()

  # z_n << b_n
  b_n_handles = Scan(b_n)
  b_n_coords = HandlesToCoords(b_n, b_n_handles)
  b_n_payloads = HandlesToPayloads(b_n, b_n_handles)
  (z_n_handles, z_n_new_fiber_handle) = InsertionScan(z_n, b_n_coords)
  z_n_payloads = HandlesToPayloads(z_n, z_n_handles)
  b_ks = PayloadsToFiberHandles(b_n, b_n_payloads)


  # a_k & b_k
  b_k_handless = Scan(b_ks)
  # Repeat a_k iteration for each b_k
  a_ks = Amplify(a_k, b_ks)
  a_k_handless = Scan(a_ks) 
  a_k_coordss = HandlesToCoords(a_ks, a_k_handless)
  b_k_coordss = HandlesToCoords(b_ks, b_k_handless)
  (ab_k_coordss, ab_a_k_handless, ab_b_k_handless) = Intersect(a_k_coordss, a_k_handless, b_k_coordss, b_k_handless)
  ab_a_k_payloadss = HandlesToPayloads(a_ks, ab_a_k_handless)
  ab_b_k_payloadss = HandlesToPayloads(b_ks, ab_b_k_handless)


  # z_ref += a_val * b_val
  a_valuess = PayloadsToValues(a_ks, ab_a_k_payloadss)
  b_valuess = PayloadsToValues(b_ks, ab_b_k_payloadss)
  # NOTE: MUL and ADD broken out for efficiency
  body_func = lambda a_val, b_val: a_val * b_val
  partial_productss = Compute(body_func, a_valuess, b_valuess)
  z_values = PayloadsToValues(z_n, z_n_payloads)
  # Reduce into the same value until end of rank
  z_new_values = Reduce(partial_productss, z_values)
  z_n_update_acks = UpdatePayloads(z_n, z_n_handles, z_new_values)
  
  # Update occupancy
  z_root_update_ack = UpdatePayloads(z_root, Stream0(0), z_n_new_fiber_handle)

  my_a_root = BasicIntermediateRankImplementation(1, 1)
  my_a_k = BasicFiberImplementation([1, 2, 3])
  my_b_root = BasicIntermediateRankImplementation(1, 1)
  my_b_n = BasicIntermediateRankImplementation(N, K)
  my_b_k = [BasicFiberImplementation([4, 5, 6]), 
            BasicFiberImplementation([5, 6, 7]), 
            BasicFiberImplementation([6, 7, 8])]
  my_z_root = BasicIntermediateRankImplementation(1, 1)
  my_z_n = BasicFiberImplementation([])


  a.setImplementations("root", [my_a_root])
  a.setImplementations("K", [my_a_k])
  b.setImplementations("root", [my_b_root])
  b.setImplementations("N", [my_b_n])
  b.setImplementations("K", my_b_k)
  z.setImplementations("root", [my_z_root])
  z.setImplementations("N", [my_z_n])

  evaluate(z_n_update_acks)
  evaluate(z_root_update_ack, 0)
  print("==========================")
  print(f"Final Z-Stationary result: {my_z_n.vals}")
  print("==========================")
  assert(my_z_n.vals == [32, 38, 44])
