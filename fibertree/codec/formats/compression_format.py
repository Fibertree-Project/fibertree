"""
CompressionFormat class - can be instantiated to represent a fiber
mostly just here to be inherited
"""
import sys

class CompressionFormat:
    def __init__(self, name = None):
        # self.name = ""
        self.coords = []
        self.payloads = []
        self.cur_handle = -1
        
        # stats 
        self.stats = dict()
        self.coords_write_key = "num_coords_writes"
        self.stats[self.coords_write_key] = 0
        self.payloads_write_key = "num_payloads_writes"
        self.stats[self.payloads_write_key] = 0
        self.coords_read_key = "num_coords_reads"
        self.stats[self.coords_read_key] = 0
        self.payloads_read_key = "num_payloads_reads"
        self.stats[self.payloads_read_key] = 0
        self.count_payload_reads = True
          
        # cached coord
        self.prevCoordSearched = None
        self.prevHandleAtCoordSearched = None
        self.prevHandleSearched = None
        self.prevCoordAtHandleSearched = None
        self.prevPayloadHandle = None
        self.prevPayloadAtHandle = None

    # return a handle to this payload
    def payloadToFiberHandle(self, payload):
        for i in range(0, len(self.payloads)):
            if payload == self.payloads[i]:
                return i

    def payloadToValue(self, payload):
        return payload
    # API Methods
    # helpers
    # have to overwrite this in subclasses, depends on the format
    def getSliceMaxLength(self):
        return None

    def setName(self, name):
        self.name = name

    # main functions
    # given a handle, return a coord at that handle
    # if handle is out of range, return None
    def handleToCoord(self, handle):
        # TODO: make these assertions that it's the correct type and in range
        if handle is None or handle >= len(self.coords):
            return None
        elif handle is self.prevHandleAtCoordSearched:
            return self.prevCoordSearched
        elif handle is self.prevHandleSearched:
            return self.prevCoordAtHandleSearched
        self.stats[self.coords_read_key] += 1
        
        return self.coords[handle]

    # given a handle, return payload there if in range, otherwise None
    def handleToPayload(self, handle):
        # print("handleToPayload: handle {}, prevPayloadHandle {}, payloads {}".format(handle, self.prevPayloadHandle, self.payloads))
        if handle is None or  handle >= len(self.payloads):
            return None
        elif handle == self.prevPayloadHandle:
            return self.prevPayloadAtHandle
        if self.count_payload_reads:
            self.stats[self.payloads_read_key] += 1
        # print("\thandleToPayload: num accesses {}".format(self.num_accesses))
        self.prevPayloadHandle = handle
        self.prevPayloadAtHandle = self.payloads[handle]
        return self.payloads[handle]

    # slice on coordinates
    def setupSlice(self, base = 0, bound = None, max_num = None):
        self.num_ret_so_far = 0
        self.num_to_ret = max_num
        self.base = base
        self.bound = bound
        # print("setupSlice for {}, base = {}, bound = {}, max_num = {}".format(self.name, base, bound, max_num))
        self.coords_handle = self.coordToHandle(base)

    # get next handle during iteration through slice
    def nextInSlice(self):
        # print("\tin next: handle {}, slice max {}, num to ret {}, ret so far {}".format(self.coords_handle, self.getSliceMaxLength(), self.num_to_ret, self.num_ret_so_far))
        if self.coords_handle >= self.getSliceMaxLength():
            return None
        if self.num_to_ret is not None and self.num_to_ret < self.num_ret_so_far:
            return None
        to_ret = self.coords_handle
        self.num_ret_so_far += 1
        self.coords_handle += 1
        # don't need to increment accesses for moving the handle forward
        return to_ret

    # these need to be filled in in subclasses
    # TODO: python syntax to require that you have to fill this in or assert(false)
    def coordToHandle(self, coord):
        assert(False)

    def insertElement(self, coord):
        assert(False)

    def updatePayload(self, handle, payload):
        return handle
    
    # get size of the representation in words
    # needs to be implemented by subclasses
    def getSize(self):
        assert(False)
    # at the end of execution, dump stats in YAML
    # add to the stats dict
    def dumpStats(self, stats_dict):
        self.stats["size"] = self.getSize() 
        # print("dump stats {}".format(self.name))
        stats_dict[self.name] = self.stats

    def getSize(self):
        assert(False)
    #### class methods
    # e.g. U, C
    @staticmethod 
    def getName(self):
        return self.name

    @staticmethod 
    # current_fiber = HFA fiber
    def encodeCoord(prev_ind, ind):
        return []

    # coord
    @staticmethod
    def encodePayload(prev_ind, ind, payload):
        return None

    # pad end of coordinates if necessary
    @staticmethod
    def endCoords(num_to_pad):
        return [] 

    # pad end of payloads if necessary
    @staticmethod
    def endPayloads(num_to_pad):
        return []

    @staticmethod
    def startOccupancy():
        return 0

    # todo: maybe eventually combine the encode and decode like serialization
