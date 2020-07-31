"""
CompressionFormat class - can be instantiated to represent a fiber
mostly just here to be inherited
"""
import sys

class CompressionFormat:
    def __init__(self):
        # self.name = ""
        self.coords = []
        self.payloads = []
        self.cur_handle = -1
        self.num_accesses = 0
    # API Methods
    # helpers
    # have to overwrite this in subclasses, depends on the format
    def getSliceMaxLength(self):
        return None

    # main functions
    # given a handle, return a coord at that handle
    # if handle is out of range, return None
    def handleToCoord(self, handle):
        # TODO: make these assertions that it's the correct type and in range
        if handle is None or handle >= len(self.coords):
            return None
        return self.coords[handle]

    # given a handle, return payload there if in range, otherwise None
    def handleToPayload(self, handle):
        if handle is None or  handle >= len(self.payloads):
            return None
        return self.payloads[handle]

    # slice on coordinates
    def setupSlice(self, base = 0, bound = None, max_num = None):
        self.num_ret_so_far = 0
        
        self.num_to_ret = max_num
        self.base = base
        self.bound = bound
        self.start_handle = self.coordToHandle(base)

    # get next handle during iteration through slice
    def nextInSlice(self):
        # print("in next: handle {}, slice max {}, num to ret {}, ret so far {}".format(self.start_handle, self.getSliceMaxLength(), self.num_to_ret, self.num_ret_so_far))
        if self.start_handle >= self.getSliceMaxLength():
            return None
        if self.num_to_ret is not None and self.num_to_ret < self.num_ret_so_far + 1:
            return None
        to_ret = self.start_handle
        self.num_ret_so_far += 1
        self.start_handle += 1

        # if you are accessing at a new handle in C, incur a new access cost
        if self.start_handle != self.cur_handle:
            self.num_accesses += 1 

        return to_ret

    # these need to be filled in in subclasses
    # TODO: python syntax to require that you have to fill this in or assert(false)
    def coordToHandle(self, coord):
        assert(False)

    def insertElement(self, coord):
        assert(False)

    def updatePayload(self, handle, payload):
        return handle

    # at the end of execution, dump stats in YAML
    # add to the stats dict
    def dumpStats(self, stats_dict):
        # key = stats_name + fiber_name
        # stats_dict[key] = list()
        print("num accesses {}".format(self.num_accesses))

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
