"""
CompressionFormat class
mostly just here to be inherited
"""
import sys

class CompressionFormat:
    def __init__(self):
        # self.name = ""
        self.coords = []
        self.payloads = []

    # API Methods
    # helpers
    # have to overwrite this
    def getSliceMaxLength(self):
        return None
    # must be overriden in inherited classes

    # main functions
    def coordToHandle(self, coord):
        return coord

    # given a handle, return a coord at that handle
    # if handle is out of range, return None
    def handleToCoord(self, handle):
        if handle is None or handle >= len(self.coords):
            return None
        return self.coords[handle]

    # given a handle, return payload there if in range, otherwise None
    def handleToPayload(self, handle):
        if handle is None or  handle >= len(self.payloads):
            return None
        return self.payloads[handle]

    def setupSlice(self, base, bound, max_num = sys.maxsize):
        self.num_ret_so_far = -1
        self.num_to_ret = max_num
        self.base = base
        self.bound = bound
        self.start_handle = self.coordToHandle(base) - 1

    # get next handle during iteration through slice
    def nextInSlice(self):
        # print("in next: handle {}, slice max {}, num to ret {}, ret so far {}".format(self.start_handle, self.getSliceMaxLength(), self.num_to_ret, self.num_ret_so_far))
        if self.start_handle >= self.getSliceMaxLength() or self.num_to_ret < self.num_ret_so_far + 1:
            return None
        to_ret = self.start_handle
        self.num_ret_so_far += 1
        self.start_handle += 1
        return to_ret

    # given a handle, return a coord at that handle
    # if handle is out of range, return None
    def handleToCoord(self, handle):
        if handle is None or handle >= len(self.coords):
            return None
        return self.coords[handle]

    # given a handle, return payload there if in range, otherwise None
    def handleToPayload(self, handle):
        if handle is None or  handle >= len(self.payloads):
            return None
        return self.payloads[handle]

    # these need to be filled in in subclasses
    def coordToHandle(self, coord):
        return None

    def insertElement(self, coord):
        return None

    def updatePayload(self, handle, payload):
        return None


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
