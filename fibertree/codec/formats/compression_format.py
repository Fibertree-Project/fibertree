"""
CompressionFormat class - can be instantiated to represent a fiber
mostly just here to be inherited
"""
import sys
class CompressionFormat:
    def __init__(self, name = None):
        self.coords = list()
        self.payloads = list()
        self.occupancies = list()
        self.cur_handle = -1
        self.idx_in_rank = None
        self.shape = None
        self.words_in_line = 4

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
        self.count_payload_writes = True 

        self.cache = None
        self.next_fmt = None 
    
    # API Methods
    def payloadToFiberHandle(self, payload):
        print("\t{} payloadToFiberHandle:: ret {}".format(self.name, payload))
        return payload

    # default payload to value
    def payloadToValue(self, payload):
        print("\t{}: payloadToValue, payload {}, len payloads {}".format(self.name, payload, len(self.payloads)))
        # self.printFiber()
        if payload >= len(self.payloads):
            return None
        self.stats[self.payloads_read_key] += 1
        
        print("DRAM {} payloadToValue {}, miss count before {}".format(self.name, payload, self.cache.miss_count))
        # TODO: cache line here
        # key = self.name + "_payloadToValue_" + str(payload)
        key = self.name + "_handleToPayload_" + str(payload)
        cached_val = self.cache.get(key) # try to access it
        self.cache[key] = self.payloads[payload] # put it in the cache 
        print("DRAM {} payloadToValue {}, miss count after {}".format(self.name, payload, self.cache.miss_count))
        print(self.cache)

        # read in the cache line
        end_of_range = self.round_up(max(1, payload), self.words_in_line)
        # end_of_range = min(end_of_line, len(self.payloads)) 
        for i in range(payload, end_of_range):
            key = self.name + "_handleToPayload_" + str(i)
            if i < len(self.payloads):
                self.cache[key] = self.payloads[i]
            else:
                self.cache[key] = 0 # end of cache line, so read it as empty
        return self.payloads[payload]
    # helpers
    # have to overwrite this in subclasses, depends on the format
    def getSliceMaxLength(self):
        return None

    def setName(self, name):
        self.name = name

    def round_up(self, n, multiple):
        if n % multiple == 0:
            n += 1
        return ((n + multiple - 1) // multiple) * multiple
    # main functions
    # given a handle, return a coord at that handle
    # if handle is out of range, return None
    def handleToCoord(self, handle):
        # print("\t{} handleToCoord: handle {}, coords {}".format(self.name, handle, self.coords))
        if handle == None or handle >= len(self.coords):
            return None
	
        key = self.name + "_handleToCoord_" + str(handle)
        cached_val = self.cache.get(key)
        self.cache[key] = self.coords[handle]
        # read in a line
        end_of_line = self.round_up(handle, self.words_in_line)
        print("\thandle {}, end of line {}".format(handle, end_of_line))
        end_of_range = min(end_of_line, len(self.coords))
        for i in range(handle, end_of_range):
            key = self.name + "_handleToCoord_" + str(i)
            self.cache[key] = self.coords[i]
            print("\t\t{}, misses {}".format(key, self.cache.miss_count)) 
        print(self.cache)
        # coords read charge
        self.stats[self.coords_read_key] += 1
        
        return self.coords[handle]

    # given a handle, return payload there if in range, otherwise None
    def handleToPayload(self, handle):
        if handle == None or handle >= len(self.payloads):
            return None
        # do stats counting in handleToPayload because it later can go to
        # -> payloadToValue
        # -> payloadToFiberHandle
        if self.count_payload_reads:
            self.stats[self.payloads_read_key] += 1
        print("\t{} handleToPayload {}".format(self.name, handle))
        return handle # switch to just passing around the ptr

    # slice on coordinates
    def setupSlice(self, base = 0, bound = None, max_num = None):
        self.num_ret_so_far = 0
        self.num_to_ret = max_num
        self.base = base
        self.bound = bound
        # print("setupSlice for {}, base = {}, bound = {}, max_num = {}".format(self.name, base, bound, max_num))
        self.coords_handle = self.coordToHandle(base)
        # self.printFiber()
    
    # get next handle during iteration through slice
    def nextInSlice(self):
        # print("\t{} in next: handle {}, slice max {}, num to ret {}, ret so far {}".format(self.name, self.coords_handle, self.getSliceMaxLength(), self.num_to_ret, self.num_ret_so_far))
        if self.coords_handle == None or self.coords_handle >= self.getSliceMaxLength():
            return None
        if self.num_to_ret != None and self.num_to_ret < self.num_ret_so_far:
            return None
        # for formats that don't need to touch memory to get next
        to_ret = self.coords_handle
        self.num_ret_so_far += 1
        self.coords_handle += 1
        # print("\t\thandle to ret: {}".format(to_ret))
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

    def getUpdatedFiberHandle(self):
        return 0 # TODO: make this an actual indexable fiber handle to you
        # return self

    def getPayloads(self):
        return self.payloads

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
