from .compression_format import CompressionFormat

class Uncompressed(CompressionFormat):
    # constructor
    def __init__(self):
        self.name = "U"
        CompressionFormat.__init__(self)
        self.occupancies = list()
        self.count_payload_reads = False

    # instantiate this fiber in the format
    def encodeFiber(self, a, dim_len, codec, depth, ranks, output, output_tensor):
        # import codec
        from ..tensor_codec import Codec
        coords_key, payloads_key = codec.get_keys(ranks, depth)
        
        # init vars
        fiber_occupancy = 0
        
        occ_list = list()

        # keep track of shape during encoding
        self.shape = dim_len
        
        if depth < len(ranks) - 1:
            cumulative_occupancy = codec.get_start_occ(depth + 1)

            if codec.fmts[depth + 1].encodeUpperPayload():
                self.count_payload_reads = True
        else: # leaf level is always read payloads
            self.count_payload_reads = True
        for i in range(0, dim_len):
            # internal levels
            if depth < len(ranks) - 1:
                fiber, child_occupancy = codec.encode(depth + 1, a.getPayload(i), ranks, output, output_tensor)
                self.payloads.append(fiber)
                # print(fiber)
                # print(child_occupancy)
                # keep track of occupancy (cumulative requires ordering)
                if isinstance(cumulative_occupancy, int):
                    # print("\tcumulative {}, child {}".format(cumulative_occupancy, child_occupancy))
                    cumulative_occupancy = cumulative_occupancy + child_occupancy
                else:
                    cumulative_occupancy = [a + b for a, b in zip(cumulative_occupancy, child_occupancy)]
                codec.add_payload(depth, occ_list, cumulative_occupancy, child_occupancy)

                # store occupancy
                if codec.fmts[depth+1].encodeUpperPayload():
                    output[payloads_key].append(cumulative_occupancy)
                    self.occupancies.append(cumulative_occupancy)
            else: # leaf level
                if a.getPayload(i) == 0:
                    output[payloads_key].append(0)
                    self.payloads.append(0)
                else:
                    output[payloads_key].append(a.getPayload(i).value)
                    self.payloads.append(a.getPayload(i).value)
        
        # TODO: is 1 correct? maybe this should be occupancy?
        return 1

    ## SWOOP API functions 
    def handleToCoord(self, handle):
        return handle

    # TODO: stop passing around the ptr? maybe payload could just be the offset   
    def payloadToFiberHandle(self, payload):
        # print("\t{}::payloadToFiberHandle (U): payload {}, fiber_handle {}".format(self.name, payload, self.idx_in_rank * self.shape + payload))
        # print("\tshould print fiber")
        # self.printFiber()
        assert payload < self.shape
        return self.idx_in_rank * self.shape + payload
    
    # max number of elements in a slice is proportional to the shape
    def getSliceMaxLength(self):
        return self.shape

    def coordToHandle(self, coord):
        # print("{} coordToHandle {}, shape {}".format(self.name, coord, self.shape))
        if coord < 0 or coord >= self.shape:
            return None
        return coord
    
    def insertElement(self, coord):
        assert coord < self.shape
        return coord

    def updatePayload(self, handle, payload):
        # print("\t{} updatePayload: handle {}, payload {}".format(self.name, handle, payload))
        assert handle is not None and handle < self.shape
        
        # testing adding to the cache
        key = self.name + "_payloads_" + str(handle)
        self.cache.get(key) # try to access it
        self.cache[key] = payload # put it in the cache

        # if the payloads from lower level are explicit 
        if self.count_payload_reads:
            self.stats[self.payloads_write_key] += 1

        # print("\tupdate {}, handle {}, payload {}".format(self.name, handle, payload))
        if isinstance(payload, tuple):
            self.occupancies[handle] = payload[0]
            self.payloads[handle] = payload[1]
        else:
            self.payloads[handle] = payload
        # print("\tupdatePayload {}: handle {}, ret {}".format(self.name, handle, handle))
        return handle

    def getPayloads(self):
        return self.payloads

    def printFiber(self):
        print("{} :: occupancies {}, payloads {}".format(self.name, self.occupancies, self.payloads))
    
    def getSize(self):
        assert(len(self.payloads) > 0)
        assert(len(self.coords) == 0)
        size = len(self.occupancies)
        if not isinstance(self.payloads[0], CompressionFormat):
            size += len(self.payloads)
        
        # print("size of {} = {}. coords {}, occupancies {}, payloads {}".format(self.name, size, self.coords, self.occupancies, self.payloads))
        return size

    ### static methods
    @staticmethod
    def encodeCoord(prev_ind, ind):
        return []

    # default implementation is like in C
    # overwrite if this is changed
    @staticmethod
    def encodePayload(prev_ind, ind, payload):
        output = list()
        for i in range(prev_ind, ind):
            output.append(0)
        output.append(payload)
        return output

    @staticmethod
    def endPayloads(num_to_pad):
        return [0] * num_to_pad

    # implicit coords
    @staticmethod
    def encodeCoords():
        return False

    # implicit prev payloads
    @staticmethod
    def encodeUpperPayload():
        return False
