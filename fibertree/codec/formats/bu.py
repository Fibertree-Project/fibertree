from .compression_format import CompressionFormat

class UncompressedBitvector(CompressionFormat):
    def __init__(self):
        self.name = "U"

    @staticmethod
    def getName(self):
        return self.getName()

    def encodeFiber(a, dim_len, codec, depth, ranks, output):
        # import codec
        from ..tensor_codec import Codec
        
        # init vars
        fiber_occupancy = 0
        cumulative_occupancy = 0
        occ_list = list()
		prev_nz = 0        


        return fiber_occupancy, occ_list

    @staticmethod
    def encodeCoord(prev_ind, ind):
        output = list()
        for i in range(prev_ind, ind):
            output.append(0)
        output.append(1)
        return output
        # return []

    # default implementation is like in C
    # overwrite if this is changed
    @staticmethod
    def encodePayload(prev_ind, ind, payload):
        return [payload]

	@staticmethod
	def endCoords(num_to_pad):
		return [0]*num_to_pad

    @staticmethod
    def endPayloads(num_to_pad):
        return []
		# return [0] * num_to_pad

    # implicit coords
    @staticmethod
    def encodeCoords():
`        return True

    # implicit prev payloads
    @staticmethod
    def encodeUpperPayload():
        return False
