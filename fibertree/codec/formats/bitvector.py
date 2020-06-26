from .compression_format import CompressionFormat

class Bitvector(CompressionFormat):
    # untruncated bitvector
    def __init__(self):
        self.name = "B"

    @staticmethod
    def getName(self):
        return self.getName()

    @staticmethod
    def encodeCoord(prev_ind, ind):
        # return []
        output = list()	
        for i in range(prev_ind, ind):
            output.append(0)
        output.append(1)
        return output

    # bitvector 
    @staticmethod
    def encodePayload(prev_ind, ind, payload):
        return [payload]

    def endCoords(num_to_pad):
        return [0]*num_to_pad

    @staticmethod
    def endPayloads(num_to_pad):
        return []
        # return [0] * num_to_pad

    # explicit coords
    @staticmethod
    def encodeCoords():
        return True

    # implicit prev payloads
    @staticmethod
    def encodeUpperPayload():
        return False
