"""
CompressionFormat class
mostly just here to be inherited
"""
class CompressionFormat:
    def __init__(self):
        self.name = ""

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

    # 

    # todo: maybe eventually combine the encode and decode like serialization
