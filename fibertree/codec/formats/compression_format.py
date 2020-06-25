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
    # def encodeCoord(current_fiber, current_scratchpad, prev_ind, ind)
    def encodeCoord(prev_ind, ind):
        return None
        # append new_entries to end of scratchpad
        # return new_entries

    @staticmethod
    def encodePayload(prev_ind, ind, payload):
        return None

    @staticmethod
    def endPayloads(num_to_pad):
        return [] # None

    # todo: maybe eventually combine the encode and decode like serialization
