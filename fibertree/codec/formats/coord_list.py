from .compression_format import CompressionFormat

class CoordinateList(CompressionFormat):
    def __init__(self):
        self.name = "C"

    # TODO: put formats in a subdir

    # should you pass in the cumulative position or the relative position (from
    # the last nz?)
    @staticmethod
    def encodeCoord(prev_ind, ind):
        return [ind]

    @staticmethod
    def encodePayload(prev_ind, ind, payload):
        return [payload]

    # explicit coords
    @staticmethod
    def encodeCoords():
        return True

    # explicit prev payloads
    @staticmethod
    def encodeUpperPayload():
        return True
