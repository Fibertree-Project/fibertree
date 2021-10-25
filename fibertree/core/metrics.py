class Metrics:
    # Create a class instance variable for the metrics collection
    collecting = False
    metrics = []


    @classmethod
    def beginCollect(cls):
        cls.metrics.append({})
        cls.collecting = True

    @classmethod
    def dump(cls):
        return cls.metrics[-1]

    @classmethod
    def endCollect(cls):
        cls.collecting = False

    @classmethod
    def inc(cls, line, metric, inc):
        assert(cls.collecting)

        line = line.strip()

        if line not in cls.metrics[-1]:
            cls.metrics[-1][line] = {}

        if metric not in cls.metrics[-1][line]:
            cls.metrics[-1][line][metric] = 0

        cls.metrics[-1][line][metric] += inc


    @classmethod
    def isCollecting(cls):
        return cls.collecting

