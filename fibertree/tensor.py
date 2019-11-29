import yaml

from fibertree.rank  import Rank
from fibertree.fiber import Fiber

""" Tensor """

class Tensor:
    """ Tensor Class """

    def __init__(self, yamlfile="", rank_ids=None):
        """__init__"""

        self.yamlfile = yamlfile
        
        if (yamlfile != ""):
            # Note: rank_ids are ignored...
            self.parse(yamlfile)
            return

        #
        # Initialize an empty tensor with an empty root fiber
        #
        assert(not rank_ids is None)

        self.set_rank_ids(rank_ids)

        root_fiber = Fiber()
        self.ranks[0].append(root_fiber)

    @classmethod
    def fromUncompressed(cls, rank_ids=None, root_list=None):
        """Create tensor from uncompressed fiber tree"""

        fiber = Fiber.fromUncompressed(root_list)
        return Tensor.fromFiber(rank_ids, fiber)


    @classmethod
    def fromFiber(cls, rank_ids=None, fiber=None):
        """Create a tensor from a fiber"""

        tensor = cls(rank_ids=rank_ids)

        tensor.setRoot(fiber)

        return tensor


#
# Accessor methods
#

    # TBD: Fix style of this method name

    def set_rank_ids(self, rank_ids):
        """set_rank_ids"""

        self.rank_ids = rank_ids

        #
        # Create a linked list of ranks
        #
        self.ranks = []
        old_rank = None
        for id in rank_ids:
            new_rank = Rank(name=id)
            if not old_rank is None: old_rank.set_next(new_rank)
            old_rank = new_rank
            self.ranks.append(new_rank)


    def setRoot(self, root):
        """(Re-)populate self.ranks with "root"""

        # Clear out existing rank information
        for r in self.ranks:
            r.clearFibers()

        self._addFiber(root)


    def _addFiber(self, fiber, level=0):
        """Recursively fill in ranks from "fiber"."""

        self.ranks[level].append(fiber)

        for p in fiber.getPayloads():
            if isinstance(p, Fiber):
                self._addFiber(p, level+1)


    def root(self):
        """root"""

        return self.ranks[0].getFibers()[0]


    def values(self):
        """Count of values in the tensor"""

        return self.root().values()

#
# String methods
#
    def print(self, title=None):
        """print"""

        if not title is None:
            print("%s" % title)

        print("%s" % self)
        print("")

    def __repr__(self):
        """__repr__"""

        str = "T(%s)/[" % ",".join(self.rank_ids) + "\n"
        for r in self.ranks:
            str += "  " + r.__repr__() + "\n"
        str += "]"
        return str

#
# Yaml input/output methods
#

    def parse(self, file):
        """Parse a yaml file containing a tensor"""

        with open(file, 'r') as stream:
            try:
                y_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)

        #
        # Make sure key "tensor" exists
        #
        if not isinstance(y_file, dict) or 'tensor' not in y_file:
            print("Yaml is not a tensor")
            exit(1)

        y_tensor = y_file['tensor']

        #
        # Make sure key "rank_ids" exists
        #
        if not isinstance(y_tensor, dict) or 'rank_ids' not in y_tensor:
            print("Yaml has no rank_ids")
            exit(1)

        self.set_rank_ids(y_tensor['rank_ids'])

        #
        # Make sure key "root" exists
        #
        if 'root' not in y_tensor:
            print("Yaml has no root")
            exit(1)

        y_root = y_tensor['root']

        #
        # Generate the tree recursively
        #   Note: fibers are added into self.ranks inside method
        #
        Fiber.dict2fiber(y_root[0], ranks=self.ranks)


    def dump(self, filename):
        """Dump a tensor to a file in YAML format"""

        
        root_dict = self.root().fiber2dict()

        tensor_dict = { 'tensor':
                        { 'rank_ids': self.rank_ids,
                          'root': [ root_dict ]
                        } }
        with open(filename, 'w') as file:
            document = yaml.dump(tensor_dict, file)

