import operator

""" Spec """

def GetIndexLevel(index_name, remaining_indices):
    # Omit index if it only occurs once in the mapping.
    # if all_indices.count(index_name) == 0:
    #    return None
    cnt = remaining_indices.count(index_name)
    if cnt:
        return cnt - 1
    else:
        return None
    return cnt


class ProblemSpec:
    """ ProblemSpec Class """

    def __init__(self, lhs, rhs):
        """__init__"""

        self.lhs = lhs
        self.rhs = rhs
        # This is a bit hacky.
        SpecTensor.tensor_id = 0
    
    def GetTensors(self):
        return self.lhs.GetTensors() | self.rhs.GetTensors()
    
    def PrintBody(self, indent):
        result = indent * " "
        result += self.lhs.PrintBody(True)
        result += " += "
        result += self.rhs.PrintBody(False)
        return result

    def PrintCoIterators(self, index_name, remaining_indices):
        result_lhs = self.lhs.PrintCoIterators(index_name, remaining_indices, True)
        result_rhs = self.rhs.PrintCoIterators(index_name, remaining_indices, False)
        if result_lhs and result_rhs:
            result = "("
            result += result_lhs
            result += ", "
            result += result_rhs
            result += ")"
            return result
        elif result_lhs:
            return result_lhs
        elif result_rhs:
            return result_rhs
        else:
            return ""

    def PrintCoIteration(self, index_name, remaining_indices):
        result_lhs = self.lhs.PrintCoIteration(index_name, remaining_indices, True)
        result_rhs = self.rhs.PrintCoIteration(index_name, remaining_indices, False)
        if result_lhs and result_rhs:
            result = result_lhs
            result += " << "
            result += result_rhs
            return result
        elif result_lhs:
            return result_lhs
        elif result_rhs:
            return result_rhs
        else:
            return ""

class TensorOp:
    """ TensorOp Class """

    def __init__(self, lhs, rhs, op):
        """__init__"""

        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def GetTensors(self):
        return self.lhs.GetTensors() | self.rhs.GetTensors()

    def PrintBody(self, is_ref):
        result = self.lhs.PrintBody(is_ref)
        result += " " + str(self.op) + " " # TODO
        result += self.rhs.PrintBody(is_ref)
        return result

    def PrintCoIterators(self, index_name, remaining_indices, is_ref):
        result_lhs = self.lhs.PrintCoIterators(index_name, remaining_indices, is_ref)
        result_rhs = self.rhs.PrintCoIterators(index_name, remaining_indices, is_ref)
        if result_lhs and result_rhs:
            result = "(" + result_lhs
            result += ", "
            result += result_rhs + ")"
            return result
        elif result_lhs:
            return result_lhs
        elif result_rhs:
            return result_rhs
        else:
            return None

    def PrintCoIteration(self, index_name, remaining_indices, is_ref):
        result_lhs = self.lhs.PrintCoIteration(index_name, remaining_indices, is_ref)
        result_rhs = self.rhs.PrintCoIteration(index_name, remaining_indices, is_ref)
        if result_lhs and result_rhs:
            result = "(" + result_lhs
            result += " & " # TODO: Smarter here.
            result += result_rhs + ")"
            return result
        elif result_lhs:
            return result_lhs
        elif result_rhs:
            return result_rhs
        else:
            return None

class TensorAccess:
    """ TensorAccess Class """

    def __init__(self, tensor, rank_ids):
        """__init__"""

        self.target = tensor
        # I hate python at the moment
        if not isinstance(rank_ids, tuple):
            self.rank_ids = [rank_ids]
        else:
            self.rank_ids = rank_ids
   
    def __mul__(self, other):
        return TensorOp(self, other, operator.__mul__)
 
    def __lshift__(self, other):
        return ProblemSpec(self, other)
    
    def GetTensors(self):
        return set([self.target])

    def PrintCoIterators(self, index_name, remaining_indices, is_ref):
        if self.target.HasIndex(index_name):
            next_index = self.target.GetNextIndex(remaining_indices)
            if next_index is None:
                return self.target.GetValueName(is_ref)
            else:
                index_level = GetIndexLevel(next_index, remaining_indices)
                return self.target.GetFiberName(next_index, index_level)
        else:
            return None

    def PrintCoIteration(self, index_name, remaining_indices, is_ref):
        if self.target.HasIndex(index_name):
            index_level = GetIndexLevel(index_name, remaining_indices)
            return self.target.GetFiberName(index_name, index_level)
        else:
            return None

    def PrintBody(self, is_ref):
        return self.target.GetValueName(is_ref)

class SpecTensor:
    """ SpecTensor Class """

    tensor_id = 0

    def __init__(self, *rank_ids):
        """__init__"""

        self.rank_ids = rank_ids
        self.id = SpecTensor.tensor_id
        SpecTensor.tensor_id = SpecTensor.tensor_id + 1
   
    # This is a rank-0 acccess (e.g., no indices)
    def __lshift__(self, other):
        assert(self.rank_ids is ())
        return ProblemSpec(TensorAccess(self, ()), other)

    # This is a rank-0 acccess (e.g., no indices)
    def __mul__(self, other):
        assert(self.rank_ids is ())
        return TensorOp(TensorAccess(self, ()), other, operator.__mul__)
 
    def __getitem__(self, rank_ids):
        return TensorAccess(self, rank_ids)
    
    def GetID(self):
        return self.id
    
    def GetIndexVar(self, position):
        if not self.rank_ids:
            return None
        else:
            return self.rank_ids[position]

    def GetName(self):
        return "T" + str(self.id)
    
    def HasIndex(self, index_name):
        for rank_id in self.rank_ids:
            if index_name in rank_id.GetIndexNames():
                return True
        return False
    
    def GetNextIndex(self, remaining_indices):
        for rank_id in self.rank_ids:
            for index_name in remaining_indices:
                if self.HasIndex(index_name):
                    return index_name
        return None

    def GetValueName(self, is_ref):
        if is_ref:
            return self.GetName() + "_ref"
        else:
            return self.GetName() + "_val"

    def GetFiberName(self, index_name, index_level):
        assert(self.HasIndex(index_name))
        result = self.GetName() + "_" + index_name
        result += str(index_level)
        return result

class IndexOp:
    """ IndexOp Class """

    def __init__(self, lhs, rhs, op):
        """__init__"""
        self.lhs = lhs
        self.rhs = rhs
        self.op = op

    def __add__(self, other):
        return IndexOp(self, other, operator.__add__)

    def __sub__(self, other):
        return IndexOp(self, other, operator.__sub__)

    def __mul__(self, other):
        return IndexOp(self, other, operator.__mul__)

    def __div__(self, other):
        return IndexOp(self, other, operator.__div__)

    def GetIndexNames(self):
        if isinstance(self.lhs, IndexOp):
            lhs_names = self.lhs.GetIndexNames()
        elif isinstance(self.lhs, SpecIndex):
            lhs_names = self.lhs.GetIndexNames()
        else:
            lhs_names = set()
        if isinstance(self.rhs, IndexOp):
            rhs_names = self.rhs.GetIndexNames()
        elif isinstance(self.rhs, SpecIndex):
            rhs_names = self.rhs.GetIndexNames()
        else:
            rhs_names = set()
        return lhs_names | rhs_names

class SpecIndex:
    """ SpecIndex Class """

    spec_id = 0

    def __init__(self, name=None):
        """__init__"""

        self.id = SpecIndex.spec_id
        SpecIndex.spec_id = SpecIndex.spec_id + 1
        if name is None:
            self.name = "I" + str(self.id)
        else:
            self.name = name
  
    def __add__(self, other):
        return IndexOp(self, other, operator.__add__)

    def __sub__(self, other):
        return IndexOp(self, other, operator.__sub__)

    def __mul__(self, other):
        return IndexOp(self, other, operator.__mul__)

    def __div__(self, other):
        return IndexOp(self, other, operator.__div__)

    def GetIndexNames(self):
        return set([self.name])

class Schedule:
    """ Schedule Class"""
    
    def __init__(self, problem_spec, mapping):
    
        self.problem_spec = problem_spec
        self.mapping = mapping
        # A bit hacky
        SpecIndex.spec_id = 0
    
    def PrintLoop(self, position, indent):
        index_name = self.mapping[position]
        remaining_indices = self.mapping[position:]
        result_iterators = self.problem_spec.PrintCoIterators(index_name, remaining_indices[1:])
        result_iteration = self.problem_spec.PrintCoIteration(index_name, remaining_indices)
        result = indent * " "
        index_level = GetIndexLevel(index_name, remaining_indices)
        result += "for " + index_name + str(index_level)
        if result_iterators:
            result += ", "
            result += result_iterators
        assert(result_iteration)
        result += " in "
        result += result_iteration
        result += ":\n"
        return result
    
    def __str__(self, indent = 0):

        all_tensors = self.problem_spec.GetTensors()
        result = indent * " "
        for tensor in all_tensors:
            result += indent * " "
            next_index = tensor.GetNextIndex(self.mapping)
            if next_index is None:
                # rank 0 tensor
                result += tensor.GetValueName(True) # TODO: really True?
            else:
                next_level = GetIndexLevel(next_index, self.mapping)
                result += tensor.GetFiberName(next_index, next_level)
            result += " = "
            result += tensor.GetName() + ".getRoot()\n"
        result += "\n"
        for position in range(0, len(self.mapping)):
             result += self.PrintLoop(position, indent)
             indent += 4
        result += self.problem_spec.PrintBody(indent)
        return result 
    

#Temporary conveniences
T=SpecTensor
I=SpecIndex

def spec_dot_product():
  m = I("m")
  Z = T()
  A = T(m)
  return Z << A[m]
  
m0_dot_product = Schedule(spec_dot_product(), ["m"])

def spec_cartesian_mul():
  m = I("m")
  n = I("n")
  Z = T(m, n)
  A = T(m)
  B = T(n)
  return Z[m, n] << A[m] * B[n]
  
m0n0_cartesian_mul = Schedule(spec_cartesian_mul(), ["m", "n"])
n0m0_cartesian_mul = Schedule(spec_cartesian_mul(), ["n", "m"])

def spec_matrix_mul():
  m = I("m")
  k = I("k")
  n = I("n")
  Z = T(m, n)
  A = T(m, k)
  B = T(k, n)
  return Z[m, n] << A[m, k] * B[k, n]

mkn_matrix_mul = Schedule(spec_matrix_mul(), ["m", "k", "n"])
mnk_matrix_mul = Schedule(spec_matrix_mul(), ["m", "n", "k"])
kmn_matrix_mul = Schedule(spec_matrix_mul(), ["k", "m", "n"])

n_mkn_matrix_mul = Schedule(spec_matrix_mul(), ["n", "m", "k", "n"])
k_mnk_matrix_mul = Schedule(spec_matrix_mul(), ["k", "m", "n", "k"])
n_kmn_matrix_mul = Schedule(spec_matrix_mul(), ["n", "k", "m", "n"])

kn_mkn_matrix_mul = Schedule(spec_matrix_mul(), ["k", "n", "m", "k", "n"])
nk_mnk_matrix_mul = Schedule(spec_matrix_mul(), ["n", "k", "m", "n", "k"])
mn_kmn_matrix_mul = Schedule(spec_matrix_mul(), ["m", "n", "k", "m", "n"])

mkn_mkn_matrix_mul = Schedule(spec_matrix_mul(), ["m", "k", "n", "m", "k", "n"])
mnk_mnk_matrix_mul = Schedule(spec_matrix_mul(), ["m", "n", "k", "m", "n", "k"])
kmn_kmn_matrix_mul = Schedule(spec_matrix_mul(), ["k", "m", "n", "k", "m", "n"])

n_mkn_mkn_matrix_mul = Schedule(spec_matrix_mul(), ["n", "m", "k", "n", "m", "k", "n"])
k_mnk_mnk_matrix_mul = Schedule(spec_matrix_mul(), ["k", "m", "n", "k", "m", "n", "k"])
n_kmn_kmn_matrix_mul = Schedule(spec_matrix_mul(), ["n", "k", "m", "n", "k", "m", "n"])


tactile_matrix_mul = Schedule(spec_matrix_mul(), ["m", "k", "n", "m", "k", "n", "m", "k"])
tactil2_matrix_mul = Schedule(spec_matrix_mul(), ["n", "k", "m", "n", "k", "m", "n", "k"])


def spec_2d_conv():
  p = I("p")
  q = I("q")
  r = I("r")
  s = I("s")
  Out = T(p, q)
  In  = T(p + r - 1, q + s - 1)
  Wt  = T(r, s)
  return Out[p, q] << In[p + r, q + s] * Wt[r, s]

pqrs_2d_conv = Schedule(spec_2d_conv(), ["p", "q" ,"r", "s"])
prqs_2d_conv = Schedule(spec_2d_conv(), ["p", "r" ,"q", "s"])
rspq_2d_conv = Schedule(spec_2d_conv(), ["r", "s" ,"p", "q"])
rpsq_2d_conv = Schedule(spec_2d_conv(), ["r", "p" ,"s", "q"])

def convert_xy_to_yxyx():
  x  = I("x")
  y  = I("y")
  x0 = I("x0")
  x1 = I("x1")
  y0 = I("y0")
  y1 = I("y1")
  Z = T(y1, x1, y0, x0)
  A = T(x, y)
  return Z[y1, x1, y0, x0] << A[x, y]

yxyx_convert_xy_to_yxyx = Schedule(convert_xy_to_yxyx(), ["y", "x", "y", "x"])

#t_yxyx = t.transform(["y", "x", "y", "x")], {"y" = [YO], "x" = [X0]}

#
#for y, t_x in t_y:
#    y1 = 
#    y0 = 
#    out_x1 = out.insertOrLookup(y1, Fiber())
#    for x, t_val in t_x:
#      x1 = 
#      x0 =
#      out_y0 = out_x1.insertOrLookup(x1, Fiber())
#      out_x0 = out_y0.insertOrLookup(y0, Fiber())
#      out_x0.append(x0, t_val)
#
#
#src_fiber = src.getRoot()
#src_fiber_stack.push(src_fiber)
#
#while !src_fiber.isLeaf():
#    src_rank_id = src_fiber.rankID()
#    for src_coord, src_payload in src_fiber:
#        for tgt_rank in tgt_ranks[src_rank_id]:
#            tgt_coord[src_rank_id][tgt_rank] = 
#                coord_func[src_rank_id][tgt_rank](src_coord)
#        src_fiber = src_payload
#    
#                  
#    


