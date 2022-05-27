import unittest

from fibertree import Tensor, Fiber, Payload


class TestCalculations(unittest.TestCase):

    def test_traverse(self):
        """Traverse a tensor"""

        a = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")

        a_m = a.getRoot()

        sum = 0

        for m, (a_k) in a_m:
            for k, (a_val) in a_k:
                sum += a_val

        self.assertEqual(sum, 2713)


    def test_copy(self):
        """Copy a tensor"""

        a = Tensor.fromYAMLfile("./data/test_tensor-1.yaml")
        z = Tensor(rank_ids=["M", "K"])

        a_m = a.getRoot()
        z_m = z.getRoot()

        for m, (z_k, a_k) in z_m << a_m:
            for k, (z_ref, a_val) in z_k << a_k:
                z_ref += a_val

        self.assertEqual(a, z)

    def test_sum(self):
        "Test sum"

        a = Tensor.fromYAMLfile("./data/tensor_sum_a.yaml")
        b = Tensor.fromYAMLfile("./data/tensor_sum_b.yaml")
        z = Tensor(rank_ids=["M"])

        a_m = a.getRoot()
        b_m = b.getRoot()
        z_m = z.getRoot()

        for m_coord, (z_ref, (op, a_k, b_k)) in z_m << (a_m | b_m):
            for k_coord, (op, a_val, b_val) in a_k | b_k:
                z_ref += a_val + b_val


        z_correct = Tensor.fromYAMLfile("./data/tensor_sum_z.yaml")

        self.assertEqual(z, z_correct)

    def test_dot(self):
        "Test dot product"

        a = Tensor.fromYAMLfile("./data/tensor_sum_a.yaml")
        b = Tensor.fromYAMLfile("./data/tensor_sum_b.yaml")
        z = Tensor(rank_ids=["M"])

        a_m = a.getRoot()
        b_m = b.getRoot()
        z_m = z.getRoot()

        for m_coord, (z_ref, (a_k, b_k)) in z_m << (a_m & b_m):
            for k_coord, (a_val, b_val) in a_k & b_k:
                z_ref += a_val * b_val

        z_correct = Tensor.fromYAMLfile("./data/tensor_dot_z.yaml")

        self.assertEqual(z, z_correct)

    def test_0D(self):
        "Test sum to rank 0 tensor"

        a = Tensor.fromYAMLfile("./data/conv-activations-a.yaml")
        z = Tensor(rank_ids=[])

        a_m = a.getRoot()
        z_ref = z.getRoot()

        for m_coord, (a_val) in a_m:
            z_ref += a_val

        self.assertEqual(z_ref, 12)

    def test_conv1d_ws(self):
        """Convolution 1d ws"""

        w = Tensor.fromYAMLfile("./data/conv-weights-a.yaml")
        i = Tensor.fromYAMLfile("./data/conv-activations-a.yaml")
        o = Tensor(rank_ids=["Q"])

        w_r = w.getRoot()
        i_h = i.getRoot()
        o_q = o.getRoot()

        W = w_r.maxCoord() + 1
        I = i_h.maxCoord() + 1
        Q = I - W + 1

        for r, (w_val) in w_r:
            for q, (o_q_ref, i_val) in o_q << i_h.project(lambda h: h-r, (0, Q)):
                o_q_ref += w_val * i_val

        o_ref = Tensor.fromYAMLfile("./data/conv-output-a.yaml")

        self.assertEqual(o, o_ref)

    def test_conv1d_is(self):
        """Convolution 1d is"""

        w = Tensor.fromYAMLfile("./data/conv-weights-a.yaml")
        i = Tensor.fromYAMLfile("./data/conv-activations-a.yaml")
        o = Tensor(rank_ids=["Q"])

        w_r = w.getRoot()
        i_h = i.getRoot()
        o_q = o.getRoot()

        W = w_r.maxCoord() + 1
        I = i_h.maxCoord() + 1
        Q = I - W + 1

        for h, (i_val) in i_h:
            for q, (o_q_ref, w_val) in o_q << w_r.project(lambda r: h-r, (0, Q)):
                o_q_ref += w_val * i_val

        o_ref = Tensor.fromYAMLfile("./data/conv-output-a.yaml")

        self.assertEqual(o, o_ref)

    def test_conv1d_os(self):
        """Convolution 1d os"""

        w = Tensor.fromYAMLfile("./data/conv-weights-a.yaml")
        i = Tensor.fromYAMLfile("./data/conv-activations-a.yaml")
        o = Tensor(rank_ids=["Q"])

        w_r = w.getRoot()
        i_h = i.getRoot()
        o_q = o.getRoot()

        W = w_r.maxCoord() + 1
        I = i_h.maxCoord() + 1
        Q = I - W + 1

        output_shape = Fiber(coords=range(Q), initial=1)

        for q, (o_q_ref, _) in o_q << output_shape:
            for h, (w_val, i_val) in w_r.project(lambda r: q+r) & i_h:
                o_q_ref += w_val * i_val

        o_ref = Tensor.fromYAMLfile("./data/conv-output-os-a.yaml")

        self.assertEqual(o, o_ref)


if __name__ == '__main__':
    unittest.main()

