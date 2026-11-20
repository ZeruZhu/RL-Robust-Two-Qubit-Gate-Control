import numpy as np

from walsh_rl.walsh_basis import sequency, sequency_order, walsh_basis, walsh_matrix


def test_walsh_matrix_is_orthogonal():
    matrix = walsh_matrix(8)
    gram = matrix @ matrix.T
    assert np.allclose(gram, 8 * np.eye(8))


def test_sequency_order_is_monotone():
    matrix = walsh_matrix(8)
    sequencies = [sequency(row) for row in matrix]
    assert sequencies == sorted(sequencies)
    assert sequency_order(8)[0] == 0


def test_walsh_basis_shape():
    basis = walsh_basis(num_modes=4, n_bins=8)
    assert basis.shape == (4, 8)
