# Copyright © 2024 QunaSys Inc. All rights reserved.
#
# Licensed under the QunaSys Source Available License Version 1.0 (QSAL-1.0);
# you may not use this file except in compliance with the License.
# Please refer to the LICENSE file distributed with this source code for the
# full terms.

from collections import defaultdict
from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
import scipy
from quri_parts.core.operator import Operator, is_hermitian
from quri_parts.core.operator.representation import (
    transition_amp_comp_basis,
    transition_amp_representation,
)
from quri_parts.core.sampling import ConcurrentSampler, MeasurementCounts
from quri_parts.core.state import (
    CircuitQuantumState,
    ComputationalBasisState,
    ComputationalBasisSuperposition,
)
from scipy.sparse import coo_matrix, csc_matrix


def pick_out_states(
    qubit_count: int,
    meas_counts: MeasurementCounts,
    num_states_pick_out: Optional[int] = None,
) -> Sequence[ComputationalBasisState]:
    """Pick the states out of :class:`MeasurementCounts`

    Args:
        qubit_count: Number of qubits
        meas_counts: :class:`MeasurementCounts`
        num_states_pick_out: Number of states to pick out from ``meas_counts``
    """
    keys_sorted = sorted(meas_counts.keys(), key=lambda x: meas_counts[x], reverse=True)
    if num_states_pick_out:
        keys_sorted = keys_sorted[:num_states_pick_out]
    states = [ComputationalBasisState(qubit_count, bits=key) for key in keys_sorted]
    return states


def penalty_term_comp_state(
    penalty_state: ComputationalBasisSuperposition,
    weight: float,
) -> scipy.sparse.spmatrix:
    """Calculate penalty term
    :math:`(\\text{weight}) \\times |\\text{state}\\rangle\\langle \\text{state}|`

    Args:
        penalty_state: :class:`ComputationalBasisSuperposition`, \
            which is the tuple of coefficients and basis states.
        weight: weight for the penalty term
    """
    dim = len(penalty_state[0])
    penalty_state_coefs, _ = penalty_state

    penalty_matrix: npt.NDArray[np.complex_] = np.zeros((dim, dim), dtype=np.complex_)

    for j in range(dim):
        for k in range(dim):
            penalty_matrix[j, k] += (
                weight * penalty_state_coefs[j] * penalty_state_coefs[k].conjugate()
            )
    penalty_matrix_sparse = csc_matrix(penalty_matrix)
    return penalty_matrix_sparse


def generate_truncated_hamiltonian(
    hamiltonian: Operator,
    states: Sequence[ComputationalBasisState],
) -> scipy.sparse.spmatrix:
    """Generate truncated Hamiltonian on the given basis states.

    Args:
        hamiltonian: :class:`Operator`
        states: basis states
    """
    dim = len(states)

    # variables for constructing coo matrix
    values = []
    row_ids = []
    column_ids = []

    h_transition_amp_repr = transition_amp_representation(hamiltonian)
    for m in range(dim):
        for n in range(m, dim):
            mn_val = transition_amp_comp_basis(
                h_transition_amp_repr, states[m].bits, states[n].bits
            )
            if mn_val:
                # (m,n) element
                values.append(mn_val)
                row_ids.append(m)
                column_ids.append(n)
                if m != n:
                    # (n,m) element
                    values.append(mn_val.conjugate())
                    row_ids.append(n)
                    column_ids.append(m)

    # Create sparse operator
    truncated_hamiltonian = coo_matrix(
        (values, (row_ids, column_ids)), shape=(dim, dim)
    ).tocsc(copy=False)
    truncated_hamiltonian.eliminate_zeros()

    return truncated_hamiltonian


def _diagonalize_truncated_hamiltonian(
    truncated_hamiltonian: scipy.sparse.spmatrix,
    n: int,
    k: int,
) -> tuple[list[float], list[list[float]]]:
    """Diagonalize a truncated Hamiltonian and returns k eigenvalues and
    eigenvectors.

    Args:
        truncated_hamiltonian: A truncated Hamiltonian to be diagonalized.
        n: Dimension of the truncated Hamiltonian.
        k: Number of eigenvalues and eigenvectors to return.

    Returns:
        A pair of k eignevalues and corresponding eigenvectors.
    """
    if n == 1:
        res_eig = ([truncated_hamiltonian[0, 0]], np.array([[1.0]]))
    else:
        if k >= n - 1:
            # In this case it is necessary to use scipy.linalg.eigh(A.toarray()).
            # https://github.com/scipy/scipy/blob/dafd14bc6537e4b412fb05f9b0246a29c1ed2778/scipy/sparse/linalg/_eigen/arpack/arpack.py#L1605-L1607
            # https://github.com/scipy/scipy/blob/dafd14bc6537e4b412fb05f9b0246a29c1ed2778/scipy/sparse/linalg/_eigen/arpack/arpack.py#L1277-L1279
            res_eig = scipy.linalg.eigh(truncated_hamiltonian.toarray())
        else:
            res_eig = scipy.sparse.linalg.eigsh(truncated_hamiltonian, k, which="SA")
    # When scipy.linalg.eigh() is used, the number of returned eigenvalues can be
    # larger than k, So we need to truncated them to k elements.
    eigvals = res_eig[0][0:k]
    eigvecs = res_eig[1].T.tolist()[0:k]
    return (eigvals, eigvecs)


def qsci(
    hamiltonian: Operator,
    approx_states: Sequence[CircuitQuantumState],
    sampler: ConcurrentSampler,
    total_shots: int,
    num_states_pick_out: Optional[int] = None,
) -> tuple[Sequence[float], Sequence[ComputationalBasisSuperposition]]:
    """Quantum-selected Configuration Interaction (QSCI), a method for finding
    eigenvalues and eigenstates of a given Hamiltonian by only using important
    electronic configurations based on sampling on quantum computers and
    diagonalizing the Hamiltonian in the identified subspace.

    Args:
        hamiltonian: :class:`Operator`
        approx_states: Approximated ground (and excited) states
        sampler: :class:`ConcurrentSampler` for the target backend
        total_shots: Total number of shots available for sampling measurements
        num_states_pick_out: Number of states to pick out from ``meas_counts``

    References:
        K. Kanno et al., "Quantum-Selected Configuration Interaction: classical
        diagonalization of Hamiltonians in subspaces selected by quantum computers",
        `arXiv:2302.11320 [quant-ph] <https://arxiv.org/abs/2302.11320>`_.
    """
    if not is_hermitian(hamiltonian):
        raise ValueError("Hamiltonian must be hermitian.")

    qubit_count = approx_states[0].qubit_count
    num_eigs_calc = len(approx_states)

    if num_states_pick_out and num_states_pick_out < num_eigs_calc:
        raise ValueError(
            "num_states_pick_out must be larger than or equal to the number of"
            "approx_states."
        )

    circuits = [state.circuit for state in approx_states]
    meas_counts_list = sampler(
        [(circuit, total_shots // num_eigs_calc) for circuit in circuits]
    )

    merged_meas_counts: dict[int, Union[int, float]] = defaultdict(int)
    for meas_counts in meas_counts_list:
        for bits, counts in meas_counts.items():
            merged_meas_counts[bits] += counts

    states = pick_out_states(qubit_count, merged_meas_counts, num_states_pick_out)
    truncated_hamiltonian = generate_truncated_hamiltonian(hamiltonian, states)

    eigvals, eigvecs = _diagonalize_truncated_hamiltonian(
        truncated_hamiltonian, len(states), num_eigs_calc
    )
    return (eigvals, [(eigvecs[i], states) for i in range(num_eigs_calc)])


def sequential_qsci(
    hamiltonian: Operator,
    approx_states: Sequence[CircuitQuantumState],
    sampler: ConcurrentSampler,
    weights_penalty: Sequence[float],
    total_shots: int,
    num_states_pick_out: Optional[int] = None,
) -> tuple[Sequence[float], Sequence[ComputationalBasisSuperposition]]:
    """Quantum-selected Configuration Interaction (QSCI) method executed
    sequentially to find ground and excited states of given Hamiltonian.

    Args:
        hamiltonian: :class:`Operator`
        approx_states: Approximated ground (and excited) states
        sampler: :class:`ConcurrentSampler` for the target backend
        weigths_penalty: Weights of the penalty for finding excited states. \
            ``weights_penalty[:i]`` are used for finding ``i``-th excited state. \
            ``len(weights_penalty)`` must be equal to ``len(approx_states) - 1``.
        total_shots: Total number of shots available for sampling measurements
        num_states_pick_out: Number of states to pick out from ``meas_counts``
    """
    if not is_hermitian(hamiltonian):
        raise ValueError("Hamiltonian must be hermitian.")

    qubit_count = approx_states[0].qubit_count

    circuits = [state.circuit for state in approx_states]
    meas_counts_list = sampler(
        [(circuit, total_shots // len(approx_states)) for circuit in circuits]
    )

    ret_eigvals: list[float] = []
    ret_eigstates: list[ComputationalBasisSuperposition] = []

    for meas_counts in meas_counts_list:
        states = pick_out_states(qubit_count, meas_counts, num_states_pick_out)

        truncated_hamiltonian = generate_truncated_hamiltonian(hamiltonian, states)

        # add penalty terms
        for eigstate_idx, eigstate in enumerate(ret_eigstates):
            coefs, _ = _project_state(eigstate, states)
            truncated_hamiltonian += penalty_term_comp_state(
                (coefs, states), weights_penalty[eigstate_idx]
            )

        eigvals, eigvecs = _diagonalize_truncated_hamiltonian(
            truncated_hamiltonian, len(states), 1
        )
        ret_eigvals.append(eigvals[0])
        ret_eigstates.append((eigvecs[:][0], states))

    return ret_eigvals, ret_eigstates


def _project_state(
    state: ComputationalBasisSuperposition, basis: Sequence[ComputationalBasisState]
) -> ComputationalBasisSuperposition:
    """Project given ``state`` to the space with ``basis``."""
    state_coefs, comp_basis_states = state
    coefs: list[complex] = []
    for basis_state in basis:
        if basis_state in comp_basis_states:
            index_coef = comp_basis_states.index(basis_state)
            coefs.append(state_coefs[index_coef])
        else:
            coefs.append(0.0)
    return (coefs, basis)
