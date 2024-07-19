# QunaSys Source Available License Version 1.0 (QSAL-1.0)
# July 2024
# These license terms and conditions apply to programs (referred to as "Source Code") related to quantum algorithms that are copyrighted and distributed by QunaSys Inc. The Original Code includes the source code. The same applies hereinafter. Any person or entity that uses the Source Code must comply with the terms of this license.
# Definitions
# "Copyright Holder" refers to QunaSys Inc.
# "You" means the person (including Contributors) who uses the Original Code or Derivative Works.
# "Contributor" means the person who creates the Derivative Works.
# "Original Code" means the program (including source code) related to the Quantum Algorithm that is copyrighted and distributed by the Copyright Holder.
# "Derivative Works" means all works based on or created using the Original Code.
# A "work based on the Original Code" means a derivative work of the Original Code.
# A "work created using the Original Code" means a work created by combining or linking with the Original Code.
# Grant of Copyright License
# Subject to the terms of this License, whether you are an individual, an academic institution, a non-profit, or a for-profit organization, you may use the Original Code and Derivative Works only for personal use, academic research purposes, educational purposes, to verify the practicality of quantum computers/quantum algorithms, for development purposes, provided that such uses do not involve receiving compensation from third parties, or for collaborative research (whether paid or unpaid) by academic institutions. The Copyright Holder grants you a perpetual, worldwide, non-exclusive, royalty-free, irrevocable copyright license to use, reproduce, modify, adapt, and distribute the Original Code and Derivative Works.
# Please note, permission for commercial use of the Original Code and Derivative Works is not granted under this license. Contributors must comply with the following terms and conditions when copying or distributing the Original Code and Derivative Works.
# Grant of Patent Licenses
# Subject to the terms of this License, and in particular to the license terms of the foregoing Copyright License, the Copyright Holder or Contributor grants you a perpetual, worldwide, non-exclusive, royalty-free, irrevocable patent license to use the Original Code and Derivative Works to the extent necessary to make commercial use of them. It is important to note that this license does not grant permission for commercial use of the Original Code and Derivative Works as described below.
# Prohibition of Commercial Use
# The license to use the Original Code and Derivative Works under these terms is only for acts expressly permitted. No license is granted for any commercial use or practice of the Original Code and Derivative Works. The Copyright Holder prohibits the commercial use or implementation of the Original Code and Derivative Works. "Commercial use" includes the process of developing, manufacturing, or selling goods/services/materials, providing services to third parties (whether compensated or not), or any other use directly or indirectly for commercial or monetary gain, except as expressly permitted in this License.
# If you wish to use the Original Code or Derivative Works for commercial purposes, you must obtain a separate license from the Copyright Holder and/or Contributor.
# Distribution or Redistribution
# You may distribute or redistribute the Original Code or Derivative Works under the following terms and conditions, provided you comply with this License:
# 1. Ensure any third party using the Original Code or Derivative Works complies with the terms of this License.
# 2. Clearly identify and date any changes made to the Original Code or Derivative Works.
# 3. Attach a copy of these License Terms and Conditions to the Original Code or Derivative Works.
# 4. Include a copyright notice in the Original Code or Derivative Works.
# 5. License all Derivative Works in accordance with the terms set forth in this License.
# 6. Provide contact information for the contributor of the derivative work for third parties to obtain permission for commercial use.
# 7. Include a list of patent rights, if any, necessary for third party exploitation of the Original Code or Derivative Works.
# Disclaimer
# Unless otherwise agreed to by the Copyright Holder and Contributors, under no circumstances shall the Copyright Holder and Contributors be liable for any commercial, business, technical, or computer failures, malfunctions, commercial losses, damages, or other direct or indirect damages arising out of the use or performance of the Original Code, Derivative Works, or any related products.
# Termination
# If you file a lawsuit or other legal claim for copyright infringement or patent infringement against the Copyright Holder, Contributor, or other user of the Source Code or Derivative Works, the copyright license and patent license granted to you under this License will automatically terminate. The same applies to any breach of the terms of this License.
# Contact Information for Commercial Use and Support
# To use the Original Code for commercial purposes, prior permission from the Copyright Holder is required. Please email <info@qunasys.com> for permissions and support requests. Prior permission from the Contributor who created the derivative work may also be required for commercial use of Derivative Works.

from collections.abc import Sequence
from typing import Optional
from unittest import mock

import numpy as np
import pytest
from quri_parts.core.operator import Operator, pauli_label
from quri_parts.core.sampling import MeasurementCounts
from quri_parts.core.state import (
    ComputationalBasisState,
    ComputationalBasisSuperposition,
)
from scipy.sparse import csc_matrix, spmatrix

from quri_parts_qsci import (
    _project_state,
    generate_truncated_hamiltonian,
    penalty_term_comp_state,
    pick_out_states,
    qsci,
    sequential_qsci,
)


def _hamiltonian() -> Operator:
    hamiltonian_terms = [
        Operator(
            {
                pauli_label("Z0"): 0.1,
                pauli_label("Z1"): 0.2,
                pauli_label("Y2"): 0.3,
                pauli_label("Y4"): 0.4,
                pauli_label("X0 X2 X4"): 0.5,
                pauli_label("X1 X3 X5"): 0.6,
            }
        )
    ]
    hamiltonian = Operator()
    for term in hamiltonian_terms:
        hamiltonian += term

    return hamiltonian


def test_pick_out_states() -> None:
    qubit_count = 3
    meas_counts: MeasurementCounts = {0: 100, 1: 10, 2: 200, 3: 3, 4: 40, 7: 1000}
    expected_all = [
        ComputationalBasisState(qubit_count, bits=0),
        ComputationalBasisState(qubit_count, bits=1),
        ComputationalBasisState(qubit_count, bits=2),
        ComputationalBasisState(qubit_count, bits=3),
        ComputationalBasisState(qubit_count, bits=4),
        ComputationalBasisState(qubit_count, bits=7),
    ]
    assert set(pick_out_states(qubit_count, meas_counts)) == set(expected_all)

    expected_top_3 = [
        ComputationalBasisState(qubit_count, bits=0),
        ComputationalBasisState(qubit_count, bits=2),
        ComputationalBasisState(qubit_count, bits=7),
    ]
    assert set(pick_out_states(qubit_count, meas_counts, 3)) == set(expected_top_3)


def test_penalty_term_comp_state() -> None:
    qubit_count = 3
    basis = [
        ComputationalBasisState(qubit_count, bits=0),
        ComputationalBasisState(qubit_count, bits=2),
        ComputationalBasisState(qubit_count, bits=4),
        ComputationalBasisState(qubit_count, bits=5),
    ]
    coefs_list = [
        [np.sqrt(2) / np.sqrt(3), 1 / np.sqrt(3), 0, 0],
        [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / 2, 1 / 2, 1 / 2, 1 / 2],
    ]

    states = [(coefs, basis) for coefs in coefs_list]
    weights = [0.1, 0.2, 0.3]

    dim_penalty_mat = len(coefs_list[0])
    penalty_mat = csc_matrix((dim_penalty_mat, dim_penalty_mat))
    for state, weight in zip(states, weights):
        penalty_mat += penalty_term_comp_state(state, weight)

    expected = np.array(
        [
            [0.14166667, 0.12214045, 0.075, 0.075],
            [0.12214045, 0.10833333, 0.075, 0.075],
            [0.075, 0.075, 0.175, 0.175],
            [0.075, 0.075, 0.175, 0.175],
        ]
    )
    assert np.allclose(penalty_mat.toarray(), expected)


def test_generate_truncated_hamiltonian() -> None:
    qubit_count = 6
    states = [
        ComputationalBasisState(qubit_count, bits=0),
        ComputationalBasisState(qubit_count, bits=4),
        ComputationalBasisState(qubit_count, bits=20),
        ComputationalBasisState(qubit_count, bits=21),
    ]
    hamiltonian = _hamiltonian()

    trunc_hamiltonian = generate_truncated_hamiltonian(hamiltonian, states)
    expected = np.array(
        [
            [0.3, -0.3j, 0.0, 0.5],
            [0.3j, 0.3, -0.4j, 0.0],
            [0.0, 0.4j, 0.3, 0.0],
            [0.5, 0.0, 0.0, 0.1],
        ]
    )
    assert np.allclose(trunc_hamiltonian.toarray(), expected)


def check_subspace_eigenvectors(
    hamiltonian: spmatrix,
    eigvals: Sequence[float],
    lin_comb_states: Sequence[ComputationalBasisSuperposition],
    weights: Optional[Sequence[float]] = None,
) -> None:
    for i in range(len(lin_comb_states)):
        coefs = lin_comb_states[i][0]
        states = lin_comb_states[i][1]

        trunc_hamiltonian = generate_truncated_hamiltonian(hamiltonian, states)
        if i > 0 and weights:
            # Note: this penalty is specialized for data specified in
            # sequential_qsci test cases
            n_states = len(lin_comb_states[0][1])
            coefs_penalty = [
                0.0,
                *lin_comb_states[0][0][0 : n_states - 1],  # noqa: E203
            ]
            print(coefs_penalty)
            print(lin_comb_states[0][1])
            trunc_hamiltonian += penalty_term_comp_state(
                (coefs_penalty, lin_comb_states[0][1]), weights[0]
            )
        trunc_hamiltonian = trunc_hamiltonian.toarray()
        expectation_value = np.dot(np.conj(coefs), np.dot(trunc_hamiltonian, coefs))
        transformed_vector = np.dot(trunc_hamiltonian, coefs)
        transformed_vector /= np.linalg.norm(transformed_vector)
        assert eigvals[i] == pytest.approx(expectation_value)
        assert np.allclose(np.abs(coefs), np.abs(transformed_vector))


class TestQSCI:
    def setup(
        self,
    ) -> tuple[int, spmatrix, Sequence[ComputationalBasisState], mock.Mock, int]:
        qubit_count = 6
        hamiltonian = _hamiltonian()
        approx_states = [
            ComputationalBasisState(qubit_count, bits=0) for bits in (0, 1)
        ]
        sampler = mock.Mock()
        sampler.return_value = [
            {
                0: 40,
                1: 10,
                2: 20,
                3: 3,
                4: 20,
                8: 30,
                10: 10,
                21: 30,
            },
            {0: 50, 4: 50, 20: 100, 21: 40},
        ]
        total_shots = 10000
        return (
            qubit_count,
            hamiltonian,
            approx_states,
            sampler,
            total_shots,
        )

    def test_qsci(self) -> None:
        qubit_count, hamiltonian, approx_states, sampler, total_shots = self.setup()

        eigvals, lin_comb_states = qsci(
            hamiltonian, approx_states, sampler, total_shots, num_states_pick_out=4
        )

        # Checks if returned vectors are eigenvectors
        assert len(eigvals) == len(lin_comb_states) == len(approx_states)
        check_subspace_eigenvectors(hamiltonian, eigvals, lin_comb_states)

        assert lin_comb_states[0][1] == [
            ComputationalBasisState(qubit_count, bits=bits) for bits in (20, 0, 4, 21)
        ]

        # Checks if returned vectors are orthogonal eath other
        assert _inner_product_comp_basis_states(
            lin_comb_states[0], lin_comb_states[1]
        ) == pytest.approx(0.0, abs=1e-5)

    # Test diagonalization for k >= N-1 case
    # (k = len(approx_states) = 2, N = num_states_pick_out = 3)
    def test_qsci_few_pick_out_states1(self) -> None:
        qubit_count, hamiltonian, approx_states, sampler, total_shots = self.setup()

        eigvals, lin_comb_states = qsci(
            hamiltonian, approx_states, sampler, total_shots, num_states_pick_out=3
        )

        # Checks if returned vectors are eigenvectors
        assert len(eigvals) == len(lin_comb_states) == len(approx_states)
        check_subspace_eigenvectors(hamiltonian, eigvals, lin_comb_states)

        assert lin_comb_states[0][1] == [
            ComputationalBasisState(qubit_count, bits=bits) for bits in (20, 0, 4)
        ]

        # Checks if returned vectors are orthogonal eath other
        assert _inner_product_comp_basis_states(
            lin_comb_states[0], lin_comb_states[1]
        ) == pytest.approx(0.0, abs=1e-5)

    # Test diagonalization for k >= N case
    # (k = len(approx_states) = 2, N = num_states_pick_out = 2)
    def test_qsci_few_pick_out_states2(self) -> None:
        qubit_count, hamiltonian, approx_states, sampler, total_shots = self.setup()

        eigvals, lin_comb_states = qsci(
            hamiltonian, approx_states, sampler, total_shots, num_states_pick_out=2
        )

        # Checks if returned vectors are eigenvectors
        assert len(eigvals) == len(lin_comb_states) == len(approx_states)
        check_subspace_eigenvectors(hamiltonian, eigvals, lin_comb_states)

        assert lin_comb_states[0][1] == [
            ComputationalBasisState(qubit_count, bits=bits) for bits in (20, 0)
        ]

        # Checks if returned vectors are orthogonal eath other
        assert _inner_product_comp_basis_states(
            lin_comb_states[0], lin_comb_states[1]
        ) == pytest.approx(0.0, abs=1e-5)

    # Test diagonalization for k >= N case
    # (k = len(approx_states) = 2, N = num_states_pick_out = 1)
    def test_qsci_one_pick_out_state(self) -> None:
        qubit_count, hamiltonian, approx_states, sampler, total_shots = self.setup()
        approx_states = [
            ComputationalBasisState(qubit_count, bits=0),
        ]
        sampler.return_value = [
            {
                0: 40,
                1: 10,
                2: 20,
                3: 3,
                4: 20,
                8: 30,
                10: 10,
                21: 30,
            },
        ]

        eigvals, lin_comb_states = qsci(
            hamiltonian, approx_states, sampler, total_shots, num_states_pick_out=1
        )

        # Checks if returned vectors are eigenvectors
        assert len(eigvals) == len(lin_comb_states) == len(approx_states)
        check_subspace_eigenvectors(hamiltonian, eigvals, lin_comb_states)

        assert lin_comb_states[0][1] == [
            ComputationalBasisState(qubit_count, bits=0),
        ]


class TestSequentialQSCI:
    def setup(
        self,
    ) -> tuple[
        int,
        spmatrix,
        Sequence[ComputationalBasisState],
        mock.Mock,
        int,
        Sequence[float],
    ]:
        qubit_count = 6
        hamiltonian = _hamiltonian()
        approx_states = [
            ComputationalBasisState(qubit_count, bits=bits) for bits in (0, 1)
        ]
        sampler = mock.Mock()
        sampler.return_value = [
            {
                0: 40,
                1: 10,
                2: 20,
                3: 3,
                4: 20,
                8: 30,
                10: 10,
                21: 30,
            },
            {0: 50, 8: 50, 20: 100, 21: 40},
        ]
        total_shots = 10000
        weights = [2.0]
        return (
            qubit_count,
            hamiltonian,
            approx_states,
            sampler,
            total_shots,
            weights,
        )

    def test_sequential_qsci(self) -> None:
        (
            qubit_count,
            hamiltonian,
            approx_states,
            sampler,
            total_shots,
            weights,
        ) = self.setup()

        eigvals, lin_comb_states = sequential_qsci(
            hamiltonian,
            approx_states,
            sampler,
            weights,
            total_shots,
            num_states_pick_out=3,
        )

        assert lin_comb_states[0][1] == [
            ComputationalBasisState(qubit_count, bits=bits) for bits in (0, 8, 21)
        ]
        assert lin_comb_states[1][1] == [
            ComputationalBasisState(qubit_count, bits=bits) for bits in (20, 0, 8)
        ]

        # Checks if returned vectors are eigenvectors
        assert len(eigvals) == len(lin_comb_states) == len(approx_states)
        check_subspace_eigenvectors(hamiltonian, eigvals, lin_comb_states, weights)

        # Checks if returned vectors are orthogonal eath other
        assert _inner_product_comp_basis_states(
            lin_comb_states[0], lin_comb_states[1]
        ) == pytest.approx(0.0, abs=1e-5)

    # Test diagonalization for k >= N-1 case (k=1, N=num_states_pick_out=2)
    def test_sequential_qsci_few_pick_out_states(self) -> None:
        (
            qubit_count,
            hamiltonian,
            approx_states,
            sampler,
            total_shots,
            weights,
        ) = self.setup()

        eigvals, lin_comb_states = sequential_qsci(
            hamiltonian,
            approx_states,
            sampler,
            weights,
            total_shots,
            num_states_pick_out=2,
        )

        assert lin_comb_states[0][1] == [
            ComputationalBasisState(qubit_count, bits=bits) for bits in (0, 8)
        ]
        assert lin_comb_states[1][1] == [
            ComputationalBasisState(qubit_count, bits=bits) for bits in (20, 0)
        ]

        # Checks if returned vectors are eigenvectors
        assert len(eigvals) == len(lin_comb_states) == len(approx_states)
        check_subspace_eigenvectors(hamiltonian, eigvals, lin_comb_states, weights)

        # Checks if returned vectors are orthogonal eath other
        assert _inner_product_comp_basis_states(
            lin_comb_states[0], lin_comb_states[1]
        ) == pytest.approx(0.0, abs=1e-5)


def test_project_state() -> None:
    qubit_count = 6
    comp_basis_states = [
        ComputationalBasisState(qubit_count, bits=0),
        ComputationalBasisState(qubit_count, bits=4),
        ComputationalBasisState(qubit_count, bits=8),
        ComputationalBasisState(qubit_count, bits=10),
        ComputationalBasisState(qubit_count, bits=20),
        ComputationalBasisState(qubit_count, bits=21),
    ]

    state1 = (
        [1 / np.sqrt(6), 0.0, 1j / np.sqrt(3), 1 / np.sqrt(2)],
        [
            comp_basis_states[0],
            comp_basis_states[1],
            comp_basis_states[2],
            comp_basis_states[3],
        ],
    )
    basis = [
        comp_basis_states[0],
        comp_basis_states[4],
        comp_basis_states[3],
        comp_basis_states[4],
    ]

    projected = _project_state(state1, basis)
    assert projected[0] == [1 / np.sqrt(6), 0.0, 1 / np.sqrt(2), 0.0]
    assert projected[1] == basis


def _inner_product_comp_basis_states(
    state1: ComputationalBasisSuperposition, state2: ComputationalBasisSuperposition
) -> complex:
    state2_coefs, state2_comp_basis_states = state2
    state2_bits = [basis.bits for basis in state2_comp_basis_states]

    sum: complex = 0.0
    for state1_coef, comp_basis_state in zip(*state1):
        if comp_basis_state.bits in state2_bits:
            index_coef = state2_bits.index(comp_basis_state.bits)
            sum += (
                state1_coef.conjugate()
                * state2_coefs[index_coef]
                * 1.0j
                ** (
                    -1 * comp_basis_state.phase
                    + state2_comp_basis_states[index_coef].phase
                )
            )
    return sum
