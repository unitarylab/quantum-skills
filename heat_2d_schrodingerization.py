import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from engine import state_preparation
from engine.library import CDiff
from engine.library.schrodingerization import schro_classical


def build_periodic_heat_operator(num_spatial_qubits: int, length: float, a1: float, a2: float):
    grid_size = 2 ** num_spatial_qubits
    dx = length / grid_size
    laplacian_1d = CDiff(N=grid_size, dx=dx, order=2, boundary='periodic').get_matrix()
    identity = sp.eye(grid_size, format='csc')
    operator = a1 * sp.kron(laplacian_1d, identity, format='csc')
    operator += a2 * sp.kron(identity, laplacian_1d, format='csc')
    return operator, grid_size, dx


def build_initial_field(grid_size: int, length: float):
    x = np.arange(0.0, length, length / grid_size)
    y = np.arange(0.0, length, length / grid_size)
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
    field = np.sin(2 * np.pi * x_grid / length) * np.sin(2 * np.pi * y_grid / length)
    return field.reshape(-1), x_grid, y_grid


def main():
    length = 1.0
    final_time = 0.02
    diffusion_x = 0.8
    diffusion_y = 0.3
    num_spatial_qubits = 3

    operator, grid_size, _ = build_periodic_heat_operator(
        num_spatial_qubits=num_spatial_qubits,
        length=length,
        a1=diffusion_x,
        a2=diffusion_y,
    )
    initial_field, _, _ = build_initial_field(grid_size=grid_size, length=length)

    solution = np.asarray(
        schro_classical(
            operator,
            initial_field,
            T=final_time,
            na=7,
            R=4,
            order=3,
        )
    )
    reference = expm_multiply(final_time * operator, initial_field)
    relative_l2_error = np.linalg.norm(solution - reference) / np.linalg.norm(reference)

    normalized_solution = solution / np.linalg.norm(solution)
    preparation_circuit = state_preparation(normalized_solution)
    zero_state = np.zeros_like(normalized_solution, dtype=complex)
    zero_state[0] = 1.0
    prepared_state = np.asarray(preparation_circuit.execute(zero_state.copy()))
    state_preparation_error = np.linalg.norm(prepared_state - normalized_solution)
    probabilities = np.abs(prepared_state) ** 2

    print(f'grid_size={grid_size}x{grid_size}')
    print(f'solution_rel_l2_error_vs_matrix_exp={relative_l2_error:.6e}')
    print(f'state_preparation_error={state_preparation_error:.6e}')
    print(f'max_temperature={np.max(solution):.6f}')
    print(f'min_temperature={np.min(solution):.6f}')
    print('largest_probabilities=', np.sort(probabilities)[-5:])


if __name__ == '__main__':
    main()