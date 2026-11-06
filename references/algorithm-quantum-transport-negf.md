# NEGF Quantum Transport

## Scope and Reference Use

Follow [SKILL.md](../SKILL.md) for task routing, execution permission, and reference-only script rules.

The non-equilibrium Green function (NEGF) method describes transport through a
device coupled to leads. This implementation computes stationary coherent
transport through a clean two-dimensional square lattice with two identical
semi-infinite leads at one selected energy. It uses Kwant to construct the
Hamiltonian and lead self-energies, then UnitaryLab QSVT to solve injection
responses and obtain transmission and bond-current observables.

This reference covers the public `NEGFAlgorithm` workflow, its Kwant/QSVT
reconstruction, and comparison with an analytic clean-lead benchmark. It does
not cover finite-bias integration or interacting transport, and its dense
statevector simulation does not establish an end-to-end quantum speedup.

Source pointers: [algorithm-library usage example](../scripts/negf_algorithm.py)
and [independent teaching implementation](../scripts/negf_implementation.py).
The former demonstrates the public class call; the latter uses an analytic
transverse-mode reconstruction with its own bond order and does not define the
algorithm-library API. The mathematical sections and manual example below
describe the Kwant/QSVT reconstruction. These scripts are reading references;
generated user code must not import or execute them.

## Scope and Limitations

The source handles a clean `L × W` square lattice with onsite `4t`, hopping `-t`,
hard transverse walls and two identical semi-infinite leads of width `W`. It
chooses one energy between the first two transverse subband bottoms. It uses
Kwant for Hamiltonians and self-energies and UnitaryLab QSVT for the injection
responses. This is stationary coherent transport at one energy, without bias
integration, occupations, temperature dependence, disorder or interactions.
The bond-current expression has the source's sign and units, without an SI
current prefactor; it must not be reported as finite-bias amperes.

The source assembles sparse device matrices but densifies the solver matrix,
pads it to a power of two and executes full statevectors with `backend="numpy"`
and `dtype=np.complex128`. It exposes no backend/device/shots/seed parameter.
Dense SVD, polynomial calibration, state preparation, circuit simulation and
classical observable extraction are substantial costs. This workflow establishes
no end-to-end quantum speedup. Only injection columns are solved, rather than
materializing the device inverse in the production path.

The manual example follows the Kwant and QSVT computation in `NEGFAlgorithm.run`
through the transport observables. Plotting, logging and file exports belong to
the wrapper. The separate [analytic reconstruction](../scripts/negf_implementation.py)
uses transverse modes for a clean-lead benchmark and defines its own bond order.
Compare its currents with the Kwant result by matching directed site pairs.

## Overview

1. Validate `L`, `W`; derive the single-channel injection energy.
2. Build device and semi-infinite leads; obtain `H_C` and local self-energies.
3. Embed the self-energies in `A = E I - H_C - Sigma_L - Sigma_R`.
4. Factor each broadening into retained injection channels, `Gamma = C C†`.
5. Calibrate the reciprocal polynomial and build a shared QSVT core.
6. Execute a complete state-preparation-plus-QSVT circuit per injection column.
7. Restore amplitudes, norms and scaling; evaluate residuals and observables.
8. Export plots, actual injection circuits and supported QASM2 files.

## Prerequisites

- Complex linear algebra, retarded Green functions and coherent transport.
- NumPy, SciPy, Matplotlib, Kwant and a compatible UnitaryLab installation for
  the source/API path; `unitarylab-algorithms` must include the new NEGF module.
- The manual example needs Kwant, NumPy, SciPy and UnitaryLab's `Circuit`, `library.solve`,
  `normalize_matrix` and `_indicator`.
- A NumPy ABI compatible with compiled Kwant extensions. An import error such as
  `numpy.dtype size changed` is an environment problem before algorithm execution.

## Reference Implementation Example

This small example calls the installed `NEGFAlgorithm.run()` and prints its
public results. The algorithm constructs the transport model and QSVT circuits.

```python
from unitarylab_algorithms.quantum_transport.negf import NEGFAlgorithm

result = NEGFAlgorithm(text_mode="plain", algo_dir="results/negf-example").run(
    L=2, W=2, a=1.0, t=1.0, inverse_tolerance=2e-3,
)
print(result["transmission"])
print(result["injection_residuals"])
print(result["qasm_paths"])
```

The source exports the class through the leaf, category and package root.
The example assumes an installed package containing those exports and makes no
`sys.path` changes. `test(L=4, W=4, a=1.0, t=1.0,
inverse_tolerance=2e-3)` is the source's convenience entry with legacy text mode.

## Core Parameters Explained

| Parameter | Type | Default | Actual behavior |
|---|---|---|---|
| Constructor `text_mode` | `str` | `"plain"` | BaseAlgorithm text formatting; source test uses `"legacy"`. |
| Constructor `algo_dir` | `str` or `None` | `None` | Creates `cwd/results/quantum_transport/negf` when omitted. |
| `L` | `int` | `4` | Integer or NumPy integer at least 2, excluding booleans; number of x sites. |
| `W` | `int` | `4` | Same validation; y sites and both lead widths. |
| `a` | `float` | `1.0` | Lattice constant used by Kwant; positive finite values intended, no explicit wrapper range check. |
| `t` | `float` | `1.0` | Hopping energy scale, intended positive and in eV; no explicit wrapper range check. |
| `inverse_tolerance` | `float` | `2e-3` | Finite and strictly between 0 and 1, validated during calibration; reciprocal polynomial relative-error threshold. |

There is no independent `ENERGY` input. With
`e_m = 2t[1-cos(m*pi/(W+1))]`, the source takes `ENERGY=(e_1+e_2)/2`.
For intended positive `t`, this is inside the first propagating subband, below
the second. `a` does not rescale the fixed tight-binding hopping `t` here.
`parameters.json` describes UI metadata, not additional Python runtime checks.

## Return Fields

The returned dictionary has `status="ok"` on completion, whereas the internal
text-summary status is `"success"`. A successful status is not a residual bound.

| Field | Type / interpretation |
|---|---|
| `status` | String from BaseAlgorithm: normally `"ok"`. |
| `circuit_path` | String path of the first actual injection circuit SVG. |
| `plot` | List of `{format, filename}` descriptors: summary, geometry, LDOS, current, remaining SVGs and successful QASM exports. Filenames may be relative to `algo_dir`. |
| `circuit` | First executed `Circuit`, including input preparation. |
| `ENERGY` | Float, automatically chosen injection energy. |
| `left_channels`, `right_channels` | Integers counting retained broadening eigenmodes. |
| `transmission` | Float, dimensionless transmission from left injection responses. |
| `injection_residuals` | NumPy vector, one relative residual per left-then-right RHS. |
| `epsilon` | Float, calibrated SDK reciprocal-polynomial parameter. |
| `inverse_polynomial_error` | Float, maximum sampled reciprocal-polynomial relative error. |
| `scaling_factor` | SDK scalar restoring unnormalized linear responses. |
| `rhs_norms` | NumPy vector of original injection-column norms. |
| `system_size` | Integer `L*W`, before power-of-two padding. |
| `qasm_export_errors` | List of `{circuit, reason}` for `NotImplementedError` during QASM2 export. |
| `ldos` | NumPy vector of length `L*W`, in Kwant scattering-site order. |
| `bond_current` | NumPy vector in the directed-bond order of `fsyst.graph`. |
| `circuits` | All executed circuits, left then right injection channels. |
| `circuit_paths` | SVG path for each circuit. |
| `qasm_paths` | Only successfully exported QASM2 paths. |

The public result does not expose `A`, `H_C`, `C_L`, `C_R`, `X_L`, `X_R` or the
Kwant finalized system. Reconstruct them independently for comparisons; do not
invent result keys.

## Implementation Architecture

`NEGFAlgorithm` extends `BaseAlgorithm`. Its `run()` builds a sparse Kronecker
Hamiltonian as explanatory model data and independently builds the Kwant system;
the actual `H_C` is `fsyst.hamiltonian_submatrix(sparse=True)`. Local `W×W`
self-energies are embedded using `fsyst.lead_interfaces`.

`KwantSolvers.lead_injection_vectors()` diagonalizes the Hermitian part of local
`Gamma`. Its threshold is `100*eps_machine*max(1, W)*scale`, where `scale`
includes both the maximum absolute broadening eigenvalue and twice the spectral
norm of the local self-energy. It rejects negative eigenvalues below minus that
threshold and retains eigenvalues strictly above the positive threshold. This
prevents near-band-edge cancellation noise from becoming spurious channels.

`solve_unitarylab()` pads `A` to dimension `N=2**ceil(log2(L*W))`; unused diagonal
entries are `||A||_2`. It calls `solve(padded, e0, method="qsvt", ...)` once.
Preparing `e0` in that reusable core is compatible with subsequent injection
preparation because it is the all-zero system basis state. For each normalized
injection, it explicitly initializes low-order system qubits and appends the
shared core. Full circuits execute from zero, without an external initial state.

`_unitarylab_epsilon()` locally reproduces spectral normalization, computes
`kappa`, SDK `alpha`, and checks the actual `_indicator` coefficients with their
returned `beta`. It samples 20001 grid points plus all singular values. Up to 32
attempts increase approximation order. Singular matrices, floating-point range
failures and unmet polynomial tolerance raise errors. This sampled check is not
a rigorous supremum proof and is not a final linear-equation residual check.

## Understanding the Key Quantum Components

The matrix is generally complex and non-Hermitian because of retarded lead
self-energies. The SDK applies its Nagy block encoding and reciprocal QSVT
polynomial. The system register has `log2(N)` low-order qubits, followed by two
auxiliary qubits in the inspected SDK path. Auxiliary-zero postselection is
represented in statevector simulation by slicing the first `N` amplitudes.

For each RHS `b`, recover `x` using the first `L*W` output amplitudes multiplied
by `||b|| * scaling_factor`. Do not normalize the recovered solution, discard
complex phases, or infer transmission from computational-basis probabilities.
The algorithm needs amplitudes and their relative phases for classical overlaps.

## Theory-to-Code Mapping

| Mathematical object | Source action |
|---|---|
| `H_C = H_x tensor I_W + I_L tensor H_y` | Onsite `4t`, nearest-neighbor hopping `-t`; (x, y) ordering. |
| `Sigma^r_L/R(E)` | Kwant semi-infinite lead `selfenergy(ENERGY)`. |
| `Gamma=i(Sigma-Sigma†)` | Local retarded broadenings, positive semidefinite up to roundoff. |
| `Gamma=C C†` | Eigenvectors multiplied by square roots of retained eigenvalues. |
| `G^r=(E I-H_C-Sigma_L-Sigma_R)^(-1)` | Solve `A X=[C_L,C_R]`; avoid full inverse in production. |
| `X_L=G^r C_L`, `X_R=G^r C_R` | Restore the complex auxiliary-zero amplitudes for each channel. |
| `T=Tr(Gamma_R G^r Gamma_L G^a)` | `norm(C_R.conj().T @ X_L, "fro")**2`. |
| `rho_i=(X_L X_L†+X_R X_R†)_ii/(2*pi)` | Sum squared response amplitudes at each site. |
| `J_ij=2 Im[H_ij sum_a conj(X_L[i,a]) X_L[j,a]]` | `2*imag(H_C[i,j]*vdot(X_L[i],X_L[j]))`. |
| Relative equation residual | `norm(A@x-b)/norm(b)` separately for every injection. |

## Mathematical Deep Dive

For a clean lead with hard-wall transverse modes,
`U_ym=sqrt(2/(W+1))*sin((y+1)*m*pi/(W+1))`. The transverse energy is
`e_m=2t(1-cos(m*pi/(W+1)))`; each mode is a longitudinal chain with onsite
`e_m+2t` and hopping `-t`. Set `z_m=E-e_m-2t`. Its retarded surface self-energy is

```text
sigma_m = (z_m - i sqrt(4t^2-z_m^2))/2,           |z_m| <= 2t
sigma_m = (z_m - sign(z_m) sqrt(z_m^2-4t^2))/2,   |z_m| > 2t
Sigma = U diag(sigma_m) U^T.
```

The imaginary part is nonpositive; outside a propagating band the decaying root
is selected. This supplies an independent exact lead reference for the current
separable model. It gives the same interface block at both ends, embedded into
the first and last `W` sites. Lattice constant `a` sets geometry but not these
matrix elements when `t` remains fixed.

For this clean connected model without decoupled bound-state spectral terms or
an additional imaginary regulator, the identity
`i(G^r-G^a)=G^r(Gamma_L+Gamma_R)G^a` yields
`rho_i=-Im(G^r_ii)/pi`. The independent benchmark checks this equality, response
residuals, one propagating channel on each side and unit transmission for the
clean perfectly matched device. For real symmetric hoppings, the source's
current formula also obeys `J_ij=-J_ji`.

Densifying a device with `n=L*W` sites needs quadratic matrix storage; dense
factorizations/SVD cost cubic work in usual algorithms. These are classical
preprocessing/benchmark costs, not a QSVT query-complexity guarantee. The
independent benchmark constructs a full inverse only to check the spectral LDOS
identity; the source algorithm does not.

## Hands-On Example

Inspect transport observables and numerical residuals for a clean `2×2` device:

```python
from unitarylab_algorithms.quantum_transport.negf import NEGFAlgorithm

algorithm = NEGFAlgorithm(text_mode="plain", algo_dir="results/negf-transport")
result = algorithm.run(L=2, W=2, a=1.0, t=1.0, inverse_tolerance=2e-3)

print("Energy:", result["ENERGY"])
print("Channels (left, right):", result["left_channels"], result["right_channels"])
print("Transmission:", result["transmission"])
print("LDOS:", result["ldos"])
print("Bond current:", result["bond_current"])
print("Relative residuals:", result["injection_residuals"])
print("Polynomial error:", result["inverse_polynomial_error"])
print("QASM export errors:", result["qasm_export_errors"])
```

For this model, the classical expectation is `ENERGY=2`, one propagating channel
per lead and transmission `T=1`; the quantum approximation should approach these
values. These are reference expectations, not results observed from this example.
`inverse_tolerance` controls the sampled reciprocal-polynomial error, not a
bound on final transmission or equation residuals. Inspect both errors when
assessing a run. QASM export limitations are reported separately from transport.

For further numerical verification, compare complex responses and observables
with the same classical model. Scale absolute LDOS differences by `t` because
LDOS has inverse-energy units; compare currents using the same directed bonds.
Keep such acceptance checks in validation code rather than in the usage example.

## Minimal Manual Implementation

Start with a clean strip of length `L` and width `W`, with integer dimensions at
least two and positive hopping `t`. The three stages below construct the open
device, solve for its injection responses, and read out transport observables.
Changing `L` or `W` changes the matrix, the register size and the polynomial
calibration. The lattice spacing sets site coordinates and drops out of this
model at fixed hopping.

**Device and contacts (source steps 2–4).** The Kronecker sum in step 2
specifies the square-lattice model. Step 3 realizes that model in Kwant with
onsite energy `4*t`, nearest-neighbor hopping `-t` and two attached leads.
Step 4 takes the scattering Hamiltonian, interface indices and self-energies
from that finalized system. An interface index maps a local lead orbital to a
row of the scattering Hamiltonian. Use the same map for its self-energy and
its injection vectors.

```python
import kwant
import numpy as np
from scipy import sparse
from unitarylab import Circuit
from unitarylab.library import solve
from unitarylab.library.block_encoding.tools import normalize_matrix
from unitarylab.library.linear_solver.qsvt import _indicator


def open_device(L, W, a, t):
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
           or n < 2 for n in (L, W)):
        raise ValueError("L and W must be integers >= 2")
    band_bottoms = 2 * t * (1 - np.cos(np.pi * np.array([1, 2]) / (W + 1)))
    energy = band_bottoms.mean()

    lattice = kwant.lattice.square(a, norbs=1)
    device = kwant.Builder()
    device[(lattice(x, y) for x in range(L) for y in range(W))] = 4 * t
    device[lattice.neighbors()] = -t
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead[(lattice(0, y) for y in range(W))] = 4 * t
    lead[lattice.neighbors()] = -t
    device.attach_lead(lead)
    device.attach_lead(lead.reversed())
    system = device.finalized()

    H = system.hamiltonian_submatrix(sparse=True).tocsr()
    A = energy * sparse.eye(H.shape[0], format="csr") - H
    injections = []
    for lead, interface in zip(system.leads, system.lead_interfaces):
        sigma = lead.selfenergy(energy)
        rows, cols = np.meshgrid(interface, interface, indexing="ij")
        A -= sparse.csr_matrix((sigma.ravel(), (rows.ravel(), cols.ravel())),
                               shape=H.shape)

        gamma = 1j * (sigma - sigma.conj().T)
        values, vectors = np.linalg.eigh((gamma + gamma.conj().T) / 2)
        scale = max(np.max(np.abs(values)), 2 * np.linalg.norm(sigma, 2))
        cutoff = 100 * np.finfo(float).eps * len(interface) * scale
        if values.min() < -cutoff:
            raise ValueError("Check the retarded self-energy sign")
        keep = values > cutoff
        C = np.zeros((H.shape[0], np.count_nonzero(keep)), dtype=complex)
        C[interface] = vectors[:, keep] * np.sqrt(values[keep])
        injections.append(C)
    left, right = injections
    if left.shape[1] + right.shape[1] == 0:
        raise ValueError("No propagating injection channels at this energy")
    return system, energy, H, A, left, right
```

**Reciprocal polynomial.** Calibrate on the padded matrix using the same spectral
normalization and polynomial coefficients as the solver. On a normalized singular
value `s`, the scaled polynomial should approximate `1/s`. Multiplying by `s`
gives the dimensionless error below. The helpers used here match UnitaryLab
`1.5.1`. In particular, `_indicator` is private and couples this step to the SDK
version. Continue in the same Python file.

```python
def calibrate(matrix, tolerance):
    if not 0 < tolerance < 1:
        raise ValueError("inverse_tolerance must lie between zero and one")
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] <= np.finfo(float).eps * len(matrix) * singular[0]:
        raise ValueError("The padded system is numerically singular")
    normalized = matrix / singular[0]
    _, alpha = normalize_matrix(normalized, norm_type="spectral")
    kappa = singular[0] / singular[-1]
    scaled = (singular / singular[0]) / alpha
    grid = np.unique(np.r_[np.linspace(scaled[-1], scaled[0], 20001), scaled])
    log_ratio = -np.log(tolerance)  # largest singular value after normalization is 1
    for _ in range(32):
        epsilon = kappa * alpha * np.exp(-log_ratio)
        order = np.ceil((kappa * alpha)**2 * log_ratio)
        if not 0 < epsilon < kappa * alpha or not np.isfinite(order):
            break
        if 4 * order > epsilon:
            odd, beta = _indicator(kappa, alpha, epsilon, 2.0)
            coefficients = np.zeros(2 * len(odd))
            coefficients[1::2] = odd
            p = np.polynomial.chebyshev.chebval(grid, coefficients)
            error = np.max(np.abs(grid * p * beta * kappa * alpha - 1))
            if error <= tolerance:
                return epsilon
        log_ratio *= 2
    raise RuntimeError("Could not calibrate the reciprocal polynomial")
```

**Injection responses and readout.** QSVT needs a power-of-two matrix dimension.
An unused diagonal block of spectral norm `||A||₂` keeps the padded matrix
invertible while the zero-padded injections confine the physical solution to
the original sites. Build the solver core once, then prepare each injection on
the low-order system qubits. The auxiliary-zero sector starts at amplitude zero.
Recover the physical amplitudes by restoring the input norm and SDK scale.

```python
def injection_responses(A, B, tolerance):
    A = A.toarray()  # Source step 5 enters the dense QSVT solver here.
    n = len(A)
    system_bits = (n - 1).bit_length()
    size = 1 << system_bits
    padded = np.eye(size, dtype=complex) * np.linalg.norm(A, 2)
    padded[:n, :n] = A
    epsilon = calibrate(padded, tolerance)
    e0 = np.zeros(size)
    e0[0] = 1
    solver = solve(padded, e0, method="qsvt", epsilon=epsilon,
                   backend="numpy", dtype=np.complex128, logging=False)
    core = solver.circuit
    X = np.zeros_like(B)
    for j, b in enumerate(B.T):
        norm = np.linalg.norm(b)
        initial = np.zeros(size, dtype=complex)
        initial[:n] = b / norm
        circuit = Circuit(core.get_num_qubits())
        circuit.initialize(initial, list(range(system_bits)))
        circuit.append(core, list(range(core.get_num_qubits())))
        state = np.asarray(circuit.execute(backend="numpy", dtype=np.complex128).state)
        X[:, j] = norm * solver.scaling_factor * state[:n]
    return X


L, W, a, t = 2, 2, 1.0, 1.0
system, energy, H, A, left, right = open_device(L, W, a, t)
B = np.column_stack((left, right))
X = injection_responses(A, B, tolerance=2e-3)
X_left = X[:, :left.shape[1]]
transmission = np.linalg.norm(right.conj().T @ X_left, "fro")**2
ldos = np.sum(np.abs(X)**2, axis=1) / (2 * np.pi)
bonds = list(system.graph)  # Directed edges in the source plotting order.
current = [2 * np.imag(H[i, j] * np.vdot(X_left[i], X_left[j])) for i, j in bonds]
residuals = np.linalg.norm(A @ X - B, axis=0) / np.linalg.norm(B, axis=0)
print("Energy and transmission:", energy, transmission)
print("LDOS in Kwant site order:", ldos)
print("Directed bonds and currents:", list(zip(bonds, current)))
print("Relative residual per injection:", residuals)
```

Transmission projects the left-injected response onto the right contact. LDOS
sums responses from both leads. The directed current uses only left injection
and preserves the complex phase between neighboring sites. For a padding
example, set `L=3, W=2`. Six physical sites occupy eight basis states, and only
the first six amplitudes enter the observables.

## Debugging Tips

- Import failure mentioning `unitarylab_chemistry`: some algorithm-package
  versions eagerly import chemistry exports even for a NEGF leaf import. Check
  the full traceback and install compatible package dependencies; changing the
  NEGF import to another package path does not bypass package initialization.
- Missing NEGF export: inspect the installed algorithm package/version; the new
  checkout alone does not update a wheel. Install a distribution containing NEGF.
- `numpy.dtype size changed` on Kwant import: correct the NumPy/Kwant binary
  compatibility in the selected environment before diagnosing transport physics.
- Missing `_indicator` or a changed `normalize_matrix` signature: SDK version
  incompatibility; calibration depends on these version-sensitive helpers.
- Negative broadening eigenvalues: check the retarded sign and self-energy scale;
  do not blindly clip materially negative eigenvalues.
- No channels: check energy, positive `t`, mode thresholds and roundoff cutoff.
- Bad residual despite good polynomial error: check SDK normalization, core
  conventions, auxiliary-zero slice, original RHS norm and complex phase.
- Current sign mismatch: compare the ordered `(i,j)` bond, hopping sign and `vdot`
  argument order; the source vector order is exactly `fsyst.graph`.
- Large circuits: dense padding and conditioning increase resource requirements;
  start with `2×2`, not the default `4×4`, for a smoke run.
- Empty `qasm_paths` with `status="ok"`: inspect `qasm_export_errors`; unsupported
  QASM export is recorded without discarding successful transport outputs.

## Reference Implementation

Authoritative local sources are `quantum_transport/negf/algorithm.py`, its
`README_en.md`, `README_zh.md`, `parameters.json`, the leaf/category/root exports,
and `algo_base.py` from `unitarylab_algorithms`. The source's delegated solver is
`unitarylab.library.solve(..., method="qsvt")`; inspect its actual QSVT
implementation and normalization helpers when adapting SDK versions.

Default files under `results/quantum_transport/negf` include `negf_result.txt`,
`kwant_square_system.png`, `kwant_square_ldos.png`, `kwant_square_current.png`,
`negf_circuit.svg`, additional channel SVGs and supported per-channel QASM2 files.
The first SVG is returned in `circuit_path`; subsequent SVGs are in `plot`.

The inspected source README records an SDK Nagy decomposition limit of 6 target
qubits for QASM2 export, corresponding here to padded system dimension `N<=32`
(`L*W<=32`). This is SDK-version-specific, not a mathematical transport limit.
`NotImplementedError` during export is recorded per circuit and execution
continues; other exceptions propagate. The drawing may show only its default
25 layout layers, so SVG extent does not prove total gate count. QASM2 omits
uncontrolled global phases: physical state equivalence does not establish
amplitude-level equality for linear-response reconstruction.
