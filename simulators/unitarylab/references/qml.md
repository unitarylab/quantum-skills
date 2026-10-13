# Native Quantum Machine Learning

Use this reference for the SDK's `unitarylab.library.qml`, not the separate
`unitarylab_algorithms` examples. For algorithm theory or a packaged VQC/QCBM/CVQNN
workflow, also read the matching algorithm skill.

## Public Modules

| Need | Module and representative APIs |
|---|---|
| Execution backend | `backends`: `TorchBackend`, `UnitaryLabBackend`, `CVTorchBackend`, `create_backend`, `available_backends` |
| Classification/regression | `models`: `VQCClassifier`, `VQRegressor`, `QuantumClassifier`, `QuantumRegressor`, `QCNNClassifier`, `QCNNRegressor`, data-reuploading models |
| Generative/CV models | `models`: `QuantumBornMachine`, `QuantumGenerator`, `CVQuantumClassifier` |
| PyTorch modules with owned parameters | `nn`: `TorchQMLLayer`, `TorchBornMachine`, `TorchQuantumGenerator`, `QuantumGAN`, `Quanvolution2d` |
| Encodings, ansatzes, readouts | `layers`: angle/amplitude/ZZ encodings, hardware-efficient/QCNN ansatzes, Pauli/probability/state/density/sample readouts |
| Quantum kernels | `kernels`: `StatevectorFidelityKernel`, `OverlapKernel`, `ProjectedQuantumKernel`, classifier/regressor, KMeans, PCA, anomaly detector |
| Dataset preparation | `data`: `Dataset`, `load_csv_dataset`, standardization, padding, train/test and stratified/k-fold splits |
| Loss/objective | `objectives`: `SupervisedObjective`, Born-machine KL/NLL/MMD objectives, adversarial losses |
| Training and diagnostics | `engine`: `Trainer`, `TorchOptimizer`, `TorchScheduler`, callbacks, evaluation, gradient-variance/trainability reports |
| Noise/readout correction | `noise`: channels, `NoiseModel`, `ReadoutErrorModel`, `mitigate_readout_probabilities` |
| Execution contracts | `execution`: `ExecutionConfig`, `ExecutionRequest`, `BackendResult`, capabilities, batch/decomposition/measurement transforms and traces |
| Evaluation metrics | `metrics`: classification accuracy/F1/confusion matrix and regression MSE/MAE/R² |

Import symbols from their submodules. The `qml` root loads subpackages lazily; it does
not re-export every model or trainer. Availability depends on the installed SDK revision.

## Parameter Ownership and Forward Calls

Core models keep the trainable vector outside the model. `initialize_parameters(seed=0)`
returns a Torch tensor of shape `(model.parameter_count,)`; `validate_parameters()`
checks its length and finite values. Pass parameters explicitly to `forward(inputs, parameters)`.

```python
import torch
from unitarylab.library.qml.models import VQCClassifier

model = VQCClassifier(num_qubits=2, num_layers=1)
parameters = model.initialize_parameters(seed=7)
probability = model.forward(
    torch.tensor([0.2, 0.4], dtype=torch.float64), parameters,
)
```

The default variational backend is `TorchBackend`. Binary classifiers return probabilities;
multiclass classifiers return logits. Feature widths and readout widths depend on the selected
encoding/model; do not apply one universal input shape to every model.
Use `nn.TorchQMLLayer` when an `nn.Module` should own parameters for native Torch optimization.

## Training and Gradients

`Trainer(optimizer=..., loss_fn=... or objective=..., gradient_fn=..., epochs=..., ...)`
coordinates training. `fit(model, dataset, parameters=None, validation_data=None)` returns
`(parameters, history)`. An optimizer and an objective/loss are required; do not pass both
`loss_fn` and `objective`. `TorchOptimizer` wraps a Torch optimizer class for an explicit
parameter vector; closure-based LBFGS requires the native Torch-module path instead.

`engine` exports autograd, backend VJP, parameter-shift/Jacobian/loss-gradient,
batched/parallel parameter-shift, finite-difference, SPSA, and mixed-gradient routines.
Choose a strategy compatible with the selected backend and circuit's shift recipes;
forward execution support does not imply differentiability.

`UnitaryLabBackend` is forward-only (`autograd=False`). It accepts `shots=None` for
analytic queries or a sampling count, plus `seed`, depolarizing noise, and readout errors.
It rejects a nonempty gate-local QML `NoiseModel`; use the documented Torch backend path
for that model. QML noise classes are distinct from `unitarylab.backend.NoiseModel` used
by `Circuit.execute_density()`.

## Persistence

`engine.save_parameters(parameters, path)` writes a flat NumPy parameter array and returns
a `Path`; `engine.load_parameters(path)` returns a Torch tensor while preserving floating
precision. The file contains parameter values, not the model architecture or optimizer state.
`nn.save_checkpoint` and `nn.load_checkpoint` serve the higher-level Torch-module workflow;
inspect that checkpoint contract before assuming it is interchangeable with parameter arrays.

## Interpretation and Boundaries

- Preserve tensor dtype/device policy, parameter-block order, encoding, readout, and seed.
- Inspect backend capabilities before requesting statevectors, gradients, noise, or sampling.
- Quantum-kernel matrices and dense state simulation can be expensive; start with a small dataset.
- Generated training, checkpoint writes, and dataset loading remain subject to the user's execution
  authorization. The example above is source-guided and has not been run as part of this update.
