import torch
import math
import numpy as np 

class State:
    def __init__(self, data, num_qubits=None):
        """
        Member variable descriptions:

        self._dtype: The numeric type of the data. Defaults to torch.complex128.

        self._num_qubits: The number of qubits (n).

        self._dim: The dimension of the Hilbert space (2^n).

        self._data: A 1-dimensional vector.
        """
        self._dtype = torch.complex128
        
        if isinstance(data, int):
            # 场景 1: 传入比特数
            self._num_qubits = data
            self._dim = 2 ** data
            self._data = torch.zeros(self._dim, dtype=self._dtype)
            self._data[0] = 1.0 + 0.0j
        else:
            # 场景 2 & 3: 传入数据（Tensor, List, Numpy 等）
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=self._dtype)
            # 使用 flatten() 无论输入是 (16,), (1, 16) 还是 (16, 1)，全部变成 (16,)
            self._data = data.flatten().to(self._dtype)
            # 总元素个数
            self._dim = self._data.shape[0]
            
            # 检查维度是否合法 (必须是 2 的幂)
            n_float = math.log2(self._dim)
            if not n_float.is_integer():
                raise ValueError(f"输入数据长度 {self._dim} 不是 2 的幂，无法构成量子态。")
            self._num_qubits = int(n_float)

        # 构造时自动归一化
        self.normalize()

    @property
    def dtype(self): return self._data.dtype
    @property
    def data(self) -> torch.Tensor: return self._data
    @property
    def num_qubits(self) -> int: return self._num_qubits
    @property
    def dim(self) -> int: return self._dim

    def norm(self) -> float:
        return torch.norm(self._data).item()

    def normalize(self):
        """Normalize the quantum state."""
        norm = torch.norm(self._data)
        if norm > 1e-12:
            self._data = self._data / norm
        else:
            raise ValueError("量子态模长接近于 0,无法进行归一化。")
        return self

    def inner_product(self, other) -> complex:
        """Calculate the inner product of two quantum states <self|other>. Return the numerical result."""
        if self.dim != other.dim:
            raise ValueError("维度不匹配，无法计算内积。")
        # torch.vdot 自动对第一个向量取共轭
        return torch.vdot(self._data, other.data).item()

    def tensor(self, other):
        """
        Calculate the tensor product self ⊗ other.

        Return a new State whose bit count is the sum of the two.
        """
        new_data = torch.kron(self._data, other.data)
        return State(new_data)

    def expectation_value(self, matrix: torch.Tensor) -> complex:
        """
        Calculate the expected value of the operator (matrix) in the current state: <psi|O|psi>. Return the numerical result.
        """
        # <psi| (O @ |psi>)
        val = torch.vdot(self._data, torch.mv(matrix.to(self._dtype), self._data))
        return val.item()
    
    def _permute_for_measure(self, target_indices, endian='little'):
        """
        [Internal Auxiliary Functions] Transmutation operations prepared for measuring, probabilistically calculating, or computing the basis of quantum states.

        Returns:

        flat_permuted: The two-dimensional tensor [2^k, rest] prepared for probability computation

        inverse_permutation: The inverse permutation tuple used to recover the state

        k: The target number of bits
        """
        if isinstance(target_indices, int):
            target_indices = [target_indices]
        
        # 确保是列表副本
        target_indices = list(target_indices)
        
        n = self._num_qubits
        k = len(target_indices)

        # 1. 根据端序确定目标轴 (复用你的逻辑)
        if endian.lower() == 'little' or endian.lower() == 'small':
            # 小端序 (Qiskit默认): 比特0是张量最后一个轴
            # target_indices=[0,1] -> axis=[n-1, n-2] -> Permute后 [n-1, n-2, ...]
            # 此时 flattened index 的最高位是 axis[n-1] (即 q0)
            target_axes = [(n - 1) - i for i in target_indices]
        else:
            # 大端序
            target_axes = [i for i in target_indices]
        
        # 2. 准备置换逻辑
        remaining_axes = [ax for ax in range(n) if ax not in target_axes]
        permutation = tuple(target_axes + remaining_axes)
        
        inverse_permutation = [0] * n
        for i, p in enumerate(permutation):
            inverse_permutation[p] = i
        inverse_permutation = tuple(inverse_permutation)

        # 3. 变形、置换与展平
        state_view = self._data.view([2] * n)
        permuted_state = state_view.permute(permutation)
        flat_permuted = permuted_state.reshape(2**k, -1)
        
        return flat_permuted, inverse_permutation, k

    def probabilities_dict(self, target_indices, endian='little', threshold=1e-9):
        """
        Returns a dictionary of probability distributions for the specified bits (read-only, non-collapsed).
        """
        # 1. 调用辅助函数获取处理好的张量
        flat_permuted, _, k = self._permute_for_measure(target_indices, endian)
        
        # 2. 计算概率 (Partial Trace 的效果)
        # dim=1 求范数平方，相当于把剩余的轴(remaining_axes)求和消掉
        probs = torch.norm(flat_permuted, p=2, dim=1)**2
        
        # 3. 格式化输出
        probs_np = probs.detach().cpu().numpy()
        result_dict = {}
        
        total_prob = probs_np.sum()
        
        for idx, p in enumerate(probs_np):
            if p < threshold:
                continue
            
            # 归一化输出（防止精度误差）
            p_normalized = p / total_prob
            
            # 【关键】复用你 measure 中的字符串格式化逻辑
            # bin(idx) 生成的串，如果是 little endian，因为 permute 把 q0 放到了最高维，
            # 所以 idx 的高位对应 q0。需要 [::-1] 翻转才能变成 q_max...q0 (Qiskit Style)
            outcome_bits = bin(idx)[2:].zfill(k)[::-1]
            
            result_dict[outcome_bits] = p_normalized

        return result_dict

    def measure(self, target_indices, endian='little'):
        """
        A measurement is performed on the specified qubit, causing state collapse.

        The underlying logic reuses `_permute_for_measure` to ensure consistency with `probabilities_dict`.
        """
        # 1. 调用辅助函数 (获取用于计算和坍缩的张量视图)
        flat_permuted, inverse_permutation, k = self._permute_for_measure(target_indices, endian)
        
        # 2. 计算概率
        probs = torch.norm(flat_permuted, p=2, dim=1)**2
        
        # 数值稳定性处理
        prob_sum = probs.sum()
        if prob_sum == 0:
             raise RuntimeError("Quantum state has zero norm.")
        probs = probs / prob_sum
        
        # 3. 采样 (选择一个结果索引)
        sample_idx = torch.multinomial(probs, 1).item()
        
        # 4. 状态坍缩 (Collapse)
        # 只保留选中的那个 slice，其他置零
        new_flat = torch.zeros_like(flat_permuted)
        selected_slice = flat_permuted[sample_idx]
        
        slice_norm = torch.norm(selected_slice)
        if slice_norm > 0:
            new_flat[sample_idx] = selected_slice / slice_norm
        
        # 5. 还原张量结构 (使用逆置换)
        n = self._num_qubits
        new_state_view = new_flat.view([2] * n)
        # 这一步非常关键：必须把轴换回原本的 (q_n-1 ... q_0) 顺序
        new_state_view = new_state_view.permute(inverse_permutation)
        self._data = new_state_view.reshape(-1)

        # 6. 返回结果字符串
        # 保持你原始代码中的字符串生成逻辑
        outcome_bits = bin(sample_idx)[2:].zfill(k)[::-1]
        
        return outcome_bits
        
    def probabilities(self) -> torch.Tensor:
        """Calculate the probability distribution of all ground states and return the results (torch.Tensor type)."""
        return torch.abs(self._data) ** 2

    def sample_counts(self, shots: int = 1024) -> dict:
        """Simulated quantum measurement sampling"""
        probs = self.probabilities()
        indices = torch.multinomial(probs, shots, replacement=True)
        counts = {}
        for idx in indices:
            label = bin(idx.item())[2:].zfill(self._num_qubits)
            # 如果需要在这里也匹配 Qiskit 的 Little Endian (q_n...q0)，
            # 可以对 label 进行 [::-1] 处理，具体取决于你的需求。
            # 目前保持原样。
            counts[label] = counts.get(label, 0) + 1
        return counts

    def __repr__(self):
        # 修改点：打印名称变为 State
        return (            
            f"\n State : \n"
            f"  qubits  : {self._num_qubits}\n"
            f"  dtype   : {self._dtype}\n"
            f"  vector  : {self._data}\n"
            )
    
    def calculate_state(self, target_indices, endian='little', threshold=1e-5):
        """
        [Pure Version] Calculates the probability distribution of a specified subset of bits.

        It does not print any output, but returns a detailed dictionary containing the binary string, the probability, and the corresponding integer.

        Returns:

        dict: Structure as follows:

        {
        '00': {'prob': 0.5, 'int': 0},

        '11': {'prob': 0.5, 'int': 3}

        }
        """
        # 1. 获取基础数据 (keys 已经是符合视觉习惯的 MSB...LSB 字符串)
        raw_probs = self.probabilities_dict(target_indices, endian, threshold)
        
        detailed_result = {}
        
        # 2. 遍历并补充整数信息
        for bin_str, prob in raw_probs.items():
            # 直接转换二进制串为整数 (probabilities_dict 保证了这是物理真值)
            if endian=="big":
                int_val = int(bin_str[::-1], 2)
            else:
                int_val = int(bin_str, 2)
            
            # 构建详细信息的字典
            detailed_result[bin_str] = {
                'prob': prob,
                'int': int_val
            }
            
        return detailed_result