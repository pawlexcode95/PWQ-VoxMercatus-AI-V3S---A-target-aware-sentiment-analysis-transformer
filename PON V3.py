# =======| Imports |===================================================================================================#
import torch
import math as m
from typing import *
import time as t
# =======| Debug Printer Class |===================================================================================================#
class DebugPrint:
    def __init__(self, condition:bool=False) -> None:
        self.condition = condition
    def _log(self, msg:str) -> None:
        if self.condition:
            print(f"\n==========| {msg} |==========")
# =======| PWQ PON V2 Class |==================================================================================================#
class PWQ_PON_V_3_0:
    def __init__(self, matrix:torch.tensor=None):
        self.d = matrix.device
        self.p = matrix.dtype
        self.A = matrix
        self.Acopy = torch.clone(self.A)
        self.R = torch.clone(self.A)
        self.Q = PWQ_PON_V_3_0.iden(len(self.A), self.d, self.p)
        self.I = PWQ_PON_V_3_0.iden(len(self.A), self.d, self.p)
        self.eps = 1e-6

    @staticmethod
    def pon(w: torch.Size, layers: int, mean:float=0.0, std:float=1.0, bias_return:bool=False, perturb_bias:bool=True, debug_prints:bool=False) -> Union[torch.nn.Parameter, Tuple[torch.nn.Parameter, torch.nn.Parameter]]:
        """
        Perturbated Ortho-Normalized method which returns torch.nn.Parameter object which is the initialized weight matrix,
        and the initialized bias matrix as the perturbator matrix.
        """
        debug = DebugPrint(debug_prints)
        debug._log(f"Current Task: size-{w}, layers-{layers}")
        _in, _out = w
        in_bigger_out = True if _in > _out else False
        debug._log(f"Initializing PON V3")
        A = torch.normal(mean, std, size=(_in, _out)) if in_bigger_out else torch.normal(mean, std, size=(_out, _in))
        pwq_poni = PWQ_PON_V_3_0(matrix=A)
        debug._log(f"Constructed Initial Matrix")
        q, r = pwq_poni.hhrl_(A)
        debug._log(f"QR Decomposition Complete")
        q_c, r_c = pwq_poni.qrec_(q, r, conv_its=5)
        q_c, _ = pwq_poni.qrec_(q_c, r_c, conv_its=2)
        q_c_slcd = q_c[:_out, :].T if in_bigger_out else q_c[:_in, :]
        debug._log(f"QR Error Correction Complete")
        q_ci, pbm = pwq_poni.init(q_c_slcd, _out, layers, perturb_bias=perturb_bias)
        debug._log(f"Perturbating Q Complete")
        if bias_return:
            debug._log(f"Returning weight of shape {q_ci.shape}, and bias of shape {pbm.shape}")
            return torch.nn.Parameter(q_ci).to("cuda"), torch.nn.Parameter(pbm).to("cuda")
        return torch.nn.Parameter(q_ci).to("cuda")

    def init(self, w: torch.Tensor, out: int, layers: int, perturb_bias:bool=True) -> Tuple[torch.Tensor, torch.tensor]:
        """
        Initializer method which returns the perturbated weight matrix, and the pertubator matrix
        """
        default = 5e+0
        if 1 <= layers <= 16:
            l_div = 7.55
        elif 17 <= layers <= 20:
            l_div = 4.75
        elif 21 <= layers <= 32:
            l_div = default
        elif 33 <= layers <= 48:
            l_div = default
        elif 49 <= layers < 64:
            l_div = default
        elif layers >= 64:
            l_div = 4e-5

        P = torch.normal(0, 1/l_div, size=w.size())
        b = torch.normal(0, 1/l_div, size=(out, )) if perturb_bias else torch.zeros(size=(out, ))
        return w + P, b

    def qrec_(self, Q: torch.Tensor, R:torch.Tensor, conv_its: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        QR Error Correction method which corrects the rounding error from the householder reflection algorithm and returns
        the corrected Q and R matrices.
        """
        r_cur, q_cur = R, Q
        for i in range(conv_its):
            E = self.Acopy - q_cur @ r_cur  # Compute error
            delta_R = q_cur.T @ E  # Compute error delta of R
            R_c = r_cur + delta_R  # Correct R
            Q_s = PWQ_PON_V_3_0.solve_triangular(R_c.T, self.Acopy.T, upper=False)
            q_cur = Q_s.T
            r_cur = R_c
        error = torch.max(torch.abs(self.Acopy - q_cur @ r_cur))
        return q_cur, r_cur

    def hhrl_(self, A:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The Householder Reflection algorithm method, which decomposes matrix A into Q and R based on the reflection, each
        column vector leaves on another. Returns the decomposed Q and R matrices.
        """
        _m, _n = A.shape
        iterator = min(_m, _n) if _m != _n else _m
        R = torch.clone(self.R)
        Q = torch.clone(self.Q)

        for j in range(iterator):
            x = R[j:, j]
            norm_x = torch.norm(x).item()
            # Choose alpha to avoid cancellation
            alpha = norm_x * (-1) + self.eps
            # Householder vector
            v = x.clone()
            v[0] = v[0] - alpha
            v = v / torch.norm(v)
            # Apply reflector
            v_col = v.unsqueeze(1)
            R[j:, j:] = R[j:, j:] - 2 * v_col @ (v_col.T @ R[j:, j:])
            Q[:, j:] = Q[:, j:] - 2 * (Q[:, j:] @ v_col) @ v_col.T
        self.R = R
        self.Q = Q
        return self.Q, self.R

    @staticmethod
    def solve_triangular(L:torch.Tensor, r:torch.Tensor, upper:bool=False):
        """
        Static helper method utilized for solving Ax = b, x is upper/lower triangular/trapezoidal matrix and b is the unknown.
        Returns the unknown the matrix. PON V3 uptade handles non-square weight initialization as well as square.
        """
        rows, cols = L.shape
        r_rows, r_cols = r.shape
        if rows == cols:
            return torch.linalg.solve_triangular(L, r, upper=upper)
        return torch.linalg.lstsq(L, r).solution

    @staticmethod
    def iden(size: int, device: torch, dtype: torch) -> torch.Tensor:
        """
        Static helper method used for returning any-sized identity matrix. Returns the size sized identity matrix
        in form of torch.tensor
        """
        return torch.eye(size, device=device, dtype=dtype)
