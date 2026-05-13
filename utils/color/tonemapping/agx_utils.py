from __future__ import annotations
import torch
from typing import Iterable, Tuple

# This file is a port of functions from https://github.com/MrLixm/AgXc/blob/main/python/AgX.numpy.py ported to PyTorch

# Reference: agx_compressed_matrix in AgX.numpy.py
AGX_COMPRESSION_MATRIX = torch.tensor(
    [
        [0.84247906, 0.0784336, 0.07922375],
        [0.04232824, 0.87846864, 0.07916613],
        [0.04237565, 0.0784336, 0.87914297],
    ],
    dtype=torch.float32,
)

AgX_MIN_EV = -10.0
AgX_MAX_EV = 6.5
AgX_MIDGREY = 0.18

# ---------------------------------------------
# Simple caches to avoid recomputing per-call
# ---------------------------------------------

# Keyed by: (size, device_str, device_index_or_None, dtype)
_LUT_CACHE: dict = {}

# Keyed by: (device_str, device_index_or_None, dtype)
_MATRIX_CACHE: dict = {}


def _device_key(device: torch.device | None) -> tuple:
    if device is None:
        return ("cpu", None)
    # Normalize cpu device to ("cpu", None) for stable keys
    if device.type == "cpu":
        return ("cpu", None)
    return (device.type, device.index)


def _dtype_key(dtype: torch.dtype) -> torch.dtype:
    return dtype


def _get_cached_agx_matrix(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (*_device_key(device), _dtype_key(dtype))
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    M = AGX_COMPRESSION_MATRIX.to(dtype=dtype, device=device)
    _MATRIX_CACHE[key] = M
    return M


# ---------------------------------------------
# Math / color grading utilities (Torch ports)
# ---------------------------------------------

def _to_tensor_like(x, like: torch.Tensor) -> torch.Tensor:
    """Helper: turn a Python scalar/iterable into a tensor on like's device/dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=like.dtype, device=like.device)
    if isinstance(x, Iterable):
        return torch.tensor(x, dtype=like.dtype, device=like.device)
    return torch.tensor(x, dtype=like.dtype, device=like.device)


def cdlPowerTorch(array: torch.Tensor, power) -> torch.Tensor:
    """
    Port of cdlPower from AgX.numpy.py.
    out = |array| ** power * sign(array)

    power can be a float or a 3-element per-channel value; broadcasting is supported.
    """
    p = _to_tensor_like(power, array)
    # Ensure per-channel power broadcasts across last channel when needed
    if p.ndim == 1 and p.numel() == 3 and (array.ndim == 3 or array.shape[-1] == 3):
        # reshape to (..., 3) broadcast
        for _ in range(array.ndim - 1 - 1):
            p = p.unsqueeze(0)
    return torch.sign(array) * torch.abs(array).pow(p)


def saturateTorch(
    array: torch.Tensor,
    saturation,
    coefs: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722),
) -> torch.Tensor:
    """
    Port of saturate from AgX.numpy.py.
    Increase color saturation around luma computed with BT.709 coefficients by default.

    saturation can be a float or 3-element per-channel value; broadcasting supported.
    """
    s = _to_tensor_like(saturation, array)
    if s.ndim == 1 and s.numel() == 3 and array.shape[-1] == 3:
        for _ in range(array.ndim - 2):
            s = s.unsqueeze(0)
    c = _to_tensor_like(coefs, array)
    if c.ndim == 1:
        for _ in range(array.ndim - 2):
            c = c.unsqueeze(0)
    # luma with keepdim for proper broadcasting back to RGB
    luma = (array * c).sum(dim=-1, keepdim=True)
    out = array - luma
    out = out * s
    out = out + luma
    return out

def saturateTorch_gamut_safe(
    array: torch.Tensor,
    saturation,
    coefs: Tuple[float, float, float] = (0.2126, 0.7152, 0.0722),
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Gamut-safe variant of saturateTorch that preserves the [0,1] range by limiting
    the effective saturation per-pixel so no channel exceeds [0,1].

    - Keeps luma (per coefs) constant.
    - If `saturation` is not a scalar (e.g., 3-channel), falls back to plain saturateTorch.
    """
    s = _to_tensor_like(saturation, array)
    if s.numel() != 1:
        # Non-scalar saturation → defer to standard saturate (may exceed gamut)
        return saturateTorch(array, s, coefs)

    c = _to_tensor_like(coefs, array)
    if c.ndim == 1:
        for _ in range(array.ndim - 2):
            c = c.unsqueeze(0)

    L = (array * c).sum(dim=-1, keepdim=True)
    d = array - L

    eps_t = _to_tensor_like(eps, array)
    # Bounds so that L + s*d stays within [0, 1] per channel
    # For d > 0: s <= (1 - L) / d
    # For d < 0: s <= (0 - L) / d
    pos = d > 0
    neg = d < 0
    big = torch.full_like(d, float('inf'))
    upper = torch.where(pos, (1.0 - L) / (d + eps_t), big)
    lower = torch.where(neg, (0.0 - L) / (d - eps_t), big)
    bounds = torch.minimum(upper, lower)
    s_max = bounds.amin(dim=-1, keepdim=True)
    s_eff = torch.minimum(s.expand_as(s_max), s_max)

    out = L + s_eff * d
    return out

def convertLinearDomainToNormalizedLog2PyTorch(
    tensor: torch.Tensor,
    minimum_ev: float = AgX_MIN_EV,
    maximum_ev: float = AgX_MAX_EV,
    in_midgrey: float = AgX_MIDGREY,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Port of convertOpenDomainToNormalizedLog2 from AgX.numpy.py.
    Similar to OCIO lg2 AllocationTransform.
    """
    tiny = torch.finfo(tensor.dtype).tiny
    x = torch.clamp(tensor, min=tiny)
    x = x / _to_tensor_like(in_midgrey, x)
    output_log = torch.log2(x)
    output_log = torch.clamp(output_log, min=minimum_ev, max=maximum_ev)
    total_exposure = (maximum_ev - minimum_ev)
    out = (output_log - minimum_ev) / total_exposure
    if out_dtype is None:
        out_dtype = tensor.dtype
    return out.to(dtype=out_dtype)


def applyAgXLogTorch(
    tensor: torch.Tensor,
    precision: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Port of applyAgxLog from AgX.numpy.py.
    - Clamp negatives to 0
    - Apply compression matrix
    - Convert to normalized log2
    - Clamp to [0, 1]
    """
    x = torch.clamp(tensor, min=0)
    M = _get_cached_agx_matrix(dtype=precision, device=x.device)
    # matrix/vector multiplication: out_i = sum_j M_ij * x_j
    compressed = x @ M.T
    log = convertLinearDomainToNormalizedLog2PyTorch(compressed, out_dtype=precision)
    return torch.clamp(log, 0.0, 1.0)

def equation_scale_torch(x_pivot: torch.Tensor, y_pivot: torch.Tensor,
                         slope_pivot: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """
    Port of equation_scale from AgX.numpy.py.
    """
    a = (slope_pivot * x_pivot).pow(-power)
    b = (slope_pivot * (x_pivot / y_pivot)).pow(power) - 1.0
    return (a * b).pow(-1.0 / power)


def equation_hyperbolic_torch(x: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """Port of equation_hyperbolic from AgX.numpy.py."""
    return x / (1.0 + x.pow(power)).pow(1.0 / power)


def equation_term_torch(x: torch.Tensor, x_pivot: torch.Tensor,
                        slope_pivot: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Port of equation_term from AgX.numpy.py."""
    return (slope_pivot * (x - x_pivot)) / scale


def equation_curve_torch(
    x: torch.Tensor,
    x_pivot: torch.Tensor,
    y_pivot: torch.Tensor,
    slope_pivot: torch.Tensor,
    power: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Port of equation_curve from AgX.numpy.py."""
    t = equation_term_torch(x, x_pivot, slope_pivot, scale)
    a = equation_hyperbolic_torch(t, power[..., 0]) * scale + y_pivot
    b = equation_hyperbolic_torch(t, power[..., 1]) * scale + y_pivot
    return torch.where(scale < 0.0, a, b)


def equation_full_curve_torch(
    lut_array: torch.Tensor,
    x_pivot: float,
    y_pivot: float,
    slope_pivot: float,
    power: Tuple[float, float],
) -> torch.Tensor:
    """Port of equation_full_curve from AgX.numpy.py."""
    device = lut_array.device
    dtype = lut_array.dtype
    lut_size = lut_array.numel()

    x_p = torch.full((lut_size,), x_pivot, dtype=dtype, device=device)
    y_p = torch.full((lut_size,), y_pivot, dtype=dtype, device=device)
    s_p = torch.full((lut_size,), slope_pivot, dtype=dtype, device=device)
    pow_base = torch.tensor(power, dtype=dtype, device=device)  # shape (2,)
    powv = pow_base.unsqueeze(0).expand(lut_size, 2).contiguous()

    scale_x_pivot = torch.where(lut_array >= x_p, 1.0 - x_p, x_p)
    scale_y_pivot = torch.where(lut_array >= x_p, 1.0 - y_p, y_p)

    toe_scale = equation_scale_torch(scale_x_pivot, scale_y_pivot, s_p, powv[..., 0])
    shoulder_scale = equation_scale_torch(scale_x_pivot, scale_y_pivot, s_p, powv[..., 1])
    scale = torch.where(lut_array >= x_p, shoulder_scale, -toe_scale)
    return equation_curve_torch(lut_array, x_p, y_p, s_p, powv, scale)


def generateAgxLutTorch(size: int = 4096, device=None, dtype=torch.float32) -> torch.Tensor:
    """Port of generateAgxLut from AgX.numpy.py."""
    lut_array = torch.linspace(0.0, 1.0, size, device=device, dtype=dtype)

    AgX_min_EV = -10.0
    AgX_max_EV = +6.5
    AgX_x_pivot = abs(AgX_min_EV) / (AgX_max_EV - AgX_min_EV)
    AgX_y_pivot = 0.50

    general_contrast = 2.0
    limits_contrast = (3.0, 3.25)

    y_LUT = equation_full_curve_torch(
        lut_array,
        AgX_x_pivot,
        AgX_y_pivot,
        general_contrast,
        limits_contrast,
    )
    return y_LUT


def _get_cached_agx_lut(size: int, device: torch.device, dtype: torch.dtype):
    key = (size, *_device_key(device), _dtype_key(dtype))
    entry = _LUT_CACHE.get(key)
    if entry is not None:
        return entry  # (samples, lut, dx, slopes)

    # Build and cache
    lut = generateAgxLutTorch(size=size, device=device, dtype=dtype)
    samples = torch.linspace(0.0, 1.0, lut.numel(), device=device, dtype=dtype)
    N = samples.numel()
    x0 = samples[0]
    x1 = samples[-1]
    dx = (x1 - x0) / (N - 1)
    slope_lo = (lut[1] - lut[0]) / dx
    slope_hi = (lut[-1] - lut[-2]) / dx
    _LUT_CACHE[key] = (samples, lut, dx, slope_lo, slope_hi)
    return _LUT_CACHE[key]


def _apply_cached_lut(x: torch.Tensor, samples: torch.Tensor, lut: torch.Tensor,
                      dx: torch.Tensor, slope_lo: torch.Tensor, slope_hi: torch.Tensor) -> torch.Tensor:
    """
    Faster path that reuses cached dx and end slopes. Assumes samples is [0,1] linspace.
    """
    N = samples.numel()
    x0 = samples[0]
    x1 = samples[-1]

    idx = (x - x0) / dx
    i0 = torch.floor(idx).to(torch.long).clamp(0, N - 2)
    i1 = i0 + 1
    t = (idx - i0.to(idx.dtype)).clamp(0.0, 1.0)
    y_in = torch.lerp(lut[i0], lut[i1], t)

    left = x < x0
    right = x > x1
    y_left = lut[0] + (x - x0) * slope_lo
    y_right = lut[-1] + (x - x1) * slope_hi
    # Compose results without in-place writes to keep grad graph intact
    y = torch.where(left, y_left, torch.where(right, y_right, y_in))
    return y


def applyAgxLutTorch(
    tensor: torch.Tensor,
    precision: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Port of applyAgxLut from AgX.numpy.py.
    Convert log data to AgX Base via 1D LUT with linear interpolation and linear extrapolation.
    """
    device = tensor.device
    dtype = precision
    samples, lut, dx, slope_lo, slope_hi = _get_cached_agx_lut(4096, device, dtype)

    # Apply channel-wise uniformly; works for any shape by flattening and restoring
    x = tensor.to(dtype)
    y = _apply_cached_lut(x.to(torch.float32), samples.to(torch.float32), lut.to(torch.float32), dx.to(torch.float32), slope_lo.to(torch.float32), slope_hi.to(torch.float32))
    return y.to(tensor.dtype)

def applyLookPunchyTorch(
    array: torch.Tensor,
    punchy_gamma: float = 1.3,
    punchy_saturation: float = 1.2,
    preserve_range: bool = True,
) -> torch.Tensor:
    """
    Port of applyLookPunchy from AgX.numpy.py.
    Implements CDL gamma (power) and saturation boost.
    """
    original_dtype = array.dtype
    
    # 1. Cast input to float32 for stable gradient computation
    x = array.to(torch.float32)

    # 2. Perform the power and saturation operations in float32
    out = cdlPowerTorch(x, punchy_gamma)
    if preserve_range:
        out = saturateTorch_gamut_safe(out, saturation=punchy_saturation)
    else:
        out = saturateTorch(out, saturation=punchy_saturation)
        
    # 3. Cast the result back to the original dtype
    return out.to(original_dtype)

def applyAgXTorch(array: torch.Tensor, precision: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Port of applyAgX from AgX.numpy.py.
    Pipeline:
      customLook1 -> applyAgXLogTorch -> applyAgxLutTorch -> applyLookPunchyTorch
    Input: float tensor with shape (..., 3) in linear sRGB
    Output: display-ready tensor encoded for sRGB SDR monitors
    """
    x = applyAgXLogTorch(array, precision=precision)
    x = applyAgxLutTorch(x, precision=precision)
    x = applyLookPunchyTorch(x)
    return x
