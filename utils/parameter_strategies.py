"""Parameter strategies for handling RGB vs HSV parameter spaces during optimization.

This module implements the Strategy pattern to eliminate conditional logic
for different parameter representations (RGB vs HSV).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from utils.color.hsv_utils import hsv_to_rgb


class ParameterStrategy(ABC):
    """Abstract base class for parameter optimization strategies."""

    @abstractmethod
    def initialize_parameters(
        self,
        n_results: int,
        num_lights: int,
        mean: float | tuple[float, float, float],
        std: float | tuple[float, float, float],
        torch_precision: torch.dtype,
        device: str,
        learning_rate: float,
        learning_rate_scheduler_creator_callback: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        """Initialize parameters and optimizers.

        Args:
            n_results: Number of optimization results.
            num_lights: Number of light sources.
            mean: Mean for parameter initialization.
            std: Standard deviation for parameter initialization.
            torch_precision: PyTorch data type.
            device: Device to place tensors on.
            learning_rate: Learning rate for optimizer.
            learning_rate_scheduler_creator_callback: Optional callback to create an LR scheduler.
        """

    @abstractmethod
    def zero_grad(self) -> None:
        """Zero gradients of all optimizers."""

    @abstractmethod
    def update_parameter_constraints(self, epoch: int) -> None:
        """Update which parameters are trainable based on current epoch.

        Args:
            epoch: Current training iteration.
        """

    @abstractmethod
    def get_multipliers(self) -> torch.Tensor:
        """Get current multipliers tensor in RGB space.

        Returns:
            Tensor of shape [n_results, num_lights, 3] in RGB space.
        """

    @abstractmethod
    def step(self) -> None:
        """Perform optimizer step."""

    @abstractmethod
    def apply_physical_constraints(self) -> None:
        """Apply physical plausibility constraints to parameters."""

    def get_parameters_for_saving(self) -> torch.Tensor:
        """Get parameters in their native representation for saving.

        Returns:
            Parameters tensor (RGB or HSV depending on strategy).
        """
        return self.get_multipliers()


class RGBParameterStrategy(ParameterStrategy):
    """Strategy for optimizing directly in RGB color space."""

    def __init__(self, verbose: bool = False):
        """Initialize RGBParameterStrategy.

        Args:
            verbose: If True, prints learning rate updates.
        """
        self.multipliers: torch.nn.Parameter | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.verbose = verbose

    def initialize_parameters(
        self,
        n_results: int,
        num_lights: int,
        mean: float | tuple[float, float, float],
        std: float | tuple[float, float, float],
        torch_precision: torch.dtype,
        device: str,
        learning_rate: float,
        learning_rate_scheduler_creator_callback: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        """Initialize RGB multipliers and optimizer.

        Args:
            n_results: Number of parallel results.
            num_lights: Number of light sources.
            mean: Mean float or 3-tuple for initialization.
            std: Std float or 3-tuple for initialization.
            torch_precision: PyTorch data type precision.
            device: Target device.
            learning_rate: Optimizer learning rate.
            learning_rate_scheduler_creator_callback: Optional LR scheduler factory.

        Raises:
            ValueError: If mean or std tuples are not of length 3.
        """
        if isinstance(mean, tuple):
            if len(mean) != 3:
                raise ValueError("RGB mean must be a single float or a tuple of three floats.")
            mean_tensor = torch.tensor(mean, dtype=torch_precision, device=device).view(1, 1, 3)
        else:
            mean_tensor = torch.full((1, 1, 3), float(mean), dtype=torch_precision, device=device)

        if isinstance(std, tuple):
            if len(std) != 3:
                raise ValueError("RGB std must be a single float or a tuple of three floats.")
            std_tensor = torch.tensor(std, dtype=torch_precision, device=device).view(1, 1, 3)
        else:
            std_tensor = torch.full((1, 1, 3), float(std), dtype=torch_precision, device=device)

        mean_expanded = mean_tensor.expand(n_results, num_lights, 3)
        std_expanded = std_tensor.expand(n_results, num_lights, 3)

        self.multipliers = torch.nn.Parameter(
            torch.normal(mean_expanded, std_expanded),
            requires_grad=True,
        )
        self.optimizer = torch.optim.Adam([self.multipliers], lr=learning_rate)
        if callable(learning_rate_scheduler_creator_callback):
            self.scheduler = learning_rate_scheduler_creator_callback(self.optimizer)

    def zero_grad(self) -> None:
        """Zero gradients for RGB optimizer."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized. Call initialize_parameters first.")
        self.optimizer.zero_grad()

    def update_parameter_constraints(self, epoch: int) -> None:
        """Update parameter constraints (no-op for RGB as all channels are trainable).

        Args:
            epoch: Current training iteration.
        """

    def get_multipliers(self) -> torch.Tensor:
        """Return RGB multipliers directly.

        Returns:
            RGB multipliers tensor of shape [n_results, num_lights, 3].
        """
        if self.multipliers is None:
            raise RuntimeError("Multipliers have not been initialized. Call initialize_parameters first.")
        return self.multipliers

    def step(self) -> None:
        """Perform optimizer step and advance scheduler if configured."""
        if self.optimizer is None:
            raise RuntimeError("Optimizer has not been initialized. Call initialize_parameters first.")
        self.optimizer.step()
        if self.scheduler is not None:
            if self.verbose:
                print(f"Learning rate: {self.scheduler.get_last_lr()}")
            self.scheduler.step()

    def apply_physical_constraints(self) -> None:
        """Clamp RGB multipliers to be non-negative."""
        if self.multipliers is None:
            raise RuntimeError("Multipliers have not been initialized. Call initialize_parameters first.")
        with torch.no_grad():
            self.multipliers.clamp_(0)


class HSVParameterStrategy(ParameterStrategy):
    """Strategy for optimizing in HSV color space."""

    def __init__(self, hsv_callback: Callable[[int], str] | None = None, verbose: bool = False):
        """Initialize HSV strategy.

        Args:
            hsv_callback: Optional callback returning trainable channels ('h', 's', 'v') per epoch.
            verbose: Whether to print verbose learning rate info.
        """
        self.hsv_callback: Callable[[int], str] | None = hsv_callback
        self.h: torch.nn.Parameter | None = None
        self.s: torch.nn.Parameter | None = None
        self.v: torch.nn.Parameter | None = None
        self.h_optimizer: torch.optim.Optimizer | None = None
        self.s_optimizer: torch.optim.Optimizer | None = None
        self.v_optimizer: torch.optim.Optimizer | None = None
        self.h_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.s_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.v_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.hsv_params: dict[str, torch.nn.Parameter] | None = None
        self.hsv_optimizers: dict[str, torch.optim.Optimizer] | None = None
        self.verbose = verbose

    def initialize_parameters(
        self,
        n_results: int,
        num_lights: int,
        mean: float | tuple[float, float, float],
        std: float | tuple[float, float, float],
        torch_precision: torch.dtype,
        device: str,
        learning_rate: float,
        learning_rate_scheduler_creator_callback: Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler] | None = None,
    ) -> None:
        """Initialize HSV parameters (H, S, V) and respective Adam optimizers.

        Args:
            n_results: Number of parallel results.
            num_lights: Number of light sources.
            mean: Float or 3-tuple (H, S, V) for initialization mean.
            std: 3-tuple of floats (H, S, V) for initialization standard deviations.
            torch_precision: PyTorch data type precision.
            device: Computation device.
            learning_rate: Optimizer learning rate.
            learning_rate_scheduler_creator_callback: Optional LR scheduler factory callback.

        Raises:
            ValueError: If mean or std formats are invalid for HSV.
        """
        if isinstance(mean, tuple):
            if len(mean) != 3:
                raise ValueError(
                    "When using HSV parameters, mean must be a single float "
                    "or a tuple of three floats (for H, S, and V)."
                )
            h_mean, s_mean, v_mean = mean
        else:
            h_mean = s_mean = v_mean = float(mean)

        if not isinstance(std, tuple) or len(std) != 3:
            raise ValueError(
                "When using HSV parameters, you must provide std as a tuple "
                "of three floats (for H, S, and V)."
            )

        h_stdev, s_stdev, v_stdev = std

        self.h = torch.nn.Parameter(
            torch.normal(h_mean, h_stdev, size=(n_results, num_lights, 1), dtype=torch_precision, device=device),
        )
        self.s = torch.nn.Parameter(
            torch.normal(s_mean, s_stdev, size=(n_results, num_lights, 1), dtype=torch_precision, device=device),
        )
        self.v = torch.nn.Parameter(
            torch.normal(v_mean, v_stdev, size=(n_results, num_lights, 1), dtype=torch_precision, device=device),
        )

        self.h_optimizer = torch.optim.Adam([self.h], lr=learning_rate)
        self.s_optimizer = torch.optim.Adam([self.s], lr=learning_rate)
        self.v_optimizer = torch.optim.Adam([self.v], lr=learning_rate)

        self.h_scheduler = None
        self.s_scheduler = None
        self.v_scheduler = None
        if callable(learning_rate_scheduler_creator_callback):
            try:
                self.h_scheduler = learning_rate_scheduler_creator_callback(self.h_optimizer)
            except Exception:
                self.h_scheduler = None
            try:
                self.s_scheduler = learning_rate_scheduler_creator_callback(self.s_optimizer)
            except Exception:
                self.s_scheduler = None
            try:
                self.v_scheduler = learning_rate_scheduler_creator_callback(self.v_optimizer)
            except Exception:
                self.v_scheduler = None

        self.hsv_params = {"h": self.h, "s": self.s, "v": self.v}
        self.hsv_optimizers = {
            "h": self.h_optimizer,
            "s": self.s_optimizer,
            "v": self.v_optimizer,
        }

    def zero_grad(self) -> None:
        """Zero gradients for all HSV optimizers."""
        if self.hsv_optimizers is None:
            raise RuntimeError("HSV optimizers have not been initialized.")
        for optimizer in self.hsv_optimizers.values():
            optimizer.zero_grad()

    def update_parameter_constraints(self, epoch: int) -> None:
        """Update which HSV channels require gradients based on callback.

        Args:
            epoch: Current training iteration.
        """
        if self.hsv_params is None:
            raise RuntimeError("HSV parameters have not been initialized.")
        legal_params = self.hsv_callback(epoch).lower() if self.hsv_callback else "hsv"
        for param_name, param in self.hsv_params.items():
            param.requires_grad = param_name in legal_params

    def get_multipliers(self) -> torch.Tensor:
        """Convert current HSV parameters to RGB multipliers.

        Returns:
            RGB multipliers tensor of shape [n_results, num_lights, 3].
        """
        if self.h is None or self.s is None or self.v is None:
            raise RuntimeError("HSV parameters have not been initialized.")
        hsv_tensor = torch.cat([self.h, self.s, self.v], dim=-1)
        hsv_reshaped = hsv_tensor.view(hsv_tensor.shape[0], hsv_tensor.shape[1], 1, 1, 3)
        rgb_reshaped = hsv_to_rgb(hsv_reshaped.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        return rgb_reshaped.squeeze(-2).squeeze(-2)

    def step(self) -> None:
        """Step only the optimizers for currently trainable HSV parameters."""
        if self.hsv_optimizers is None or self.hsv_params is None:
            raise RuntimeError("HSV parameters/optimizers have not been initialized.")
        for param_name, optimizer in self.hsv_optimizers.items():
            if self.hsv_params[param_name].requires_grad:
                optimizer.step()
                if param_name == "h" and self.h_scheduler is not None:
                    if self.verbose:
                        print(f"Learning rate for H: {self.h_scheduler.get_last_lr()}")
                    self.h_scheduler.step()
                elif param_name == "s" and self.s_scheduler is not None:
                    if self.verbose:
                        print(f"Learning rate for S: {self.s_scheduler.get_last_lr()}")
                    self.s_scheduler.step()
                elif param_name == "v" and self.v_scheduler is not None:
                    if self.verbose:
                        print(f"Learning rate for V: {self.v_scheduler.get_last_lr()}")
                    self.v_scheduler.step()

    def apply_physical_constraints(self) -> None:
        """Apply HSV constraints (wrap hue modulo 1, clamp saturation [0, 1] and value >= 0)."""
        if self.h is None or self.s is None or self.v is None:
            raise RuntimeError("HSV parameters have not been initialized.")
        with torch.no_grad():
            self.h.remainder_(1.0)
            self.s.clamp_(0, 1)
            self.v.clamp_(0)

    def get_parameters_for_saving(self) -> torch.Tensor:
        """Return native HSV parameters tensor.

        Returns:
            HSV tensor of shape [n_results, num_lights, 3].
        """
        if self.h is None or self.s is None or self.v is None:
            raise RuntimeError("HSV parameters have not been initialized.")
        return torch.cat([self.h, self.s, self.v], dim=-1)

