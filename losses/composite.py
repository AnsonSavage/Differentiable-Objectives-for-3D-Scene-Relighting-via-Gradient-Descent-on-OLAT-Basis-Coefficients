"""Composite loss for combining multiple loss functions with weights."""
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast, override

from losses.base import BaseLoss, UpdatableLoss


@dataclass
class Component:
    """Helper dataclass representing one weighted loss component.

    Attributes:
        weight_fn: Function returning weight for a step given (step, total_steps).
        current_weight: Cached current weight value.
        loss_fn: Child loss function.
    """
    weight_fn: Callable[..., Any]
    current_weight: float
    loss_fn: BaseLoss = field(repr=False)


class CompositeLoss(UpdatableLoss):
    """Weighted combination of multiple loss functions.

    Supports constant float weights or dynamic weight functions taking
    (current_step, total_steps) and returning a float.
    """

    def __init__(self, weight_and_loss_functions: list[Any]):
        """Initialize composite loss.

        Args:
            weight_and_loss_functions: List of (weight, loss_function) tuples.
        """
        super().__init__()

        self.components: list[Component] = []
        for weight, loss_fn in weight_and_loss_functions:
            if callable(weight):
                weight_fn = weight
                val = cast(float, weight(0, 1))
                current = float(val)
            else:
                weight_val = float(weight)

                def constant_weight(s, t, w=weight_val):
                    return w

                weight_fn = constant_weight
                current = weight_val

            self.components.append(Component(weight_fn=weight_fn, current_weight=current, loss_fn=loss_fn))

    @override
    def forward(self, image):
        total_loss = 0.0
        for comp in self.components:
            loss = comp.loss_fn(image)
            total_loss += comp.current_weight * loss
        return total_loss

    @override
    def get_prompt_info(self) -> dict[str, Any]:
        component_infos = []
        for comp in self.components:
            loss_fn = comp.loss_fn
            weight = comp.current_weight

            component_info = {
                "weight": float(weight),
                "loss_type": type(loss_fn).__name__,
            }
            info = loss_fn.get_prompt_info()
            if isinstance(info, dict):
                component_info.update(info)
            component_infos.append(component_info)

        return {"composite_losses": component_infos}

    @override
    def update_parameters(self, current_step: int, total_steps: int, **kwargs):
        for comp in self.components:
            comp.current_weight = float(comp.weight_fn(current_step, total_steps))

            if isinstance(comp.loss_fn, UpdatableLoss):
                comp.loss_fn.update_parameters(current_step, total_steps, **kwargs)