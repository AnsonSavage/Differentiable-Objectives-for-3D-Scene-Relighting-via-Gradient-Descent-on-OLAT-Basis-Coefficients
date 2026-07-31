"""
Composite loss for combining multiple loss functions with weights.
"""
from dataclasses import dataclass, field
from typing import Callable, Any, cast

from utils.losses.base import BaseLoss, UpdatableLoss


@dataclass
class Component:
    """Helper dataclass representing one weighted loss component.

    Attributes:
        weight_fn: Callable[[int, int], float] - function returning weight for a step
        current_weight: float - cached current weight value
        loss_fn: BaseLoss - the child loss function
    """
    weight_fn: Callable[..., Any]
    current_weight: float
    loss_fn: BaseLoss = field(repr=False)


class CompositeLoss(UpdatableLoss):
    """Weighted combination of multiple loss functions.
    
    Input is a list of tuples where each tuple is (weight, loss_function).
    Weight can be a float or a callable taking (current_step, total_steps) and returning a float.
    """

    def __init__(self, weight_and_loss_functions: list[Any]):
        """Initialize composite loss.
        
        Args:
            weight_and_loss_functions: list of (weight, loss_function) tuples
        """
        super().__init__()

        self.components: list[Component] = []
        for weight, loss_fn in weight_and_loss_functions:
            if callable(weight):
                weight_fn = weight
                val = cast(float, weight(0, 1))
                current = float(val)
            else:
                # capture the constant weight in a named function to satisfy linters
                weight_val = float(weight)
                def constant_weight(s, t, w=weight_val):
                    return w
                weight_fn = constant_weight
                current = weight_val

            self.components.append(Component(weight_fn=weight_fn, current_weight=current, loss_fn=loss_fn))

    def forward(self, image):
        """Calculate weighted sum of losses.

        Args:
            image: Image tensor to evaluate

        Returns:
            Weighted sum of individual losses
        """
        total_loss = 0.0
        for comp in self.components:
            loss = comp.loss_fn(image)
            total_loss += comp.current_weight * loss
        return total_loss

    def get_prompt_info(self):
        """Get prompt information from all component losses.

        Returns:
            Dictionary with composite_losses key containing a list of
            dictionaries, each with weight and loss prompt info
        """
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

    def update_parameters(self, current_step: int, total_steps: int, **kwargs):
        """Update weights and child losses.

        Args:
            current_step: Current optimization step
            total_steps: Total number of optimization steps
            kwargs: Additional arguments passed to child losses
        """
        for comp in self.components:
            # Update weight using the weight function
            comp.current_weight = float(comp.weight_fn(current_step, total_steps))

            # Update child loss if it is updatable
            if isinstance(comp.loss_fn, UpdatableLoss):
                comp.loss_fn.update_parameters(current_step, total_steps, **kwargs)