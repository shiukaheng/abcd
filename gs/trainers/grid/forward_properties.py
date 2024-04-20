from typing import List

# These are the properties that will be used for training the model
GAUSSIAN_MODEL_PROPERTIES = [
    "positions",
    "sh_coefficients",
    "sh_coefficients_0",
    "sh_coefficients_rest",
    "rotations",
    "scales",
    "opacities",
    "sh_degree",
    "background_color",
    "radii",
    "mean_gradient_magnitude",
    "backprop_stats",
    "scales_range",
    "scales_activation",
    "scales_inverse_activation",
    "opacities_activation",
    "opacities_inverse_activation",
    "viewspace_points",
    "_gradient_accumulator",
    "_gradient_accumulator_denominator",
    "max_radii2D",
    "sh_mlp",
    "use_camera_aware_appearance",
]


def forward_to_active_cell(properties: List[str]=GAUSSIAN_MODEL_PROPERTIES):
    """
    Decorator to forward the properties to the active cell's model.
    """
    def decorator(cls):
        # Define getters and setters that forward to GridGaussianModel.active_cell.model for the specified properties
        for prop in properties:
            def getter(self, prop=prop):
                return getattr(self.grid_active_cell.model, prop)

            def setter(self, value, prop=prop):
                setattr(self.grid_active_cell.model, prop, value)

            setattr(cls, prop, property(getter, setter))

        return cls
    
    return decorator