from pydantic import BaseModel, Field
from typing import Literal, Dict, Any
import stochastic.translation_rule as TranslationRule

class DistributionConfig(BaseModel):
    """
    Defines a probability distribution for a behavioral parameter.
    Designed to be universal by accepting a dictionary of parameters.
    """
    dist_type: str = Field(..., description="The type of distribution (e.g., 'normal', 'uniform', 'constant').")
    params: Dict[str, Any] = Field(default_factory=dict, description="Key-value pairs of parameters (e.g., {'mean': 0, 'std': 1}).")

    def get_sampler(self):
        """Returns a sampler object compatible with the stochastic library."""
        # Map common config types to the specific RuleSet method names provided
        method_map = {
            "normal": "normal_distribution_rule",
            "uniform": "uniform_distribution_rule",
            "constant": "constant_rule",
            "weighted": "weighted_value_distribution_rule"
        }
        
        # Determine the method name to look up in RuleSet
        method_name = method_map.get(self.dist_type, f"{self.dist_type}_distribution_rule")

        try:
            rule_factory = getattr(TranslationRule.RuleSet, method_name)
            dist_constructor = rule_factory() # Creates the distribution class/closure

            if not hasattr(TranslationRule.RuleSet, method_name):
                raise ValueError(f"RuleSet has no method: {method_name}")
            
            # Prepare arguments based on the specific requirements of the provided RuleSet
            args = []
            kwargs = {}

            if self.dist_type in ["normal", "uniform"]:
                if "int" in self.params:
                    kwargs["int"] = self.params["int"]

            if self.dist_type == "normal":
                # normal_distribution_rule requires: mu (values[0]), sigma (values[1])
                args = [self.params.get("mean", 0.0), self.params.get("std", 1.0)]

            elif self.dist_type == "uniform":
                # uniform_distribution_rule requires: lower (values[0]), upper (values[1])
                args = [self.params.get("min", 0.0), self.params.get("max", 1.0)]

            elif self.dist_type == "constant":
                # definite_value_rule requires: value (values[0])
                val = self.params.get("value")
                if val is None:
                    val = next(iter(self.params.values()), 0.0)
                args = [val]

            elif self.dist_type == "weighted":
                 # weighted_value_distribution_rule requires: values_with_weights dict (values[0])
                 # Since params keys are strings in JSON/Pydantic, we try to convert them to floats for weights
                 converted_params = {}
                 for k, v in self.params.items():
                     try:
                         # Attempt to convert string key to float weight
                         weight = float(k)
                         # Ensure value is a list as expected by WeightedValueDistribution
                         val_list = v if isinstance(v, list) else [v]
                         converted_params[weight] = val_list
                     except ValueError:
                         continue
                 args = [converted_params]

            else:
                # Fallback: pass values in order
                args = list(self.params.values())

            return dist_constructor(*args, **kwargs)

        except Exception as e:
            raise ValueError(f"Failed to create sampler for {self.dist_type}: {str(e)}")
