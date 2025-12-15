from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

from stochastic.distribution import Constant, NormalDistribution, UniformDistribution, WeightedValueDistribution


DistType = Literal["normal", "uniform", "constant", "weighted"]


class DistributionConfig(BaseModel):
    """
    JSON-friendly config that builds your Distribution objects directly.

    Supported:
    - normal:  params = { "mean": ..., "std": ..., "int": true/false?, "lower": ..., "upper": ... }
    - uniform: params = { "min": ..., "max": ..., "int": true/false? }
             + context can override bounds via {"lower": ..., "upper": ...}
    - constant: params = { "value": ... }
    - weighted: params = { "0.7": [8,9], "0.3": [10] }  (keys are weights)
    """
    dist_type: DistType
    params: Dict[str, Any] = Field(default_factory=dict)

    def build(self, context: Optional[dict[str, Any]] = None):
        context = context or {}
        p = self.params

        if self.dist_type == "constant":
            if "value" not in p:
                raise ValueError("constant requires params={'value': ...}")
            return Constant(float(p["value"]))

        if self.dist_type == "uniform":
            # base bounds come from params; context can override
            lo = float(p.get("min", 0.0))
            hi = float(p.get("max", 1.0))
            lo = float(context.get("lower", lo))
            hi = float(context.get("upper", hi))
            as_int = bool(p.get("int", True))
            return UniformDistribution(lower=lo, upper=hi, int=as_int)

        if self.dist_type == "normal":
            mu = float(p.get("mean", 0.0))
            sigma = float(p.get("std", 1.0))

            # bounds can come from params or context; context wins
            lb = float(p.get("lower", float("-inf")))
            ub = float(p.get("upper", float("inf")))
            lb = float(context.get("lower", lb))
            ub = float(context.get("upper", ub))

            as_int = bool(p.get("int", True))
            return NormalDistribution(mean=mu, stddev=sigma, lower=lb, upper=ub, int=as_int)

        if self.dist_type == "weighted":
            # params = {"0.7": [8, 9], "0.3": [10]}  OR  {"0.7": 8, "0.3": 9}
            values_with_weights: dict[float, list[float]] = {}

            if not p:
                raise ValueError("weighted distribution requires non-empty params")

            for k, vals in p.items():
                try:
                    weight = float(k)
                except ValueError as e:
                    raise ValueError(
                        "weighted distribution expects numeric-string keys as weights "
                        "(e.g., {'0.7': [8,9], '0.3': [10]})"
                    ) from e

                if not isinstance(vals, list):
                    vals = [vals]

                values_with_weights[weight] = [float(v) for v in vals]

            return WeightedValueDistribution(values_with_weights)

        # should be unreachable due to Literal typing
        raise ValueError(f"Unsupported dist_type: {self.dist_type}")
