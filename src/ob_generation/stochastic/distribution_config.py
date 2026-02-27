from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

from ob_generation.stochastic.distribution import Constant, NormalDistribution, UniformDistribution, CategoricalDistribution


DistType = Literal["normal", "uniform", "constant", "categorical"]


class DistributionConfig(BaseModel):
    """
    JSON-friendly config that builds your Distribution objects directly.

    Supported:
    - normal:  params = { "mean": ..., "std": ..., "int": true/false?, "lower": ..., "upper": ... }
    - uniform: params = { "min": ..., "max": ..., "int": true/false? }
             + context can override bounds via {"lower": ..., "upper": ...}
    - constant: params = { "value": ... }
    - categorical: params = { "weights" : [value1, value2, ...] }
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

        if self.dist_type == "categorical":
            weights = p.get("weights", [])
            if not weights:
                raise ValueError("categorical distribution requires non-empty 'weights' list")
            return CategoricalDistribution(weights)

        # should be unreachable due to Literal typing
        raise ValueError(f"Unsupported dist_type: {self.dist_type}")    
           