from typing import Protocol, Any
from stochastic.distribution import Distribution
import stochastic.distribution as Distribution

class FieldRule(Protocol):
    def __call__(self, *values: Any, context: dict[str, Any] | None = None) -> Distribution.Distribution: ...

class RuleSet:
    def int_normal_distribution_rule() -> FieldRule:
        def rule(
            *values: Any,
            context: dict[str, Any] | None = None
        ) -> Distribution.Distribution:
            if len(values) < 2:
                raise ValueError("normal_distribution_int_rule requires one value for mu and one for sigma")
            mu = float(values[0])  
            sigma = float(values[1])
            lb = float('-inf')
            ub = float('inf')

            if context:
                lb = context.get("lower", lb)
                ub = context.get("upper", ub)

            return Distribution.IntNormalDistribution(
                mean=mu,
                stddev=sigma,
                lower=lb,
                upper=ub
            )
        return rule
    
    def definite_value_rule() -> FieldRule:
        def rule(
            *values: Any,
            context: dict[str, Any] | None = None
        ) -> Distribution.Distribution:
            if len(values) == 0:
                raise ValueError("definite_value_rule requires at least one value for the value")
            return Distribution.DefiniteValue(float(values[0]))
        return rule