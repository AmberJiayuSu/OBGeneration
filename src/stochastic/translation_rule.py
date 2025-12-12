from typing import Protocol, Any
from stochastic.distribution import Distribution
import stochastic.distribution as Distribution

class FieldRule(Protocol):
    def __call__(self, *values: Any, context: dict[str, Any] | None = None) -> Distribution.Distribution: ...

class RuleSet:

    def uniform_distribution_rule() -> FieldRule:
        def rule(
            *values: Any,
            context: dict[str, Any] | None = None,
            int: bool = True
        ) -> Distribution.Distribution:
            if len(values) < 2:
                raise ValueError("uniform_distribution_rule requires one value for lower and one for upper")
            lower = float(values[0])  
            upper = float(values[1])

            if context:
                lower = context.get("lower", lower)
                upper = context.get("upper", upper)

            return Distribution.UniformDistribution(
                lower=lower,
                upper=upper,
                int=int
            )
        return rule

    def normal_distribution_rule() -> FieldRule:
        def rule(
            *values: Any,
            context: dict[str, Any] | None = None,
            int: bool = True
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

            return Distribution.NormalDistribution(
                mean=mu,
                stddev=sigma,
                lower=lb,
                upper=ub,
                int=int
            )
        return rule
    
    def constant_rule() -> FieldRule:
        def rule(
            *values: Any,
            context: dict[str, Any] | None = None
        ) -> Distribution.Distribution:
            if len(values) == 0:
                raise ValueError("definite_value_rule requires at least one value for the value")
            return Distribution.DefiniteValue(float(values[0]))
        return rule
    
    def weighted_value_distribution_rule() -> FieldRule:
        def rule(
            *values: Any,
            context: dict[str, Any] | None = None
        ) -> Distribution.Distribution:
            if len(values) == 0 or len(values) % 1 != 0:
                raise ValueError("assigned_value_distribution_rule requires dict of weight-value pairs")
            values_with_weights = values[0]
            return Distribution.WeightedValueDistribution(values_with_weights)
        return rule
    
    