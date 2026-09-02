"""Problem definitions shared by benchmark-suite adapters."""

from rlaopt_experiments.problems.synthetic_erm import (
    ElasticNetProblem,
    ElasticNetSpec,
    MultinomialProblem,
    MultinomialSpec,
    generate_elastic_net_problem,
    generate_multinomial_problem,
)

__all__ = [
    "ElasticNetProblem",
    "ElasticNetSpec",
    "MultinomialProblem",
    "MultinomialSpec",
    "generate_elastic_net_problem",
    "generate_multinomial_problem",
]
