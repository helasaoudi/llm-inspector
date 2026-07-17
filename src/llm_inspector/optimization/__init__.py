"""Optimization Analysis — projected strategies on top of measured reports."""

# Keep this module light to avoid circular imports with models.report.
# Import OptimizationAnalyzer from llm_inspector.optimization.analyzer directly.

from llm_inspector.optimization.models import OptimizationAnalysis

__all__ = ["OptimizationAnalysis"]
