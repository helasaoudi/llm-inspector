"""
Independent data collectors — the workhorses of LLM Inspector.

Rules (enforced by architecture):
  - A collector owns exactly one observable domain.
  - A collector never calls another collector.
  - A collector never mutates shared state.
  - A collector always returns a CollectorResult — it never raises.
"""
