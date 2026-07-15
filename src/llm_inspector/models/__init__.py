"""
Frozen Pydantic v2 data models — the shared data contract for the entire project.

Import rules (enforced):
  - models/ NEVER imports from collectors/, backends/, plugins/, or inspector/.
  - All other packages may freely import from models/.
"""
