"""
Hardware backend abstraction layer.

The Inspector never imports pynvml, sysfs, or IOKit directly.
All hardware calls go through a HardwareBackend implementation.

This allows:
  - Development on macOS (CPUBackend or MetalBackend)
  - Production on Linux CUDA machines (CUDABackend)
  - Future AMD support (ROCmBackend) with zero core changes
"""
