"""
Safe HTTP client for querying runtime REST APIs.

All functions:
  - Have a short timeout (default 2 s) so they never block inspection.
  - Return None on any error (connection refused, timeout, bad JSON, etc.).
  - Never raise — callers receive None and mark fields as Unavailable.

Works identically on macOS and Linux.
"""

from __future__ import annotations

import logging

_LOG = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 2.0  # seconds


def get_json(url: str, timeout: float = _DEFAULT_TIMEOUT) -> dict | list | None:
    """
    Perform a GET request and return the parsed JSON body, or None.

    Args:
        url:     Full URL to request.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON (dict or list), or None if the request fails for any reason.
    """
    try:
        import requests  # noqa: PLC0415

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("HTTP GET %s failed: %s", url, exc)
        return None


def get_text(url: str, timeout: float = _DEFAULT_TIMEOUT) -> str | None:
    """
    Perform a GET request and return the raw text body, or None.

    Used for Prometheus-format /metrics endpoints that return plain text.
    """
    try:
        import requests  # noqa: PLC0415

        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.text
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("HTTP GET %s failed: %s", url, exc)
        return None


def parse_prometheus(text: str) -> dict[str, float]:
    """
    Parse a Prometheus plain-text metrics response into a flat dict.

    Returns a mapping of metric_name{labels} → float value.
    Only parses simple scalar metrics — histograms are skipped.

    Example input line::
        vllm:gpu_cache_usage_perc{model_name="Llama-3-8B"} 0.423

    Example output key::
        'vllm:gpu_cache_usage_perc{model_name="Llama-3-8B"}'
    """
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        key, raw_val = parts
        try:
            result[key] = float(raw_val)
        except ValueError:
            continue
    return result


def find_metric(
    metrics: dict[str, float],
    prefix: str,
    label_filter: dict[str, str] | None = None,
) -> float | None:
    """
    Search a parsed Prometheus dict for a metric matching *prefix*.

    Args:
        metrics:      Output of parse_prometheus().
        prefix:       Metric name prefix to match (e.g. ``"vllm:gpu_cache_usage_perc"``).
        label_filter: Optional dict of label key→value that must be present.

    Returns:
        The float value of the first matching metric, or None.
    """
    for key, value in metrics.items():
        if not key.startswith(prefix):
            continue
        if label_filter:
            if not all(f'{k}="{v}"' in key for k, v in label_filter.items()):
                continue
        return value
    return None
