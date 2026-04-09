#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def fetch_once(url: str, method: str, headers: dict[str, str], body: bytes | None, timeout: float) -> dict[str, object]:
    started_at = time.perf_counter()
    request = Request(url, data=body, method=method.upper())
    for key, value in headers.items():
        request.add_header(key, value)
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read()
            latency_ms = (time.perf_counter() - started_at) * 1000
            return {"ok": True, "status": response.status, "latency_ms": latency_ms}
    except HTTPError as exc:
        latency_ms = (time.perf_counter() - started_at) * 1000
        return {"ok": False, "status": exc.code, "latency_ms": latency_ms, "error": str(exc)}
    except URLError as exc:
        latency_ms = (time.perf_counter() - started_at) * 1000
        return {"ok": False, "status": None, "latency_ms": latency_ms, "error": str(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded HTTP load against a single URL.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--method", default="GET")
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--json-body")
    parser.add_argument("--report-path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers: dict[str, str] = {}
    for item in args.header:
        if "=" not in item:
            raise SystemExit(f"invalid header: {item}")
        key, value = item.split("=", 1)
        headers[key] = value
    body = None
    if args.json_body is not None:
        headers.setdefault("Content-Type", "application/json")
        body = args.json_body.encode("utf-8")

    results: list[dict[str, object]] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(fetch_once, args.url, args.method, headers, body, args.timeout)
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            result = future.result()
            with lock:
                results.append(result)

    latencies = [float(item["latency_ms"]) for item in results]
    ok_count = sum(1 for item in results if item["ok"])
    report = {
        "url": args.url,
        "method": args.method.upper(),
        "requests": args.requests,
        "concurrency": args.concurrency,
        "ok_count": ok_count,
        "error_count": len(results) - ok_count,
        "success_rate": ok_count / len(results) if results else 0.0,
        "latency_ms": {
            "min": min(latencies) if latencies else 0.0,
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else 0.0,
        },
        "statuses": sorted({item["status"] for item in results}, key=lambda value: (value is None, value)),
        "errors": [item["error"] for item in results if item.get("error")][:5],
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.report_path:
        Path(args.report_path).write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
