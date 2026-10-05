#!/usr/bin/env python3
"""Measure decode stalls while a cold prefill runs in the fourth TP4 slot.

This is a mixed-load extension to RigMark's solo tests, not RigMark protocol 1.3.
Run the same command after each server restart, changing only OWNER_BIDIR.
"""

import argparse
import hashlib
import json
import secrets
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def request_json(base, path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def tokens(base, text):
    out = request_json(base, "/tokenize", {"model": "glm-5.3-flash", "prompt": text,
                                                "add_special_tokens": False})["tokens"]
    if not out or any(type(x) is not int for x in out):
        raise RuntimeError("tokenization failed")
    return out


def make_tokens(base, depth, unit, nonce):
    prefix = tokens(base, f"Unique mixed-load request {nonce}.\n")
    repeated = tokens(base, unit)
    if len(prefix) > depth:
        raise ValueError("prefix exceeds target")
    result = prefix[:]
    while len(result) < depth:
        result.extend(repeated[:depth - len(result)])
    return result


def stream(base, payload, first_event=None):
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    events = []
    usage = {}
    finish_reason = None
    chunks = []
    with urllib.request.urlopen(req, timeout=240) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                value = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
                if value:
                    events.append(time.monotonic())
                    chunks.append(value)
                    if first_event:
                        first_event.set()
    return {"started": started, "ended": time.monotonic(), "events": events,
            "usage": usage, "finish_reason": finish_reason,
            "output_sha256": hashlib.sha256("".join(chunks).encode()).hexdigest()}


def percentile(values, p):
    if not values:
        return None
    xs = sorted(values)
    return xs[round((len(xs) - 1) * p)]


def summarise_decode(result, prefill_start, prefill_first):
    events = result["events"]
    gaps = [(b - a) * 1000 for a, b in zip(events, events[1:])]
    during = [(b - a) * 1000 for a, b in zip(events, events[1:])
              if prefill_start <= a and b <= prefill_first]
    outside = [(b - a) * 1000 for a, b in zip(events, events[1:])
               if b < prefill_start or a > prefill_first]
    return {"completion_tokens": result["usage"].get("completion_tokens"),
            "finish_reason": result["finish_reason"], "sse_events": len(events),
            "output_sha256": result["output_sha256"],
            "first_token_seconds": events[0] - result["started"] if events else None,
            "wall_seconds": result["ended"] - result["started"],
            "all_gap_p95_ms": percentile(gaps, .95),
            "during_gap_p50_ms": percentile(during, .50),
            "during_gap_p95_ms": percentile(during, .95),
            "during_gap_max_ms": max(during) if during else None,
            "during_gap_count": len(during),
            "outside_gap_p50_ms": percentile(outside, .50),
            "outside_gap_p95_ms": percentile(outside, .95),
            "outside_gap_count": len(outside),
            "crossing_gap_ms": [round((b-a)*1000, 3) for a,b in zip(events, events[1:])
                                if a < prefill_start <= b or a <= prefill_first < b],
            "event_offsets_seconds": [round(t - result["started"], 6) for t in events]}


def one_trial(base, depth, unit, repeat, comparison_id):
    nonce = f"{comparison_id}-{depth}-{repeat}"
    salt = secrets.token_hex(12)
    decode_prompt = make_tokens(base, 256, "Write a complete Go implementation of a concurrent token bucket. ", nonce)
    prefill_prompt = make_tokens(base, depth, unit, nonce)
    starts = [threading.Event() for _ in range(3)]
    decode_payloads = [{"model": "glm-5.3-flash", "prompt": decode_prompt,
                        "add_special_tokens": False, "max_tokens": 4096, "ignore_eos": True,
                        "temperature": 0.0, "stream": True,
                        "stream_options": {"include_usage": True},
                        "cache_salt": f"decode-{nonce}-{salt}-{i}"} for i in range(3)]
    prefill_payload = {"model": "glm-5.3-flash", "prompt": prefill_prompt,
                       "add_special_tokens": False, "max_tokens": 8,
                       "ignore_eos": True, "temperature": 0.0, "stream": True,
                       "stream_options": {"include_usage": True},
                       "cache_salt": f"prefill-{nonce}-{salt}"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        decodes = [pool.submit(stream, base, decode_payloads[i], starts[i]) for i in range(3)]
        for started in starts:
            if not started.wait(40):
                raise TimeoutError("decode stream did not start within 40 seconds")
        # Leave a short decode-only segment before the cold prefill begins.
        time.sleep(2)
        prefill = pool.submit(stream, base, prefill_payload)
        prefill_result = prefill.result()
        decode_results = [future.result() for future in decodes]
    prefill_start = prefill_result["started"]
    first = prefill_result["events"][0]
    prefill_usage = prefill_result["usage"]
    cached = (prefill_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached != 0 or prefill_usage.get("prompt_tokens") != depth:
        raise RuntimeError(f"invalid cold prefill: cached={cached}, usage={prefill_usage}")
    if not all(x["events"] and x["ended"] > first for x in decode_results):
        raise RuntimeError("decode did not fully overlap the prefill")
    return {"depth": depth, "repeat": repeat,
            "prefill": {"ttft_seconds": first - prefill_start,
                        "effective_tokens_per_second": depth / (first - prefill_start),
                        "cached_tokens": cached,
                        "completion_tokens": prefill_usage.get("completion_tokens"),
                        "output_sha256": prefill_result["output_sha256"]},
            "decode": [summarise_decode(x, prefill_start, first) for x in decode_results]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18890")
    parser.add_argument("--mode", required=True, choices=["0", "1"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--comparison-id", default="mixed-owner-ring-20261006-v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    unit = "The repository contains a service, tests, documentation, and release notes. Each observation records behaviour, evidence, constraints, and the next action. "
    before = request_json(base, "/health")
    data = {"mode": args.mode, "comparison_id": args.comparison_id,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "health_before": before, "trials": []}
    if before["requests_running"]:
        raise RuntimeError(f"external traffic before run: {before['requests_running']} requests running")
    try:
        for depth in (32768, 65536):
            for repeat in range(1, args.repeats + 1):
                print(f"mode={args.mode} depth={depth} repeat={repeat}", flush=True)
                result = one_trial(base, depth, unit, repeat, args.comparison_id)
                data["trials"].append(result)
                print(f"  cold TTFT {result['prefill']['ttft_seconds']:.2f}s; decode during p95 "
                      + ", ".join(f"{x['during_gap_p95_ms']:.1f}ms" for x in result["decode"]), flush=True)
                args.output.write_text(json.dumps(data, indent=2) + "\n")
        data["health_after"] = request_json(base, "/health")
        data["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.output.write_text(json.dumps(data, indent=2) + "\n")
    except BaseException:
        data["failed"] = True
        args.output.write_text(json.dumps(data, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
