"""Deterministic manifest sharding for isolated CPU execution."""

from __future__ import annotations

from collections import defaultdict


def shard_cpu_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Alternate solver placement between two nodes for every problem group."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    order: list[tuple] = []
    for record in records:
        key = (record["n"], record["p"], record["alpha"], record["seed"])
        if key not in groups:
            order.append(key)
        groups[key].append(record)

    shards: tuple[list[dict], list[dict]] = ([], [])
    for group_index, key in enumerate(order):
        jobs = groups[key]
        if len(jobs) % 2:
            raise ValueError(f"problem group {key} has an odd number of jobs")
        for solver_index, job in enumerate(jobs):
            shards[(group_index + solver_index) % 2].append(job)
    return shards
