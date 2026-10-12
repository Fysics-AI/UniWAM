from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadPrefixSplicePlan:
    executed_during_request: int
    keep_old: int
    drop_new: int


def plan_head_prefix_splice(
    *,
    queue_len_at_request: int,
    queue_len_at_response: int,
    prefix_length: int,
    suffix_length: int,
    queue_epoch_at_request: int | None = None,
    queue_epoch_at_response: int | None = None,
) -> HeadPrefixSplicePlan:
    """Align a generated suffix with actions executed while its request was in flight."""
    if (queue_epoch_at_request is None) != (queue_epoch_at_response is None):
        raise ValueError("Queue splice epochs must either both be provided or both be omitted.")
    if (
        queue_epoch_at_request is not None
        and int(queue_epoch_at_request) != int(queue_epoch_at_response)
    ):
        raise ValueError(
            "Queue was reset while inference was in flight: "
            f"request_epoch={queue_epoch_at_request}, "
            f"response_epoch={queue_epoch_at_response}."
        )
    values = {
        "queue_len_at_request": int(queue_len_at_request),
        "queue_len_at_response": int(queue_len_at_response),
        "prefix_length": int(prefix_length),
        "suffix_length": int(suffix_length),
    }
    if any(value < 0 for value in values.values()):
        raise ValueError(f"Queue splice arguments must be non-negative: {values}")
    if values["queue_len_at_response"] > values["queue_len_at_request"]:
        raise ValueError(f"Queue grew while one request was in flight: {values}")
    if values["prefix_length"] > values["queue_len_at_request"]:
        raise ValueError(f"Prefix is longer than the request-time queue: {values}")
    executed = values["queue_len_at_request"] - values["queue_len_at_response"]
    return HeadPrefixSplicePlan(
        executed_during_request=executed,
        keep_old=max(0, values["prefix_length"] - executed),
        drop_new=min(values["suffix_length"], max(0, executed - values["prefix_length"])),
    )
