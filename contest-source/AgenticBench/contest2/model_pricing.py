"""Frozen off-peak DeepSeek Flash Responses API accounting in Chinese yuan.

This is a trusted proxy component, not a participant cost declaration. The
proxy must verify its provider model mapping and account currency at release.
"""
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal


MODEL_ID = "deepseek-flash"
MODEL_REVISION = "DeepSeek-V4.1-Flash-1017"
CONTEXT_WINDOW = 1048576
MAX_OUTPUT_PER_CALL = 16384
RMB_PER_MILLION = {"cache_hit_input": Decimal("0.02"),
                   "cache_miss_input": Decimal("1"),
                   "output": Decimal("4")}
_MILLION = Decimal(1000000)
_BEIJING = timezone(timedelta(hours=8), "Asia/Shanghai")


def conservative_offpeak(moment):
    """Allow weekends and weekday hours outside published peak windows.

    Public holidays would also be off-peak, but are conservatively treated as
    ordinary weekdays until the organizer freezes an authoritative calendar.
    """
    if not isinstance(moment, datetime) or moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("Off-peak decision requires a timezone-aware datetime")
    local = moment.astimezone(_BEIJING)
    if local.weekday() >= 5:
        return True
    clock = local.time()
    return not (time(9) <= clock < time(12) or time(14) <= clock < time(18))


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer token count")
    return value


def peak_during(start, end):
    """Use the higher tariff if any part of a provider call touches peak time."""
    if (not isinstance(start, datetime) or not isinstance(end, datetime)
            or start.tzinfo is None or end.tzinfo is None or end < start):
        raise ValueError("Pricing interval requires ordered aware datetimes")
    if end - start > timedelta(days=1):
        return True
    cursor = start
    while cursor <= end:
        if not conservative_offpeak(cursor):
            return True
        cursor += timedelta(minutes=1)
    return not conservative_offpeak(end)


def charge_usage(usage, *, peak=False):
    """Compute exact charge from the provider's terminal Responses usage."""
    if not isinstance(usage, dict):
        raise ValueError("Provider usage must be an object")
    prompt = _count(usage.get("input_tokens"), "input_tokens")
    details = usage.get("input_tokens_details")
    if not isinstance(details, dict):
        raise ValueError("Provider input token details are missing")
    hit = _count(details.get("cached_tokens"), "cached_tokens")
    miss = prompt - hit
    if miss < 0:
        raise ValueError("Cached tokens exceed input tokens")
    output = _count(usage.get("output_tokens"), "output_tokens")
    total = _count(usage.get("total_tokens"), "total_tokens")
    if prompt + output != total or prompt + output > CONTEXT_WINDOW or output > MAX_OUTPUT_PER_CALL:
        raise ValueError("Provider usage token totals are inconsistent")
    tools = usage.get("server_tool_use", {})
    if not isinstance(tools, dict) or any(_count(tools.get(k, 0), k) != 0 for k in ("tool_search", "tool_fetch")):
        raise ValueError("Unmetered server-side tool use is forbidden")
    multiplier = 2 if peak else 1
    amount = (Decimal(hit) * RMB_PER_MILLION["cache_hit_input"]
              + Decimal(miss) * RMB_PER_MILLION["cache_miss_input"]
              + Decimal(output) * RMB_PER_MILLION["output"]) * multiplier / _MILLION
    return amount


def max_charge(input_token_upper_bound, max_output_tokens, *, peak=False):
    """Reserve a trusted upper bound using cache-miss input and output rates."""
    inputs = _count(input_token_upper_bound, "input_token_upper_bound")
    outputs = _count(max_output_tokens, "max_output_tokens")
    if outputs == 0:
        raise ValueError("A provider request needs a positive output limit")
    multiplier = 2 if peak else 1
    return (Decimal(inputs) * RMB_PER_MILLION["cache_miss_input"]
            + Decimal(outputs) * RMB_PER_MILLION["output"]) * multiplier / _MILLION


def reserve_full_context(*, peak=False):
    """A provider-enforced context bound makes this a true pre-call upper bound."""
    return max_charge(CONTEXT_WINDOW, MAX_OUTPUT_PER_CALL, peak=peak)
