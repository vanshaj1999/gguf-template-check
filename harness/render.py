"""Render chat templates the way transformers does, over a fixed set of conversations."""

import json
import re

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}

SYSTEM = {"role": "system", "content": "You are a concise assistant. Answer in one sentence."}
Q1 = {"role": "user", "content": "What is 17 multiplied by 23?"}
A1 = {"role": "assistant", "content": "17 multiplied by 23 is 391."}
Q2 = {"role": "user", "content": "And divided by 17?"}
QW = {"role": "user", "content": "What is the weather in Adelaide right now?"}
CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"id": "call00001", "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Adelaide"}}}],
}
RESULT = {"role": "tool", "tool_call_id": "call00001", "name": "get_weather",
          "content": json.dumps({"city": "Adelaide", "temp_c": 24, "sky": "clear"})}

A1_THINK = {"role": "assistant", "content": "<think>\n17 * 23 = 391.\n</think>\n\n391."}
A1_REASON = {"role": "assistant", "content": "391.", "reasoning_content": "17 * 23 = 391."}
CALL_NULL = {**CALL, "content": None}
CALL_STR = {**CALL, "tool_calls": [{**CALL["tool_calls"][0], "function": {
    "name": "get_weather", "arguments": json.dumps({"city": "Adelaide"})}}]}
QW2 = {"role": "user", "content": "Weather in Adelaide and Darwin?"}
CALL2 = {"role": "assistant", "content": "", "tool_calls": [
    {"id": "call00001", "type": "function",
     "function": {"name": "get_weather", "arguments": {"city": "Adelaide"}}},
    {"id": "call00002", "type": "function",
     "function": {"name": "get_weather", "arguments": {"city": "Darwin"}}}]}
RESULT2 = {"role": "tool", "tool_call_id": "call00002", "name": "get_weather",
           "content": json.dumps({"city": "Darwin", "temp_c": 33, "sky": "storms"})}

SCENARIOS = {
    "plain": {"messages": [Q1]},
    "system": {"messages": [SYSTEM, Q1]},
    "multi_turn": {"messages": [SYSTEM, Q1, A1, Q2]},
    "tools": {"messages": [QW], "tools": [WEATHER_TOOL]},
    "system_tools": {"messages": [SYSTEM, QW], "tools": [WEATHER_TOOL]},
    "tool_result": {"messages": [QW, CALL, RESULT], "tools": [WEATHER_TOOL]},
    "thinking_off": {"messages": [SYSTEM, Q1], "enable_thinking": False},
    "history_thinking": {"messages": [Q1, A1_THINK, Q2]},
    "reasoning_field": {"messages": [Q1, A1_REASON, Q2]},
    "null_content_call": {"messages": [QW, CALL_NULL, RESULT], "tools": [WEATHER_TOOL]},
    "string_args_call": {"messages": [QW, CALL_STR, RESULT], "tools": [WEATHER_TOOL]},
    "parallel_calls": {"messages": [QW2, CALL2, RESULT, RESULT2], "tools": [WEATHER_TOOL]},
}

_GEN_TAG = re.compile(r"{%-?\s*(end)?generation\s*-?%}")


def _raise(message):
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                      sort_keys=sort_keys)


def _env() -> ImmutableSandboxedEnvironment:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: "23 Nov 2026" if "%b" in fmt else "2026-11-23"
    return env


ENV = _env()


def compile_template(src: str):
    return ENV.from_string(_GEN_TAG.sub("", src))


def render_all(src: str, bos: str = "", eos: str = "") -> dict[str, str]:
    """Return scenario -> rendered prompt, or 'ERROR: ...' when the template raises."""
    try:
        tpl = compile_template(src)
    except Exception as e:  # noqa: BLE001
        return {name: f"PARSE_ERROR: {type(e).__name__}: {e}" for name in SCENARIOS}
    out = {}
    for name, sc in SCENARIOS.items():
        kwargs = {"add_generation_prompt": True, "bos_token": bos, "eos_token": eos, **sc}
        try:
            out[name] = tpl.render(**kwargs)
        except Exception as e:  # noqa: BLE001
            out[name] = f"ERROR: {type(e).__name__}: {str(e)[:200]}"
    return out


def squash(s: str) -> str:
    return re.sub(r"\s+", "", s)
