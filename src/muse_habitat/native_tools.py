"""PARTNR planner with native tool calling (Muse Glimmer's ATEM format), for setup C.

PARTNR's planners keep one prompt string and parse `Tool[args]` lines out of the reply. This one keeps the episode as
Glimmer's own chat turns instead, rendered as its chat template would with `tools=`:
- the skills (Navigate, Pick, Place, ...) plus Done are functions with JSON schemas in the system block;
- each turn the model thinks (to=self channel), then calls one tool: `to=<tool><|message|><atem:function_calls>...`;
  the recipient and the argument values are grammar-constrained to the agent's tools and the world graph's names (as
  tool_choice="required" with enum arguments would be);
- skill results come back as `tool` messages.
The call is turned back into `Tool[a, b, ...]` for the rest of PARTNR, so execution and scoring are unchanged.

Use with the glimmer_vllm_think llm config (server, thinking channel, reasoning strength) and the
zero_shot_native_tools instruct (no action list or reply format in the prompt).
"""
import json
import re
from typing import Dict

from habitat_llm.llm.instruct.utils import get_objects_descr
from habitat_llm.planner.zero_shot_react_planner import ZeroShotReactPlanner
from habitat_llm.utils.grammar import (
    FURNITURE,
    NAV_TARGET,
    OBJECT,
    OBJECT_OR_FURNITURE,
    ROOM,
    SPATIAL_RELATION,
)

# PARTNR argument type -> (parameter name, description); the optional spatial constraint is a pair, see _params
_PARAM = {
    OBJECT: ("object", "Name of an object, as listed in the observations."),
    FURNITURE: ("furniture", "Name of a piece of furniture, as listed in the observations."),
    ROOM: ("room", "Name of a room, as listed in the observations."),
    NAV_TARGET: ("target", "Name of a furniture, room or object to navigate to."),
    OBJECT_OR_FURNITURE: ("target", "Name of an object or a piece of furniture."),
    SPATIAL_RELATION: ("spatial_relation", 'Where to place: "on" or "within".'),
}
_CONSTRAINT = [
    ("spatial_constraint", '"next_to" to place next to a reference object, or "None".', '("next_to" | "None")'),
    ("reference_object", "The object or furniture to place next to, or \"None\".", f'({OBJECT_OR_FURNITURE} | "None")'),
]
_DONE = ("Done", "Used to indicate that the agent has finished the task.")

# Glimmer chat template (render_tool_defs), verbatim
_TOOLS_HEAD = (
    "In this environment you have access to a set of tools you can use to answer the user's question.\n\n"
    'You can invoke a function by writing a "<atem:function_calls>" block like the following:\n'
    '<atem:function_calls>\n<atem:invoke name="$FUNCTION_NAME">\n'
    '<atem:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n'
    "</atem:function_calls>\n\n"
    "String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note "
    "that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with "
    "regular expressions.\n"
    "Here are the functions available in JSONSchema format:\n"
    "// Tool metadata\n"
)
_TOOLS_TAIL = (
    "\n\nHere's an example of how to call a function in the tool set:\n"
    "(If the tool namespace is not specified, invoke the function directly as `example_function_name` rather than "
    "`example_tool_name.example_function_name`)\n\n"
    "to=example_tool_name.example_function_name\n\n"
    '<atem:function_calls>\n<atem:invoke name="example_tool_name.example_function_name">\n'
    '<atem:parameter name="example_parameter_1">value_1</atem:parameter>\n'
    '<atem:parameter name="example_parameter_2">This is the value for the second parameter\nthat can span\n'
    '"multiple" lines\n</atem:parameter>\n'
    "</atem:invoke>\n</atem:function_calls>"
)


def _params(arg_types):
    """[(name, description, grammar rule)] for a skill's argument types."""
    out = []
    for t in arg_types:
        if t in _PARAM:
            out.append((*_PARAM[t], t))
        else:  # Place / Rearrange: optional (spatial_constraint, reference_object), "None, None" when unused
            out.extend(_CONSTRAINT)
    return out


def _q(s):
    """EBNF string literal."""
    return json.dumps(s)


class NativeToolsPlanner(ZeroShotReactPlanner):
    def reset(self):
        super().reset()
        self._turns = None  # Glimmer-format text of the episode so far (system + task + calls + results)
        self._last_tool = None
        # single-call mode: tool calls made, calls with unknown tool / parameter names, turns without a call
        self.native_stats = {"calls": 0, "mismatch": 0, "no_call": 0}

    def _tools(self):
        """{name: [(param, description, rule)]} for this agent's skills, plus Done."""
        tools = {name: _params(tool.argument_types) for name, tool in self.agents[0].tools.items()}
        tools[_DONE[0]] = []
        return tools

    def _tool_defs(self):
        descs = {name: tool.description for name, tool in self.agents[0].tools.items()}
        descs[_DONE[0]] = _DONE[1]
        tools = self._tools()
        text = _TOOLS_HEAD
        for name in tools:
            text += '{"name": ' + json.dumps(name) + ', "description": ' + json.dumps("") + "}\n"
        text += "// Function schemas"
        for name, params in tools.items():
            schema = {"type": "object", "properties": {p: {"type": "string", "description": d} for p, d, _ in params},
                      "required": [p for p, _, _ in params]}
            text += ("\n{\"name\": " + json.dumps(name) + ", \"description\": " + json.dumps(descs[name])
                     + ", \"parameters\": " + json.dumps(schema) + "}")
        recipients = ['"self"'] + [f'"{name}.*"' for name in tools] + ['"user"']
        return text + _TOOLS_TAIL, "# Valid recipients: " + ", ".join(recipients) + "."

    def _start(self):
        """System + task turn from PARTNR's prompt, with the system block completed as Glimmer's template does."""
        c = self.planner_config.llm
        prompt = self.curr_prompt
        if prompt.endswith(c.assistant_tag):
            prompt = prompt[: -len(c.assistant_tag)]
        defs, recipients = self._tool_defs()
        suffix = f"\n\nReasoning strength: {c.reasoning_strength}.\n\n{defs}\n\n{recipients}"
        return prompt.replace(c.eot_tag, suffix + c.eot_tag, 1)

    def _replan_single(self):
        """One unconstrained generation, as native tool calling is normally served: the model thinks, then writes
        the call. The call goes to PARTNR as written (an unknown tool or name gets PARTNR's error as the result).
        None when the turn holds no call (the model answered the user), so the constrained path takes over."""
        llm, think = self.llm, self.llm.llm_conf.thinking
        text, n = llm._complete(self._turns + "<|start|>assistant", think.max_tokens + 400,
                                ["</atem:function_calls>"], keep_special=True)
        call = re.search(r'to=([^<\s]+)<\|message\|>\s*<atem:function_calls>\s*<atem:invoke name="([^"]+)">(.*)$',
                         text, re.S)
        thought = " ".join(re.findall(r"to=self<\|message\|>(.*?)(?:<\|eom\|>|$)", text, re.S)).strip()
        if call is None:
            self.native_stats["no_call"] += 1
            print(f"[NativeToolsPlanner] single call: no tool call in {n} tokens, constrained fallback: "
                  f"{text[-300:]!r}", flush=True)
            return None
        name = call.group(2)
        args = dict(re.findall(r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>', call.group(3), re.S))
        spec = self._tools().get(name)
        if spec is None or sorted(args) != sorted(p for p, _, _ in spec):
            self.native_stats["mismatch"] += 1  # unknown tool or parameter names: passed on as written
            values = list(args.values())
        else:
            values = [args[p].strip() for p, _, _ in spec]
        # history keeps the call as written (the stop string itself is not returned); thinking is dropped
        body = text[text.rfind("<atem:function_calls>"):] + "</atem:function_calls>"
        self._turns += f"<|start|>assistant to={call.group(1)}<|message|>{body}<|eot|>"
        self._last_tool = name
        self.native_stats["calls"] += 1
        action = f"{name}[{', '.join(values)}]"
        print(f"[NativeToolsPlanner] thinking: {thought}\n[NativeToolsPlanner] single call: {action} ({n} tokens) "
              f"stats {dict(self.native_stats)}", flush=True)
        return {"llm_response": action}

    def replan(self, instruction, observations, world_graph):
        if self._turns is None:
            self._turns = self._start()
        if self.llm.llm_conf.get("tool_single_call", False):
            out = self._replan_single()
            if out is not None:
                return out
        llm, wg = self.llm, world_graph[self._agents[0].uid]
        think = llm.llm_conf.thinking
        # 1. think
        base = self._turns + think.open
        thought, n_think = llm._complete(base, think.max_tokens, list(think.stop), keep_special=True)
        for s in think.stop:
            thought = thought.split(s)[0]
        prompt = base + thought + think.close + "<|start|>assistant"
        # 2. recipient: one of the tools usable now (as build_tool_grammar: none needing objects before any is seen)
        has_objects = len(wg.get_all_objects()) != 0
        tools = {n: p for n, p in self._tools().items()
                 if n == _DONE[0] or has_objects or OBJECT not in self.agents[0].tools[n].argument_types}
        header, _ = llm._complete(prompt, 16, [], grammar='root ::= " to=" (' + " | ".join(map(_q, tools)) + ")")
        name = header.strip()[len("to="):]
        prompt += header + "<|message|>"
        # 3. arguments: ATEM block with values from the world graph's names (PARTNR's tool grammar rules)
        body_rule = _q(f'<atem:function_calls>\n<atem:invoke name="{name}">\n')
        for p, _, rule in tools[name]:
            body_rule += " " + _q('<atem:parameter name="' + p + '">') + f" {rule} " + _q("</atem:parameter>\n")
        body_rule += " " + _q("</atem:invoke>\n</atem:function_calls>")
        grammar = "root ::= " + body_rule + "\n" + self.build_tool_grammar(wg)
        body, n_body = llm._complete(prompt, 400, [], grammar=grammar)
        self._turns += f"<|start|>assistant to={name}<|message|>{body}<|eot|>"
        self._last_tool = name
        values = [body.split(f'<atem:parameter name="{p}">', 1)[1].split("</atem:parameter>", 1)[0]
                  for p, _, _ in tools[name]]
        action = f"{name}[{', '.join(values)}]"
        print(f"[NativeToolsPlanner] thinking ({n_think} tokens): {thought.strip()}\n"
              f"[NativeToolsPlanner] call: {action} ({n_body} tokens)", flush=True)
        return {"llm_response": action}

    def _add_responses_to_prompt(self, responses: Dict[int, str]) -> str:
        agent = self.agents[0]
        if self._turns is not None and responses.get(agent.uid):
            objects = get_objects_descr(
                self.env_interface.world_graph[agent.uid], agent.uid, include_room_name=True,
                add_state_info=self.planner_config.objects_response_include_states)
            name = self._last_tool
            self._turns += (f'<|start|>tool {name}<|message|><tool_output name="{name}">\n'
                            f"Result: {responses[agent.uid]}\nObjects: {objects}\n</tool_output><|eot|>")
        return super()._add_responses_to_prompt(responses)
