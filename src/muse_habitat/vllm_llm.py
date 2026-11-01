"""PARTNR LLM backend: raw completions from a local vLLM server.

The planner builds one prompt string from the chat tags in the llm config (system/user/assistant/eot), so the model's
own tags go in the yaml and the string is sent as is. Constrained generation passes the planner's grammar (GBNF-style
EBNF, as for transformers-CFG) to the server's structured outputs instead of a logits processor.

Native thinking (`thinking` block in the llm config): when the prompt ends with the assistant tag, the turn is
generated in two calls, as the model's chat template renders an assistant message with reasoning: first the thinking
channel (unconstrained, up to `thinking.max_tokens`), then the answer channel with the grammar. vLLM's own
"grammar after reasoning" cannot do this for Glimmer: its reasoning parser only ends reasoning at a tool channel, never
at a `to=user` answer. Thinking is not kept in the planner's history, as chat clients drop earlier reasoning.
"""
import json
import time
import urllib.request

from habitat_llm.llm.base_llm import BaseLLM

# root rule of LLMPlanner.build_response_grammar: a free-text line, then the action or termination
_ROOT_WITH_THOUGHT = 'root ::= free_text "\\n" '


class VLLMCompletion(BaseLLM):
    def __init__(self, conf):
        super().__init__(conf)
        self.url = f"http://{conf.host}:{conf.port}/v1/completions"
        self.response = None
        self.calls = 0

    def _complete(self, prompt, max_tokens, stop, grammar=None, keep_special=False):
        gp = self.generation_params
        body = {"model": gp.engine, "prompt": prompt, "max_tokens": max_tokens, "temperature": gp.temperature,
                "stop": stop}
        if keep_special:
            body["skip_special_tokens"] = False  # so special-token stop strings (<|eom|>) match
        if grammar is not None:
            body["structured_outputs"] = {"grammar": grammar}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.llm_conf.timeout) as r:
            out = json.loads(r.read())
        return out["choices"][0]["text"], out["usage"]["completion_tokens"]

    def generate(self, prompt, stop=None, max_length=None, generation_args=None, **kwargs):
        if not isinstance(prompt, str):
            raise NotImplementedError("VLLMCompletion takes text prompts only")
        gp = self.generation_params
        if stop is None:
            stop = gp.stop
        stop = [stop] if isinstance(stop, str) else list(stop)
        # The planner closes the system block with eot_tag; some templates end it with extra lines (Glimmer: reasoning
        # strength and the valid recipients), which system_suffix adds before the first eot
        suffix = self.llm_conf.get("system_suffix", "")
        if suffix and prompt.startswith(self.llm_conf.system_tag):
            prompt = prompt.replace(self.llm_conf.eot_tag, suffix + self.llm_conf.eot_tag, 1)
        grammar = None
        if generation_args is not None and "grammar_definition" in generation_args:
            grammar = generation_args["grammar_definition"]

        t0, n_think = time.time(), 0
        think = self.llm_conf.get("thinking")
        assistant = self.llm_conf.assistant_tag
        if think and prompt.endswith(assistant):
            base = prompt[: -len(assistant)] + think.open
            thought, n_think = self._complete(base, think.max_tokens, list(think.stop), keep_special=True)
            for s in think.stop:
                thought = thought.split(s)[0]
            prompt = base + thought + think.close + assistant
            # The planner's grammar starts with a free-text line (the ReAct thought) whose characters include
            # "Assigned!"; with the reasoning done in the thinking channel the answer is the action alone
            if grammar is not None:
                grammar = grammar.replace(_ROOT_WITH_THOUGHT, "root ::= ", 1)
            print(f"[{type(self).__name__}] thinking ({n_think} tokens): {thought.strip()}", flush=True)

        text, n_answer = self._complete(prompt, max_length if max_length is not None else gp.max_tokens, stop,
                                        grammar)
        self.calls += 1
        print(f"[{type(self).__name__}] call {self.calls}: think {n_think} + answer {n_answer} tokens, "
              f"{time.time() - t0:.1f} s", flush=True)
        # Same post-processing as habitat_llm.llm.Llama: cut at the stop word, strip trailing space
        for s in stop:
            text = text.split(s)[0]
        self.response = text.rstrip()
        return self.response
