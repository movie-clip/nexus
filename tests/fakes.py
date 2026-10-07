"""Scripted stand-ins for Claude: no API calls, no cost."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

MODEL = "claude-opus-5-5"


class ScriptedModel(GenericFakeChatModel):
    """Replays fixed AI messages and records what the agent sent it."""

    seen_tools: list[list[str]] = []
    seen_prompts: list[list] = []

    def bind_tools(self, tools, **kwargs):
        self.seen_tools.append([getattr(t, "name", None) or t.get("name") for t in tools])
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen_prompts.append(messages)
        return super()._generate(messages, *args, **kwargs)


def scripted(*steps: AIMessage) -> ScriptedModel:
    model = ScriptedModel(messages=iter(steps))
    model.seen_tools, model.seen_prompts = [], []
    return model


def tool_call(name: str, **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{name}"}])


def final(
    text: str, input_tokens: int = 0, output_tokens: int = 0, model: str = MODEL
) -> AIMessage:
    return AIMessage(
        content=text,
        response_metadata={"model_name": model},
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


class ScriptedOrchestrator:
    """Replays orchestrator decisions through ``with_structured_output``.

    Each decision is validated against the schema the pipeline offered, as the
    API's structured output would enforce it. Records every call's schema and
    messages.
    """

    def __init__(self, *decisions: dict) -> None:
        self.decisions = list(decisions)
        self.calls: list[tuple[type, list]] = []

    def with_structured_output(self, schema, method: str):
        assert method == "json_schema", "forced tool use would be rejected by Opus 5.5"

        def respond(messages):
            self.calls.append((schema, messages))
            return schema(**self.decisions.pop(0))

        return RunnableLambda(respond)


def decide(
    next: str,
    brief: str = "",
    answer: str = "",
    reason: str = "because",
    command: str = "",
    args: dict[str, str] | None = None,
) -> dict:
    # FinalDecision has no command/args; pydantic ignores the extra keys.
    return {
        "next": next,
        "reason": reason,
        "brief": brief,
        "command": command,
        "args": [{"name": k, "value": v} for k, v in (args or {}).items()],
        "answer": answer,
    }
