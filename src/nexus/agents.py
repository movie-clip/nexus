"""Building blocks shared by nexus's Deep Agents."""

from collections.abc import Callable

from deepagents import FilesystemPermission
from langchain.agents.middleware import AgentMiddleware, ModelRequest

WRITE_TOOLS = frozenset({"write_file", "edit_file", "delete", "execute"})

# Enforced at the tool layer, so they hold even if a model calls a hidden tool.
READ_ONLY_PERMISSIONS = [
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
]
NO_SECRETS_PERMISSIONS = [
    FilesystemPermission(operations=["read"], paths=["/**/.env", "/**/.env.*"], mode="deny"),
]


class HideTools(AgentMiddleware):
    """Stop offering some tools to the model, for this agent only.

    Deep Agents' own switch (``HarnessProfile.excluded_tools``) is registered
    per model for the whole process, which would also hide write tools from a
    later agent that needs them. Permissions still deny the calls; hiding the
    tools just keeps the model from planning around tools it can't use.
    """

    def __init__(self, names: frozenset[str]) -> None:
        super().__init__()
        self.names = names

    def wrap_model_call(self, request: ModelRequest, handler: Callable):
        tools = [t for t in request.tools if _tool_name(t) not in self.names]
        return handler(request.override(tools=tools))


def _tool_name(tool) -> str | None:
    if isinstance(tool, dict):
        return tool.get("name")
    return getattr(tool, "name", None)
