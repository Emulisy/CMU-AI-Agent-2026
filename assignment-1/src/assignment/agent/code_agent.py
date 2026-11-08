"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        self.task_prompt = task

        system_information = json.dumps(
            {
                "machine": self.env.machine,
                "release": self.env.release,
                "system": self.env.system,
                "version": self.env.version,
            },
            indent=2,
        )
        self.system_prompt = (
            "You are a coding agent that fixes software issues in a sandbox. "
            "Inspect the relevant files to understand the task, make focused "
            "changes, and run appropriate tests to verify the result. "
            "Use the execute tool to run commands and inspect their results "
            "before deciding what to do next. Base your conclusions on observed "
            "results rather than assuming commands succeeded.\n\n"
            "<system_information>\n"
            f"{system_information}\n"
            "</system_information>"
        )
        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        return_msg: list[dict[str, str]] = []
        for call in tool_calls:
            try:
                tool_name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be a JSON object")

                match tool_name:
                    case "execute":
                        content = self.tool_execute(args)
                    case "send_message":
                        content = self.tool_send_message(args)
                    case _:
                        content = f"Tool {tool_name} not supported"
            except (KeyError, TypeError, ValueError) as exc:
                content = f"Tool call failed: {exc}"

            return_msg.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content,
                }
            )
        return return_msg

    def tool_execute(self, args: dict[str, Any]) -> str:
        allowed = {"command", "shell", "cwd", "timeout", "env"}
        if args.keys() - allowed:
            raise ValueError("Unsupported execute arguments")
        command = args["command"]
        if not isinstance(command, str) and not (
            isinstance(command, list)
            and all(isinstance(part, str) for part in command)
        ):
            raise ValueError("command must be a string or a list of strings")

        result = self.env.execute(
            command,
            shell=args.get("shell", True),
            cwd=args.get("cwd"),
            timeout=args.get("timeout"),
            env=args.get("env"),
        )
        return format_tool_output(result)

    def tool_send_message(self, args: dict[str, Any]) -> str:
        if args.keys() - {"summary"}:
            raise ValueError("Unsupported send_message arguments")
        summary = args["summary"]
        if not isinstance(summary, str):
            raise ValueError("summary must be a string")
        print(summary)
        self.finished = True
        return "Message delivered."
