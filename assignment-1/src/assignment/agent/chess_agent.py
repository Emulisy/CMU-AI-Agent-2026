"""The Part 3 chess agent: reuse the shared loop with one new domain tool."""

from __future__ import annotations

import json
from typing import Any

import httpx

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
)
from assignment.agent.chess_tools import (
    _game_state,
    _invoke_skill,
    _play_move,
    CHESS_PORT,
    _run_python,
    _simulate_move,
)
from assignment.agent.tools import (
    INVOKE_SKILL_TOOL,
    PLAY_MOVE_TOOL,
    RUN_PYTHON_TOOL,
    SIMULATE_MOVE_TOOL,
)
from assignment.prompts import (
    CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE,
    CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE,
    PROGRAMMATIC_CHESS_PROMPT,
)
from assignment.env import Environment


def format_chess_state(
    state: dict[str, Any], *, include_legal_moves: bool = True
) -> str:
    """Turn chess API JSON into a compact observation an LLM can act on."""

    squares = state.get("squares", {})
    board_lines = ["    a b c d e f g h"]
    for rank in range(8, 0, -1):
        pieces = [squares.get(f"{file}{rank}", ".") for file in "abcdefgh"]
        board_lines.append(f"{rank} | {' '.join(pieces)} | {rank}")
    board_lines.append("    a b c d e f g h")

    recent_history = state.get("history", [])[-8:]
    history = (
        " ".join(
            f"{item.get('ply', '?')}:{item.get('san', '?')}" for item in recent_history
        )
        or "(none)"
    )
    legal_moves = " ".join(state.get("legal_moves", [])) or "(none)"
    board_text = "\n".join(board_lines)

    observation = (
        "<chess_state>\n"
        f"status: {state.get('status', 'unknown')}\n"
        f"turn: {state.get('turn', 'unknown')}\n"
        f"in_check: {state.get('in_check', False)}\n"
        f"game_over: {state.get('game_over', False)}\n"
        f"fen: {state.get('fen', '')}\n"
        f"human_move: {state.get('human_move') or '(none)'}\n"
        f"engine_move: {state.get('engine_move') or '(none)'}\n"
        "board:\n"
        f"{board_text}\n"
        f"recent_history: {history}\n"
    )
    if include_legal_moves:
        observation += f"legal_moves: {legal_moves}\n"
    return observation + "</chess_state>"


class ChessAgent(Agent):
    """An agent that plays White against the server's deterministic Black bot."""

    def __init__(
        self,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 200,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        http_client: Any | None = None,
        reset_game: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = None,
        compaction_max_tokens: int = None,
        programmatic_tools: bool = False,
        python_sandbox_port: int = CHESS_PORT,
        include_legal_moves: bool = True,
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

        # TODO(Part 3): Register the play_move tool schema from tools.py.
        self.tools.append(PLAY_MOVE_TOOL)

        if programmatic_tools:
            self.tools.append(RUN_PYTHON_TOOL)
            self.tools.append(SIMULATE_MOVE_TOOL)

        # run_python always executes in the sandbox, on the port the chess
        # server is listening on there.
        self.python_sandbox_port = python_sandbox_port
        self.include_legal_moves = include_legal_moves

        if http_client is None:
            server_url = getattr(environment, "server_url", "")
            if not server_url:
                raise ValueError(
                    "ChessAgent needs an environment with server_url or an http_client."
                )
            http_client = httpx.Client(base_url=server_url, timeout=20)
        self.chess_client = http_client

        initial_state = _game_state(self.chess_client, reset=reset_game)
        self.last_state = initial_state
        self.finished = bool(initial_state.get("game_over"))
        prompt_template = (
            CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE
            if include_legal_moves
            else CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE
        )
        self.system_prompt = prompt_template.render()
        if programmatic_tools:
            self.system_prompt += "\n\n" + PROGRAMMATIC_CHESS_PROMPT
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call `invoke_skill` with a "
                "skill's name to load its instructions, and follow them in place "
                f"of your default approach.\n\n<skills>\n{catalog}\n</skills>\n"
            )
        opening_instruction = (
            "Choose one move from legal_moves and call play_move."
            if include_legal_moves
            else "Infer a legal move from the board and FEN, then call play_move."
        )
        self.task_prompt = (
            f"Play this game as White. {opening_instruction}\n\n"
            f"{self.format_state(initial_state)}"
        )

    def format_state(self, state: dict[str, Any]) -> str:
        """Format a state according to this run's observation condition."""

        return format_chess_state(state, include_legal_moves=self.include_legal_moves)

    def build_prompt(self) -> list[dict[str, Any]]:
        """Remind a stalled model to act, using the last observed game state."""

        messages = super().build_prompt()
        if (
            self.finished
            or self.last_state.get("game_over")
            or not self.history
            or self.history[-1].get("role") != "assistant"
            or self.history[-1].get("tool_calls")
        ):
            return messages

        programmatic = any(
            tool.get("function", {}).get("name") == "run_python"
            for tool in self.tools
        )
        next_action = (
            "Use run_python to plan and commit one move, following any loaded skill."
            if programmatic
            else "Call play_move with one White move in UCI notation."
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your last response did not invoke a tool. The last observed "
                    "server state below is not terminal; only the server's "
                    "game_over field determines whether the game has ended. "
                    "Keep reasoning concise and produce a tool call instead of "
                    "commentary or a claim that the game has concluded. "
                    f"{next_action}\n\n{self.format_state(self.last_state)}"
                ),
            }
        )
        return messages

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute model-generated ``play_move`` calls against the chess API."""

        # TODO(Part 3.1):
        # 1. Dispatch on the function name, and ignore a tool this agent did
        #    not register.
        # 2. Hand the raw arguments to the matching chess_tools helper, with
        #    self.chess_client as its first argument. Each helper takes the
        #    client explicitly so the same code can run inside the sandbox.
        # 3. Format a played move with self.format_state, then update
        #    last_state and finished.
        # 4. Link every observation to its call with tool_call_id.
        # 5. Turn malformed, unknown, rejected, or extra parallel calls into
        #    recoverable <chess_error> observations instead of crashing.
        # TODO(Part 3.3-4): add cases for simulate_move and run_python, with
        # linked observations and recoverable errors, just like play_move.
        rtrn_msg: list[dict[str, str]] = []
        moved = False
        registered_names = {
            item.get("function", {}).get("name") for item in self.tools
        }

        for tool in tool_calls:
            try:
                name = tool["function"]["name"]
                args = tool["function"]["arguments"]
                if name not in registered_names:
                    raise ValueError(f"Tool {name!r} is not registered")

                match name:
                    case "play_move":
                        if moved:
                            raise ValueError(
                                "Only one move may be attempted per assistant turn; "
                                "inspect the returned board before choosing another"
                            )
                        if self.finished:
                            raise ValueError("The game is already over")
                        # Count the attempt even if the server rejects it: a
                        # later parallel call must wait for a new observation.
                        moved = True
                        msg = _play_move(self.chess_client, args)
                        if not msg.startswith("<chess_error>"):
                            state = json.loads(msg)
                            msg = self.format_state(state)
                            self.last_state = state
                            self.finished = bool(state.get("game_over"))
                    case "simulate_move":
                        msg = _simulate_move(self.chess_client, args)
                    case "run_python":
                        if moved:
                            raise ValueError(
                                "Only one live-action tool may run per assistant turn; "
                                "inspect the returned board before choosing another"
                            )
                        moved = True
                        msg = _run_python(self.env, self.python_sandbox_port, args)
                        # A snippet can commit a move before reporting an error,
                        # so refresh even when the runner reports a failure.
                        try:
                            state = _game_state(self.chess_client)
                            formatted_state = self.format_state(state)
                            self.last_state = state
                            self.finished = bool(state.get("game_over"))
                            msg += "\n\n" + formatted_state
                        except Exception as exc:
                            # Keep the runner output if the board refresh fails.
                            msg += (
                                "\n\n<chess_error>Could not refresh the live board: "
                                f"{exc}</chess_error>"
                            )
                    case "invoke_skill":
                        msg = _invoke_skill(self.skills, args)
                    case _:
                        raise ValueError(f"Tool {name!r} is not implemented yet")
            except Exception as exc:
                msg = f"<chess_error>{exc}</chess_error>"

            # Confirm a shutdown after a tool failure. A failed liveness check
            # alone is inconclusive, so transient network errors stay recoverable.
            # Keep the fatal raise outside the recovery block above.
            if "<chess_error>" in msg:
                is_alive = getattr(self.env, "is_alive", None)
                if callable(is_alive):
                    try:
                        sandbox_alive = is_alive()
                    except Exception:
                        sandbox_alive = None
                    if sandbox_alive is False:
                        raise RuntimeError(
                            "Chess sandbox has stopped; aborting the agent run "
                            "instead of retrying unavailable tools."
                        )

            rtrn_msg.append({
                "role": "tool",
                "tool_call_id": tool["id"],
                "content": msg,
            })
        return rtrn_msg
