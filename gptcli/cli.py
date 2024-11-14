import re
from typing import Any, Dict, Tuple

from openai import BadRequestError, OpenAIError
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands
from rich.console import Console

from gptcli.session import (
    ALL_COMMANDS,
    COMMAND_CLEAR,
    COMMAND_QUIT,
    COMMAND_RERUN,
    ChatListener,
    InvalidArgumentError,
    ResponseStreamer,
    UserInputProvider,
)
from .markdown import CustomMarkdown

TERMINAL_WELCOME = """
>
"""


class StreamingMarkdownPrinter:
    def __init__(self, console: Console, markdown: bool):
        self.console = console
        self.markdown = markdown
        self.current_text = ""
        self.first_token = True

    def __enter__(self) -> "StreamingMarkdownPrinter":
        return self

    def print(self, text: str):
        if self.first_token and text.startswith(" "):
            text = text[1:]
        self.first_token = False

        self.current_text += text
        if self.markdown:
            # Stream tokens as plain text
            self.console.print(text, end="", style="green", soft_wrap=True)
        else:
            self.console.print(text, end="", style="green", soft_wrap=True)

    def __exit__(self, *args):
        if self.markdown:
            # Move to a new line before rendering markdown
            self.console.print()
            # Re-render the full content as Markdown
            markdown_content = CustomMarkdown(self.current_text, style="green")
            self.console.print(markdown_content)
        else:
            self.console.print()


class CLIResponseStreamer(ResponseStreamer):
    def __init__(self, console: Console, markdown: bool):
        self.console = console
        self.markdown = markdown
        self.printer = StreamingMarkdownPrinter(self.console, self.markdown)

    def __enter__(self):
        self.printer.__enter__()
        return self

    def on_next_token(self, token: str):
        self.printer.print(token)

    def __exit__(self, *args):
        self.printer.__exit__(*args)


class CLIChatListener(ChatListener):
    def __init__(self, markdown: bool):
        self.markdown = markdown
        self.console = Console()

    def on_chat_start(self):
        self.console.print(CustomMarkdown(TERMINAL_WELCOME))

    def on_chat_clear(self):
        self.console.print("[bold]Cleared the conversation.[/bold]")

    def on_chat_rerun(self, success: bool):
        if success:
            self.console.print("[bold]Re-running the last message.[/bold]")
        else:
            self.console.print("[bold]Nothing to re-run.[/bold]")

    def on_error(self, e: Exception):
        if isinstance(e, BadRequestError):
            self.console.print(
                f"[red]Request Error. The last prompt was not saved: {type(e)}: {e}[/red]"
            )
        elif isinstance(e, OpenAIError):
            self.console.print(
                f"[red]API Error. Type `r` or Ctrl-R to try again: {type(e)}: {e}[/red]"
            )
        elif isinstance(e, InvalidArgumentError):
            self.console.print(f"[red]{e.message}[/red]")
        else:
            self.console.print(f"[red]Error: {type(e)}: {e}[/red]")

    def response_streamer(self) -> ResponseStreamer:
        return CLIResponseStreamer(self.console, self.markdown)


def parse_args(input: str) -> Tuple[str, Dict[str, Any]]:
    # Extract parts enclosed in specific delimiters (triple backticks, triple quotes, single backticks)
    extracted_parts = []
    delimiters = ['```', '"""', '`']

    def replacer(match):
        for i, delimiter in enumerate(delimiters):
            part = match.group(i + 1)
            if part is not None:
                extracted_parts.append((part, delimiter))
                break
        return f"__EXTRACTED_PART_{len(extracted_parts) - 1}__"

    # Construct the regex pattern dynamically from the delimiters list
    pattern_fragments = [re.escape(d) + '(.*?)' + re.escape(d) for d in delimiters]
    pattern = re.compile('|'.join(pattern_fragments), re.DOTALL)

    input = pattern.sub(replacer, input)

    # Parse the remaining string for arguments
    args = {}
    regex = r'--(\w+)(?:=(\S+)|\s+(\S+))?'
    matches = re.findall(regex, input)

    if matches:
        for key, value1, value2 in matches:
            value = value1 if value1 else value2 if value2 else ''
            args[key] = value.strip("\"'")
        input = re.sub(regex, "", input).strip()

    # Add back the extracted parts, with enclosing backticks or quotes
    for i, (part, delimiter) in enumerate(extracted_parts):
        input = input.replace(
            f"__EXTRACTED_PART_{i}__", f"{delimiter}{part.strip()}{delimiter}"
        )

    return input, args


class CLIFileHistory(FileHistory):
    def append_string(self, string: str) -> None:
        if string in ALL_COMMANDS:
            return
        return super().append_string(string)


class CLIUserInputProvider(UserInputProvider):
    def __init__(self, history_filename) -> None:
        self.prompt_session = PromptSession[str](
            history=CLIFileHistory(history_filename)
        )

    def get_user_input(self) -> Tuple[str, Dict[str, Any]]:
        while (next_user_input := self._request_input()) == "":
            pass

        user_input, args = self._parse_input(next_user_input)
        return user_input, args

    def prompt(self, multiline=False):
        bindings = KeyBindings()

        bindings.add("c-a")(named_commands.get_by_name("beginning-of-line"))
        bindings.add("c-b")(named_commands.get_by_name("backward-char"))
        bindings.add("c-e")(named_commands.get_by_name("end-of-line"))
        bindings.add("c-f")(named_commands.get_by_name("forward-char"))
        bindings.add("c-left")(named_commands.get_by_name("backward-word"))
        bindings.add("c-right")(named_commands.get_by_name("forward-word"))

        @bindings.add("c-c")
        def _(event: KeyPressEvent):
            if len(event.current_buffer.text) == 0 and not multiline:
                event.current_buffer.text = COMMAND_CLEAR[0]
                event.current_buffer.cursor_right(len(COMMAND_CLEAR[0]))
            else:
                event.app.exit(exception=KeyboardInterrupt, style="class:aborting")

        @bindings.add("c-d")
        def _(event: KeyPressEvent):
            if len(event.current_buffer.text) == 0:
                if not multiline:
                    event.current_buffer.text = COMMAND_QUIT[0]
                event.current_buffer.validate_and_handle()

        @bindings.add("c-r")
        def _(event: KeyPressEvent):
            if len(event.current_buffer.text) == 0:
                event.current_buffer.text = COMMAND_RERUN[0]
                event.current_buffer.validate_and_handle()

        try:
            return self.prompt_session.prompt(
                "> " if not multiline else "multiline> ",
                vi_mode=True,
                multiline=multiline,
                enable_open_in_editor=True,
                key_bindings=bindings,
            )
        except KeyboardInterrupt:
            return ""

    def _request_input(self):
        line = self.prompt()

        if line != "\\":
            return line

        return self.prompt(multiline=True)

    def _parse_input(self, input: str) -> Tuple[str, Dict[str, Any]]:
        input, args = parse_args(input)
        return input, args
