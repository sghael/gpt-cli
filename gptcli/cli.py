import re
from typing import Any, Dict, Optional, Tuple

from openai import BadRequestError, OpenAIError
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.key_binding.bindings import named_commands

from rich.console import Console, ConsoleOptions, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.text import Text
from rich.layout import Layout
from rich.group import Group
from .markdown import CustomMarkdown
from gptcli.completion import Message

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

TERMINAL_WELCOME = """
>
"""


class StreamingMarkdownPrinter:
    def __init__(self, console: Console, markdown: bool):
        self.console = console
        self.markdown = markdown
        self.current_text = ""
        self.first_token = True
        self.live: Optional[Live] = None
        
        # Create a layout for chat history
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="history", ratio=1),
            Layout(name="input_area", size=2)  # Area for input prompt
        )
        
        # Initialize chat history
        self.chat_history = []
    def add_to_history(self, text: str, role: str = "assistant"):
        """Add a message to chat history"""
        self.chat_history.append((text, role))
        
    def __enter__(self) -> "StreamingMarkdownPrinter":
        self.live = Live(
            self.layout,
            console=self.console,
            refresh_per_second=20,  # Increased for smoother updates
            vertical_overflow="visible",
            auto_refresh=True,
            transient=False  # Changed to preserve history
        )
        self.live.__enter__()
        return self

    def _render_chat_history(self) -> RenderableType:
        """Render the entire chat history including current response"""
        rendered_history = []
        for text, role in self.chat_history:
            if role == "user":
                rendered_history.append(Text(f"\n> {text}\n", style="bold green"))
            else:
                if self.markdown:
                    rendered_history.append(self._render_partial_markdown(text))
                else:
                    rendered_history.append(Text(text))
                rendered_history.append(Text("\n"))
                
        # Add current response if any
        if self.current_text:
            if self.markdown:
                rendered_history.append(self._render_partial_markdown(self.current_text))
            else:
                rendered_history.append(Text(self.current_text))
                
        return Group(*rendered_history)

    def _render_partial_markdown(self, text: str) -> RenderableType:
        """Render potentially incomplete markdown, handling unclosed blocks"""
        if not self.markdown:
            return Text(text)
            
        # Add temporary closing markers for unclosed code blocks
        temp_text = text
        backtick_count = temp_text.count("```")
        if backtick_count % 2 == 1:
            # Odd number of backticks means unclosed code block
            temp_text += "\n```"
            
        try:
            # Use CustomMarkdown with the temporary text
            content = CustomMarkdown(
                temp_text,
                code_theme="monokai",
                justify="left"
            )
            return content
        except Exception:
            # Fallback to plain text if markdown parsing fails
            return Text(text)

    def print(self, text: str):
        if self.first_token and text.startswith(" "):
            text = text[1:]
        self.first_token = False

        self.current_text += text
        
        # Update the layout with full chat history
        self.layout["history"].update(
            Padding(
                self._render_chat_history(),
                (0, 2, 0, 2)
            )
        )
        
        # Update input area
        self.layout["input_area"].update(
            Text("> ", style="bold green")
        )
        
        if self.live:
            self.live.update(self.layout)

    def __exit__(self, *args):
        if self.live:
            # Add completed response to history before exiting
            if self.current_text:
                self.add_to_history(self.current_text, "assistant")
            self.live.__exit__(*args)


class CLIResponseStreamer(ResponseStreamer):
    def __init__(self, console: Console, markdown: bool):
        self.console = console
        self.markdown = markdown
        self.printer = StreamingMarkdownPrinter(self.console, self.markdown)

    def __enter__(self):
        self.printer.__enter__()
        self.console.print()  # Add a newline between prompt and output
        return self

    def on_next_token(self, token: str):
        self.printer.print(token)

    def __exit__(self, *args):
        self.printer.__exit__(*args)


class CLIChatListener(ChatListener):
    def __init__(self, markdown: bool):
        self.markdown = markdown
        self.console = Console()
        self.current_printer: Optional[StreamingMarkdownPrinter] = None

    def on_chat_message(self, message: Message):
        if self.current_printer and message["role"] == "user":
            self.current_printer.add_to_history(message["content"], "user")

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
    # Extract parts enclosed in specific delimiters
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
    pattern_fragments = [re.escape(d) + "(.*?)" + re.escape(d) for d in delimiters]
    pattern = re.compile("|".join(pattern_fragments), re.DOTALL)

    input = pattern.sub(replacer, input)

    # Parse the remaining string for arguments
    args = {}
    regex = r"--(\w+)(?:=(\S+)|\s+(\S+))?"
    matches = re.findall(regex, input)

    if matches:
        for key, value1, value2 in matches:
            value = value1 if value1 else value2 if value2 else ""
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
