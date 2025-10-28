#!/usr/bin/env python3
# deepscientist_cli.py - Enhanced CLI with Claude Code style
"""
DeepScientistCLI - Professional Research Platform CLI
"""

import ast
import os
import sys
import time
import json
import shlex
import shutil
import difflib
import re
import builtins
from contextlib import suppress, contextmanager
import click
import socketio
import requests
import threading
import queue
import subprocess
import logging
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs
from gitingest import ingest
from rich.console import Console, Group
from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.box import ROUNDED, DOUBLE, HEAVY, DOUBLE_EDGE
from rich import box
from rich.prompt import Prompt
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, Optional, Set, Generator, List

# Optional prompt toolkit enhancements
PROMPT_TOOLKIT_AVAILABLE = False
PyperclipClipboard = None
ANSI = None
PromptSession = None
AutoSuggestFromHistory = None
FileHistory = None
KeyBindings = None

try:  # pragma: no cover - optional dependency
    from prompt_toolkit import PromptSession as _PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory as _AutoSuggestFromHistory
    from prompt_toolkit.formatted_text import ANSI as _ANSI
    from prompt_toolkit.history import FileHistory as _FileHistory
    from prompt_toolkit.key_binding import KeyBindings as _KeyBindings
    try:
        from prompt_toolkit.clipboard.pyperclip import PyperclipClipboard as _PyperclipClipboard
    except Exception:
        _PyperclipClipboard = None

    PROMPT_TOOLKIT_AVAILABLE = True
    PromptSession = _PromptSession
    AutoSuggestFromHistory = _AutoSuggestFromHistory
    ANSI = _ANSI
    FileHistory = _FileHistory
    KeyBindings = _KeyBindings
    PyperclipClipboard = _PyperclipClipboard
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

# Global variables
CLI_VERSION = "v0.3.2"
VERSION_CHECK_ENDPOINT = "/api/version/latest"
VERSION_CHECK_TIMEOUT = 3
SERVER_URL = "http://deepscientist.ai-researcher.net:8888"
console = Console()
CURRENT_CLI_VERSION = CLI_VERSION
UPDATE_AVAILABLE_VERSION = None
UPDATE_CHECK_ERROR = None
UPDATE_CHECK_COMPLETED = False
LATEST_BROADCASTS: List[Dict[str, Any]] = []

# Interactive session state
IN_INTERACTIVE_SESSION = False
_INTERACTIVE_BANNER_SHOWN = False
LAST_LOGIN_PAYLOAD: Optional[Dict[str, Any]] = None

# Load CLI installation directory from config
def _load_cli_config():
    """Load CLI configuration to determine installation directory"""
    # Check environment variable first
    env_install_dir = os.environ.get('DEEPSCIENTIST_INSTALL_DIR')
    if env_install_dir:
        return Path(env_install_dir).expanduser().resolve()

    # Try to find config.json in multiple locations
    default_dir = Path.home() / ".deepscientist"

    # First check if config exists in default location
    config_file = default_dir / "config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                install_dir = config.get('install_dir')
                if install_dir:
                    return Path(install_dir)
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load config: {e}[/yellow]")

    # Also check if we're running from an installed location
    # (install_cli.sh creates config.json in the installation directory)
    script_dir = Path(__file__).parent.parent
    alt_config = script_dir / "config.json"
    if alt_config.exists():
        try:
            with open(alt_config, 'r', encoding='utf-8') as f:
                config = json.load(f)
                install_dir = config.get('install_dir')
                if install_dir:
                    return Path(install_dir)
        except Exception:
            pass

    return default_dir

CONFIG_DIR = _load_cli_config()
INSTALL_CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULT_RESEARCH_STUDIO_URL = "http://ai-researcher.net:3001"
PLACEHOLDER_HOSTS = {'localhost', '127.0.0.1', '0.0.0.0'}


def _load_research_studio_url_from_config() -> Optional[str]:
    if INSTALL_CONFIG_PATH.exists():
        try:
            with open(INSTALL_CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            for key in ('research_studio_url', 'researchStudioUrl', 'researchStudioURL'):
                value = config.get(key)
                if value:
                    return str(value).strip()
        except Exception:
            pass
    return None


def _resolve_research_studio_url() -> str:
    env_value = os.environ.get('DEEPSCIENTIST_RESEARCH_STUDIO_URL') or os.environ.get('RESEARCH_STUDIO_URL')
    if env_value:
        return env_value.rstrip('/')
    config_value = _load_research_studio_url_from_config()
    if config_value:
        return config_value.rstrip('/')
    return DEFAULT_RESEARCH_STUDIO_URL


RESEARCH_STUDIO_URL = _resolve_research_studio_url()


def get_workspace_dir():
    """Get workspace directory from config or use default"""
    # Check environment variable first
    env_workspace_dir = os.environ.get('DEEPSCIENTIST_WORKSPACE_DIR')
    if env_workspace_dir:
        return Path(env_workspace_dir).expanduser().resolve()

    config_file = INSTALL_CONFIG_PATH
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                workspace_dir = config.get('workspace_dir')
                if workspace_dir:
                    return Path(workspace_dir)
        except Exception:
            pass
    return CONFIG_DIR / 'workspace'




# ============================================================================
# Claude Code Stream Processing Classes
# ============================================================================

class ClaudeCodeFilter:
    """Filter and format Claude Code messages in real-time."""

    def __init__(self, show_types: Set[str] = None, show_tools: bool = True):
        self.show_types = show_types or {'assistant', 'result', 'system'}
        self.show_tools = show_tools

        # Color codes for different message types
        self.colors = {
            'system': '\033[94m',      # Blue
            'user': '\033[95m',        # Magenta
            'assistant': '\033[92m',   # Green
            'result': '\033[93m',      # Yellow
            'error': '\033[91m',       # Red
            'reset': '\033[0m'         # Reset
        }

    def should_display(self, json_obj: Dict[str, Any]) -> bool:
        """Determine if message should be displayed based on filters."""
        msg_type = json_obj.get('type', '').lower()

        # Filter by message type
        if self.show_types and msg_type not in self.show_types:
            return False

        # Filter assistant messages without meaningful content
        if msg_type == 'assistant':
            message = json_obj.get('message', {})
            content = message.get('content', [])

            # Skip empty messages
            if not content:
                return False

            # Check if there's actual text content
            has_text = any(
                block.get('type') == 'text' and block.get('text', '').strip()
                for block in content
            )
            has_tools = any(
                block.get('type') == 'tool_use'
                for block in content
            )

            return has_text or (self.show_tools and has_tools)

        return True

    def format_message(self, json_obj: Dict[str, Any]) -> str:
        """Convert JSON message to human-readable format."""
        msg_type = json_obj.get('type', '').lower()
        timestamp = datetime.now().strftime('%H:%M:%S')
        color = self.colors.get(msg_type, '')
        reset = self.colors['reset']

        if msg_type == 'system':
            return self._format_system_message(json_obj, timestamp, color, reset)
        elif msg_type == 'assistant':
            return self._format_assistant_message(json_obj, timestamp, color, reset)
        elif msg_type == 'result':
            return self._format_result_message(json_obj, timestamp, color, reset)
        elif msg_type == 'user':
            return self._format_user_message(json_obj, timestamp, color, reset)
        else:
            return f"{color}[{timestamp}] {msg_type.upper()}: {json_obj}{reset}"

    def _format_system_message(self, obj: Dict[str, Any], timestamp: str, color: str, reset: str) -> str:
        subtype = obj.get('subtype', '')
        if subtype == 'init':
            model = obj.get('model', 'unknown')
            tools = obj.get('tools', [])
            return f"{color}[{timestamp}] SYSTEM: Initialized with {model}, tools: {', '.join(tools)}{reset}"
        return f"{color}[{timestamp}] SYSTEM: {subtype}{reset}"

    def _format_assistant_message(self, obj: Dict[str, Any], timestamp: str, color: str, reset: str) -> str:
        message = obj.get('message', {})
        content_blocks = message.get('content', [])

        formatted_parts = []

        for block in content_blocks:
            block_type = block.get('type', '')

            if block_type == 'text':
                text = block.get('text', '').strip()
                if text:
                    # Clean up text for display
                    text = self._clean_text_for_display(text)
                    formatted_parts.append(f"💬 {text}")

            elif block_type == 'tool_use' and self.show_tools:
                tool_name = block.get('name', '')
                tool_input = block.get('input', {})

                if tool_name == 'Write':
                    path = tool_input.get('file_path', '')
                    formatted_parts.append(f"📝 Writing to: {path}")
                elif tool_name == 'Read':
                    path = tool_input.get('file_path', '')
                    formatted_parts.append(f"📖 Reading: {path}")
                elif tool_name == 'Bash':
                    command = tool_input.get('command', '')
                    formatted_parts.append(f"💻 Running: {command}")
                else:
                    formatted_parts.append(f"🔨 Using tool: {tool_name}")

        if formatted_parts:
            content = '\n    '.join(formatted_parts)
            return f"{color}[{timestamp}] CLAUDE:\n    {content}{reset}"

        return ""

    def _format_user_message(self, obj: Dict[str, Any], timestamp: str, color: str, reset: str) -> str:
        message = obj.get('message', {})
        content_blocks = message.get('content', [])

        formatted_parts = []

        for block in content_blocks:
            block_type = block.get('type', '')

            if block_type == 'text':
                text = block.get('text', '').strip()
                if text:
                    text = self._clean_text_for_display(text)
                    formatted_parts.append(f"💬 {text}")

            elif block_type == 'tool_result' and self.show_tools:
                content = block.get('content', '')
                formatted_parts.append(f"🔨 {content}")

        if formatted_parts:
            content = '\n    '.join(formatted_parts)
            return f"{color}[{timestamp}] USER:\n    {content}{reset}"

        return ""

    def _format_result_message(self, obj: Dict[str, Any], timestamp: str, color: str, reset: str) -> str:
        subtype = obj.get('subtype', '')
        duration_ms = obj.get('duration_ms', 0)
        cost = obj.get('total_cost_usd', 0)
        num_turns = obj.get('num_turns', 0)
        is_error = obj.get('is_error', False)

        status = "❌ ERROR" if is_error else "✅ SUCCESS"
        duration_s = duration_ms / 1000 if duration_ms else 0

        result = f"{color}[{timestamp}] RESULT: {status}\n"
        result += f"    Duration: {duration_s:.2f}s\n"
        result += f"    Turns: {num_turns}\n"
        result += f"    Cost: ${cost:.4f}{reset}"

        return result

    def _clean_text_for_display(self, text: str) -> str:
        """Clean text for better console display."""
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            if len(line) > 500:
                line = line[:497] + "..."
            cleaned_lines.append(line)

        if len(cleaned_lines) > 20:
            cleaned_lines = cleaned_lines[:19] + ["... (truncated)"]

        return '\n    '.join(cleaned_lines)


class ClaudeCodeStreamProcessor:
    """Production-ready Claude Code JSON stream processor."""

    def __init__(self,
                 buffer_size: int = 4096,
                 timeout: float = 30.0,
                 max_retries: int = 3,
                 cwd: str = None,
                 cuda_device: Optional[str] = None,
                 combined_logger: Optional['CombinedStreamLogger'] = None):
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.cwd = cwd
        self.cuda_device = cuda_device
        self.combined_logger = combined_logger
        # Use fast JSON library if available
        try:
            import orjson
            self.json_loads = orjson.loads
        except ImportError:
            self.json_loads = json.loads

    @contextmanager
    def stream_context(self, command: list):
        """Context manager for safe subprocess resource handling."""
        proc = None
        try:
            env_vars = os.environ.copy()
            env_vars['BASH_DEFAULT_TIMEOUT_MS'] = '270000000'
            env_vars['BASH_MAX_TIMEOUT_MS'] = '270000000'

            if self.cuda_device is not None:
                env_vars['CUDA_VISIBLE_DEVICES'] = self.cuda_device

            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True,
                text=True,
                env=env_vars,
                encoding='utf-8',
                errors='replace',
                cwd=self.cwd
            )
            yield proc
        finally:
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()

    def stream_json_objects(self, command: list) -> Generator[Dict[str, Any], None, None]:
        """Stream JSON objects with comprehensive error handling."""
        import select
        import time as time_module

        stderr_lines = []
        last_check_time = time_module.time()

        with self.stream_context(command) as proc:
            # Read stderr in a separate thread to avoid blocking
            import threading

            def read_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line.strip())

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # Use select for non-blocking read with timeout
            while True:
                # Check stop flag every iteration (includes timeout checks)
                current_time = time_module.time()
                if control_flags.get('stop_requested', False) or control_flags.get('should_exit', False):
                    reason = "server termination" if control_flags.get('should_exit', False) else "user request (/q command)"
                    logging.info(f"Stop requested - terminating Claude Code process (reason: {reason})")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    raise KeyboardInterrupt(f"Execution stopped by {reason}")

                # Check if process is still alive
                if proc.poll() is not None:
                    # Process has terminated, read remaining output
                    for line in proc.stdout:
                        line = line.strip()
                        if line:
                            try:
                                obj = self.json_loads(line)
                                if self.combined_logger:
                                    self.combined_logger.log_claude_message(obj)
                                yield obj
                            except:
                                pass
                    break

                # Use select with timeout to check for data (non-blocking)
                try:
                    readable, _, _ = select.select([proc.stdout], [], [], 1.0)  # 1 second timeout

                    if readable:
                        line = proc.stdout.readline()
                        if not line:  # EOF
                            break

                        line = line.strip()
                        if not line:
                            continue

                        try:
                            obj = self.json_loads(line)
                            # Log to combined logger if available
                            if self.combined_logger:
                                self.combined_logger.log_claude_message(obj)
                            yield obj
                        except (json.JSONDecodeError, ValueError) as e:
                            logging.warning(f"Failed to parse JSON line: {e}")
                            repaired = self._attempt_json_repair(line)
                            if repaired:
                                try:
                                    obj = self.json_loads(repaired)
                                    # Log to combined logger if available
                                    if self.combined_logger:
                                        self.combined_logger.log_claude_message(obj)
                                    yield obj
                                    logging.info("Successfully repaired malformed JSON")
                                except:
                                    continue
                except (select.error, OSError):
                    # select failed, likely due to process termination
                    break

            # Wait for stderr reading to complete
            stderr_thread.join(timeout=1.0)

            return_code = proc.poll()
            if return_code and return_code != 0:
                stderr_output = '\n'.join(stderr_lines) if stderr_lines else proc.stderr.read()
                error_msg = f"Claude Code process failed with code {return_code}"
                if stderr_output:
                    error_msg += f"\nSTDERR:\n{stderr_output}"
                logging.error(error_msg)
                # Raise exception so it can be caught and logged by caller
                raise RuntimeError(error_msg)

    def _attempt_json_repair(self, json_str: str) -> Optional[str]:
        """Attempt to fix common JSON formatting issues."""
        open_braces = json_str.count('{') - json_str.count('}')
        if open_braces > 0:
            json_str += '}' * open_braces

        open_brackets = json_str.count('[') - json_str.count(']')
        if open_brackets > 0:
            json_str += ']' * open_brackets

        return json_str if json_str.endswith(('}', ']')) else None


class CombinedStreamLogger:
    """Combined logger for Claude Code output - matches researchagent.py format"""

    def __init__(self, log_file_path: str, idea_id: str):
        self.log_file_path = log_file_path
        self.idea_id = idea_id
        self.log_content = []
        self.messages = []  # Store raw messages for .messages file

    def log_message(self, message: str, source: str = "", print_to_console: bool = True):
        """Log message to both file and memory"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_msg = f"[{timestamp}] {source}: {message}" if source else f"[{timestamp}] {message}"

        # Add to log content
        self.log_content.append(formatted_msg)

        # Write to file immediately
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(formatted_msg + '\n')

        # Print to console only if requested
        if print_to_console:
            print(formatted_msg)

    def log_claude_message(self, json_obj: Dict[str, Any]):
        """Log Claude Code message in simplified format"""
        # Store raw message for .messages file
        self.messages.append(json_obj)

        msg_type = json_obj.get('type', '').lower()
        timestamp = datetime.now().strftime('%H:%M:%S')

        if msg_type == 'system':
            subtype = json_obj.get('subtype', '')
            if subtype == 'init':
                model = json_obj.get('model', 'unknown')
                tools = json_obj.get('tools', [])
                self.log_message(f"CLAUDE SYSTEM: Initialized with {model}, tools: {', '.join(tools)}", "CLAUDE", print_to_console=False)
            else:
                self.log_message(f"CLAUDE SYSTEM: {subtype}", "CLAUDE", print_to_console=False)

        elif msg_type == 'user':
            message = json_obj.get('message', {})
            content_blocks = message.get('content', [])

            for block in content_blocks:
                block_type = block.get('type', '')

                if block_type == 'text':
                    text = block.get('text', '').strip()
                    if text:
                        # Truncate very long user messages for readability
                        display_text = text[:200] + "..." if len(text) > 200 else text
                        self.log_message(f"CLAUDE USER: {display_text}", "CLAUDE", print_to_console=False)

                elif block_type == 'tool_result':
                    content = block.get('content', '')
                    display_content = content[:100] + "..." if len(content) > 100 else content
                    self.log_message(f"CLAUDE USER: Tool result: {display_content}", "CLAUDE", print_to_console=False)

        elif msg_type == 'assistant':
            message = json_obj.get('message', {})
            content_blocks = message.get('content', [])

            for block in content_blocks:
                block_type = block.get('type', '')

                if block_type == 'text':
                    text = block.get('text', '').strip()
                    if text:
                        self.log_message(f"CLAUDE ASSISTANT: {text}", "CLAUDE", print_to_console=False)

                elif block_type == 'tool_use':
                    tool_name = block.get('name', '')
                    tool_input = block.get('input', {})

                    if tool_name == 'Write':
                        path = tool_input.get('file_path', '')
                        self.log_message(f"CLAUDE ASSISTANT: Writing to: {path}", "CLAUDE", print_to_console=False)
                    elif tool_name == 'Read':
                        path = tool_input.get('file_path', '')
                        self.log_message(f"CLAUDE ASSISTANT: Reading: {path}", "CLAUDE", print_to_console=False)
                    elif tool_name == 'Bash':
                        command = tool_input.get('command', '')
                        self.log_message(f"CLAUDE ASSISTANT: Running: {command}", "CLAUDE", print_to_console=False)
                    else:
                        self.log_message(f"CLAUDE ASSISTANT: Using tool: {tool_name}", "CLAUDE", print_to_console=False)

        elif msg_type == 'result':
            is_error = json_obj.get('is_error', False)
            duration_ms = json_obj.get('duration_ms', 0)
            duration_s = duration_ms / 1000 if duration_ms else 0
            num_turns = json_obj.get('num_turns', 0)
            cost = json_obj.get('total_cost_usd', 0)

            status = "ERROR" if is_error else "SUCCESS"
            self.log_message(f"CLAUDE RESULT: {status} (Duration: {duration_s:.2f}s, Turns: {num_turns}, Cost: ${cost:.4f})", "CLAUDE", print_to_console=False)

    def save_messages(self, messages_file_path: str):
        """Save raw messages to .messages file"""
        with open(messages_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def finalize_log(self):
        """Finalize log - compatibility method"""
        pass


class ClaudeCodeStreamManager:
    """Complete streaming manager with real-time filtering and display."""

    def __init__(self,
                 show_types: Set[str] = None,
                 show_tools: bool = True,
                 max_line_length: int = 500,
                 cwd: str = None,
                 cuda_device: Optional[str] = None,
                 log_file: str = None):
        self.processor = ClaudeCodeStreamProcessor(cwd=cwd, cuda_device=cuda_device)
        self.filter = ClaudeCodeFilter(show_types, show_tools)
        self.max_line_length = max_line_length
        self.cwd = cwd
        self.cuda_device = cuda_device
        self.log_file = log_file
        self.stats = {
            'total_messages': 0,
            'displayed_messages': 0,
            'errors': 0,
            'start_time': None
        }

    def process_claude_stream(self, command: list):
        """Process Claude Code stream with real-time output."""
        self.stats['start_time'] = time.time()

        try:
            for json_obj in self.processor.stream_json_objects(command):
                # Check if stop was requested (redundant check for safety)
                if control_flags.get('stop_requested', False):
                    logging.info("Stop requested in process_claude_stream - halting")
                    raise KeyboardInterrupt("Execution stopped by user request (/q command)")

                self.stats['total_messages'] += 1

                # Write to log file if provided
                if self.log_file:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(json_obj) + '\n')

                # Display if should
                if self.filter.should_display(json_obj):
                    formatted = self.filter.format_message(json_obj)
                    if formatted:
                        print(formatted)
                        self.stats['displayed_messages'] += 1

                        # Add to task_status for UI display
                        task_status['claude_logs'].append({
                            'timestamp': datetime.now().isoformat(),
                            'formatted': formatted,
                            'type': json_obj.get('type', '')
                        })
                        control_flags['force_ui_update'] = True
                clamp_event_scroll()

        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to propagate stop signal
            logging.info("KeyboardInterrupt caught in process_claude_stream - propagating")
            raise
        except Exception as e:
            self.stats['errors'] += 1
            error_msg = f"Stream processing error: {e}"
            logging.error(error_msg)

            # Write error to log file
            if self.log_file:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"ERROR: Claude Code execution failed\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"{error_msg}\n")
                    f.write(f"{'='*60}\n\n")

            raise
CONFIG_FILE = CONFIG_DIR / "cli_config.json"

# ---------------------------------------------------------------------------
# Version metadata helpers
# ---------------------------------------------------------------------------

def _read_install_config() -> Dict[str, Any]:
    """Read CLI installation metadata from disk."""
    if not INSTALL_CONFIG_PATH.exists():
        return {}
    try:
        with open(INSTALL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _write_install_config(config: Dict[str, Any]) -> None:
    """Persist CLI installation metadata without raising exceptions."""
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        INSTALL_CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding='utf-8')
    except Exception:
        pass


def initialize_cli_version_metadata() -> None:
    """Ensure installation metadata reflects the running CLI version."""
    global CURRENT_CLI_VERSION

    install_config = _read_install_config()
    stored_version = install_config.get('version')

    if stored_version != CLI_VERSION:
        install_config['version'] = CLI_VERSION
        install_config['updated_at'] = datetime.utcnow().isoformat()
        _write_install_config(install_config)
        CURRENT_CLI_VERSION = CLI_VERSION
        return

    if stored_version:
        CURRENT_CLI_VERSION = stored_version
    else:
        install_config['version'] = CLI_VERSION
        install_config['updated_at'] = datetime.utcnow().isoformat()
        _write_install_config(install_config)
        CURRENT_CLI_VERSION = CLI_VERSION


def _parse_version_components(version: Optional[str]) -> List[int]:
    """Convert a semantic version string into comparable integer components."""
    if not version:
        return []

    cleaned = str(version).strip()
    if cleaned.lower().startswith('v'):
        cleaned = cleaned[1:]

    components: List[int] = []
    for part in cleaned.split('.'):
        if not part:
            continue
        digits = ''.join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        try:
            components.append(int(digits))
        except ValueError:
            break

    return components


def is_version_newer(latest: Optional[str], current: Optional[str]) -> bool:
    """Return True if latest is a newer semantic version than current."""
    latest_parts = _parse_version_components(latest)
    current_parts = _parse_version_components(current)

    if not latest_parts:
        return False

    length = max(len(latest_parts), len(current_parts))
    latest_parts.extend([0] * (length - len(latest_parts)))
    current_parts.extend([0] * (length - len(current_parts)))

    return latest_parts > current_parts


def check_for_cli_update(server_url: Optional[str] = None) -> None:
    """Query backend for the latest CLI version and cache update status."""
    global UPDATE_AVAILABLE_VERSION, UPDATE_CHECK_ERROR, UPDATE_CHECK_COMPLETED

    if UPDATE_CHECK_COMPLETED:
        return

    server = server_url or resolve_server(None)
    if not server:
        UPDATE_CHECK_COMPLETED = True
        return

    try:
        response = requests.get(
            f"{server}{VERSION_CHECK_ENDPOINT}",
            timeout=VERSION_CHECK_TIMEOUT
        )
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError:
            data = {}

        latest = (data.get('latest_version') or data.get('version') or '').strip()

        if latest and is_version_newer(latest, CLI_VERSION):
            UPDATE_AVAILABLE_VERSION = latest
            UPDATE_CHECK_ERROR = None
        else:
            UPDATE_AVAILABLE_VERSION = None
            UPDATE_CHECK_ERROR = None
    except requests.exceptions.RequestException as exc:
        UPDATE_AVAILABLE_VERSION = None
        message = str(exc)
        if server:
            message = f"{server}: {message}"
        if len(message) > 160:
            message = message[:157] + '...'
        UPDATE_CHECK_ERROR = message
    finally:
        UPDATE_CHECK_COMPLETED = True


# Conda configuration - will be initialized on startup
CONDA_BASE_PATH = None
CONDA_ENV_NAME = "air"

def detect_and_configure_conda():
    """Detect conda installation and configure paths"""
    global CONDA_BASE_PATH

    # Method 1: Try to get from conda command (most reliable)
    try:
        result = subprocess.run(
            ['conda', 'info', '--base'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            conda_base = result.stdout.strip()
            if Path(conda_base).exists():
                CONDA_BASE_PATH = conda_base
                return CONDA_BASE_PATH
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    # Method 2: Try common conda locations
    common_paths = [
        '/root/miniconda3',
        '/root/anaconda3',
        Path.home() / 'miniconda3',
        Path.home() / 'anaconda3',
        '/opt/conda',
        '/usr/local/miniconda3',
        '/usr/local/anaconda3'
    ]

    for path in common_paths:
        path = Path(path)
        if path.exists() and (path / 'bin' / 'conda').exists():
            CONDA_BASE_PATH = str(path)
            return CONDA_BASE_PATH

    # Method 3: Try to find conda in PATH
    try:
        result = subprocess.run(
            ['which', 'conda'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            conda_bin = Path(result.stdout.strip())
            if conda_bin.exists():
                # conda is usually in {base}/bin/conda or {base}/condabin/conda
                conda_base = conda_bin.parent.parent
                if (conda_base / 'bin' / 'conda').exists():
                    CONDA_BASE_PATH = str(conda_base)
                    return CONDA_BASE_PATH
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass

    # If all methods fail, use default
    CONDA_BASE_PATH = '/root/miniconda3'
    return CONDA_BASE_PATH

def get_conda_activate_command(env_name=None):
    """Get the command to activate conda environment"""
    # Load from config if not specified
    if env_name is None:
        conda_config = get_conda_config()
        env_name = conda_config.get('env_name', CONDA_ENV_NAME or 'air')

    # Get base path from config or detect
    conda_config = get_conda_config()
    base_path = conda_config.get('base_path', CONDA_BASE_PATH)

    if base_path is None:
        base_path = detect_and_configure_conda()

    # Use conda shell hook for proper activation
    return f"eval \"$('{base_path}/bin/conda' 'shell.bash' 'hook')\" && conda activate {env_name}"

DEFAULT_EP_PATTERNS = [
    '*.txt',
    '*.ipynb',
    '*.json',
    '*.tex',
    '*.log',
    'eval.py',
    'requirements.txt',
    'arguments.py',
]

DEFAULT_EP_FILE_CONTENT = "['*.txt','*.ipynb','*.json','*.tex','*.log',\"eval.py\",'requirements.txt','arguments.py']\n"

PATTERN_CONTAINER_TYPES = (set, builtins.list, tuple)

# Enhanced task status with more details
task_status = {
    'connected': False,
    'task_id': None,
    'status': 'initializing',
    'current_cycle': 0,
    'max_cycles': 50,
    'current_stage': 'Initializing',
    'events': [],  # Store all events (no limit for monitoring)
    'broadcasts': deque(maxlen=5),  # Keep last 5 broadcasts
    'claude_logs': deque(maxlen=100),  # Store Claude Code execution logs
    'error': None,
    'error_code': None,
    'error_details': None,
    'start_time': None,
    'heartbeat_ok': True,
    'heartbeat_next': None,
    'can_pause': False,
    'can_resume': False,
    'token_count': 0,
    'findings_count': 0,
    'query': None,
    'terminated_by_server': False,  # Flag for server-initiated termination
    'termination_message': None,
    'cuda_device': None,
    'total_llm_tokens': 0,
    'total_prompt_tokens': 0,
    'total_completion_tokens': 0,
    'abstract': None,
    'server': None,
    'token': None,
    'dashboard_url': None,
    'completion_summary': None,
}


def _derive_backend_api_url(server: Optional[str]) -> Optional[str]:
    if not server:
        return None
    server = server.rstrip('/')
    if server.endswith('/api'):
        return server
    return f"{server}/api"


def _is_placeholder_url(url: Optional[str]) -> bool:
    if not url:
        return True
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.hostname and parsed.hostname in PLACEHOLDER_HOSTS:
        return True
    query = parse_qs(parsed.query)
    for api_url in query.get('api', []):
        try:
            api_parsed = urlparse(api_url)
        except Exception:
            continue
        if api_parsed.hostname and api_parsed.hostname in PLACEHOLDER_HOSTS:
            return True
    return False


def compute_dashboard_url(
    task_id: Optional[str] = None,
    prompt: Optional[str] = None,
    token: Optional[str] = None,
    server: Optional[str] = None,
) -> Optional[str]:
    task_id = task_id or task_status.get('task_id')
    token = token or task_status.get('token')
    server = server or task_status.get('server')
    prompt = 'DeepScientist'
    backend_api = _derive_backend_api_url(server)
    if not (task_id and backend_api and token):
        return None
    params = {
        'taskId': task_id,
        'prompt': prompt,
        'api': backend_api,
        'token': token,
    }
    return f"{RESEARCH_STUDIO_URL}/dashboard?{urlencode(params)}"


def update_dashboard_url(candidate: Optional[str] = None) -> None:
    selected = candidate
    if selected and _is_placeholder_url(selected):
        selected = None
    if not selected:
        selected = compute_dashboard_url()
    if not selected:
        return

    previous = task_status.get('dashboard_url')
    if previous == selected:
        return

    task_status['dashboard_url'] = selected
    task_status['events'].append({
        'timestamp': datetime.now().isoformat(),
        'type': 'activity',
        'title': f'ResearchStudio available: {selected}'
    })
    task_status['claude_logs'].append({
        'timestamp': datetime.now().isoformat(),
        'type': 'system',
        'formatted': f'🔬 ResearchStudio View Logs: {selected}'
    })
    control_flags['force_ui_update'] = True
    clamp_event_scroll()

# Control flags
control_flags = {
    'paused': False,
    'stop_requested': False,
    'user_input_mode': False,
    'force_ui_update': False,  # Force immediate UI update on important events
    'final_summary_ready': False,
    'should_exit': False,  # Signal for immediate exit (CTRL+C style)
}

ui_state = {
    'view': 'research',
    'event_scroll': 0,
    'impl_selected_index': 0,
    'impl_selected_id': None,
    'impl_log_scroll': 0,
}

implementation_registry: Dict[str, Dict[str, Dict[str, Any]]] = {}
implementation_lock = threading.Lock()
ui_state_lock = threading.Lock()
config_lock = threading.Lock()
keyboard_listener_started = False
keyboard_listener_lock = threading.Lock()

EVENT_DISPLAY_LIMIT = 20
IMPLEMENTATION_LOG_LINES = 40


def clamp_event_scroll() -> None:
    """Ensure event scroll offset stays within valid bounds."""
    with ui_state_lock:
        events_total = len(task_status.get('events', []))
        max_scroll = max(0, events_total - EVENT_DISPLAY_LIMIT)
        if ui_state['event_scroll'] > max_scroll:
            ui_state['event_scroll'] = max_scroll
        if ui_state['event_scroll'] < 0:
            ui_state['event_scroll'] = 0


def modify_event_scroll(delta: int) -> bool:
    """Adjust event scroll offset by delta. Returns True if changed."""
    with ui_state_lock:
        events_total = len(task_status.get('events', []))
        max_scroll = max(0, events_total - EVENT_DISPLAY_LIMIT)
        new_value = max(0, min(ui_state['event_scroll'] + delta, max_scroll))
        if new_value != ui_state['event_scroll']:
            ui_state['event_scroll'] = new_value
            return True
    return False


def reset_event_scroll() -> None:
    with ui_state_lock:
        ui_state['event_scroll'] = 0


def _persist_implementation_entry(task_id: str, idea_id: str, entry: Dict[str, Any]) -> None:
    """Persist implementation metadata to CLI config."""
    serializable = {
        'title': entry.get('title'),
        'summary': entry.get('summary'),
        'log_path': entry.get('log_path'),
        'status': entry.get('status', 'unknown'),
        'created_at': entry.get('created_at'),
        'updated_at': entry.get('updated_at'),
        'result_summary': entry.get('result_summary'),
    }
    with config_lock:
        config = load_cli_config()
        impl_map = config.setdefault('implementations', {})
        task_map = impl_map.setdefault(task_id, {})
        task_map[idea_id] = serializable
        save_cli_config(config)


def load_implementations_for_task(task_id: str) -> None:
    """Populate implementation registry from persisted config for a task."""
    config = load_cli_config()
    impl_map = config.get('implementations', {}).get(task_id, {})
    with implementation_lock:
        registry: Dict[str, Dict[str, Any]] = {}
        for idea_id, data in impl_map.items():
            registry[idea_id] = {
                'idea_id': idea_id,
                'title': data.get('title') or f'Implementation {idea_id}',
                'summary': data.get('summary') or '',
                'log_path': data.get('log_path'),
                'status': data.get('status', 'unknown'),
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
                'result_summary': data.get('result_summary') or '',
            }
        implementation_registry[task_id] = registry


def register_implementation_metadata(task_id: str, idea_id: str, *, title: str, summary: str, log_path: str) -> None:
    """Register or update implementation metadata when execution starts."""
    now = datetime.utcnow().isoformat() + 'Z'
    entry = {
        'idea_id': idea_id,
        'title': title or f'Implementation {idea_id}',
        'summary': summary or '',
        'log_path': log_path,
        'status': 'running',
        'created_at': now,
        'updated_at': now,
        'result_summary': '',
    }
    with implementation_lock:
        task_map = implementation_registry.setdefault(task_id, {})
        existing = task_map.get(idea_id, {})
        existing.update(entry)
        task_map[idea_id] = existing
    with ui_state_lock:
        ui_state['impl_selected_id'] = idea_id
        ui_state['impl_selected_index'] = 0
        ui_state['impl_log_scroll'] = 0
    _persist_implementation_entry(task_id, idea_id, entry)


def update_implementation_status(task_id: str, idea_id: str, status: str, *, result_summary: str = '') -> None:
    """Update implementation metadata when execution finishes."""
    now = datetime.utcnow().isoformat() + 'Z'
    with implementation_lock:
        task_map = implementation_registry.setdefault(task_id, {})
        entry = task_map.get(idea_id, {
            'idea_id': idea_id,
            'title': f'Implementation {idea_id}',
            'summary': '',
            'log_path': '',
            'created_at': now,
            'result_summary': '',
        })
        entry['status'] = status
        entry['updated_at'] = now
        if result_summary:
            entry['result_summary'] = result_summary
        task_map[idea_id] = entry
    _persist_implementation_entry(task_id, idea_id, entry)


def fetch_implementation_entries(task_id: str) -> List[Dict[str, Any]]:
    """Fetch implementation entries sorted by latest update."""
    with implementation_lock:
        entries = builtins.list(implementation_registry.get(task_id, {}).values())
    if not entries:
        load_implementations_for_task(task_id)
        with implementation_lock:
            entries = builtins.list(implementation_registry.get(task_id, {}).values())
    entries.sort(key=lambda e: e.get('updated_at') or e.get('created_at') or '', reverse=True)
    return entries


def ensure_impl_selection(task_id: str, entries: List[Dict[str, Any]]) -> None:
    """Sync UI selection state with available implementation entries."""
    with ui_state_lock:
        if not entries:
            ui_state['impl_selected_index'] = 0
            ui_state['impl_selected_id'] = None
            ui_state['impl_log_scroll'] = 0
            return

        selected_id = ui_state.get('impl_selected_id')
        if selected_id:
            for idx, entry in enumerate(entries):
                if entry.get('idea_id') == selected_id:
                    ui_state['impl_selected_index'] = idx
                    break
            else:
                ui_state['impl_selected_index'] = 0
                ui_state['impl_selected_id'] = entries[0].get('idea_id')
        else:
            current_idx = ui_state.get('impl_selected_index', 0)
            current_idx = max(0, min(current_idx, len(entries) - 1))
            ui_state['impl_selected_index'] = current_idx
            ui_state['impl_selected_id'] = entries[current_idx].get('idea_id')

        if ui_state['impl_log_scroll'] < 0:
            ui_state['impl_log_scroll'] = 0


def read_implementation_log_lines(entry: Dict[str, Any]) -> List[str]:
    """Read all log lines for an implementation entry."""
    log_path = entry.get('log_path')
    if not log_path:
        return []
    path = Path(log_path)
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return []
    return content.splitlines()


def get_implementation_log_tail(entry: Dict[str, Any], *, scroll_offset: int = 0, max_lines: int = IMPLEMENTATION_LOG_LINES) -> List[str]:
    """Return the tail of the implementation log respecting scroll offset."""
    lines = read_implementation_log_lines(entry)
    if not lines:
        return []
    total = len(lines)
    effective_scroll = max(0, min(scroll_offset, max(0, total - max_lines)))
    end = total - effective_scroll
    start = max(0, end - max_lines)
    return lines[start:end]


def change_impl_selection(task_id: str, delta: int) -> bool:
    entries = fetch_implementation_entries(task_id)
    if not entries:
        return False
    ensure_impl_selection(task_id, entries)
    with ui_state_lock:
        idx = ui_state.get('impl_selected_index', 0)
        new_idx = max(0, min(idx + delta, len(entries) - 1))
        if new_idx != idx:
            ui_state['impl_selected_index'] = new_idx
            ui_state['impl_selected_id'] = entries[new_idx].get('idea_id')
            ui_state['impl_log_scroll'] = 0
            return True
    return False


def modify_impl_log_scroll(task_id: str, delta: int) -> bool:
    entries = fetch_implementation_entries(task_id)
    if not entries:
        return False
    ensure_impl_selection(task_id, entries)
    with ui_state_lock:
        idx = ui_state.get('impl_selected_index', 0)
        idx = max(0, min(idx, len(entries) - 1))
        entry = entries[idx]
        lines = read_implementation_log_lines(entry)
        if not lines:
            return False
        max_scroll = max(0, len(lines) - IMPLEMENTATION_LOG_LINES)
        new_value = max(0, min(ui_state.get('impl_log_scroll', 0) + delta, max_scroll))
        if new_value != ui_state.get('impl_log_scroll', 0):
            ui_state['impl_log_scroll'] = new_value
            return True
    return False


def switch_view(target: str) -> bool:
    if target not in {'research', 'implementer'}:
        return False
    changed = False
    if target == 'implementer' and task_status.get('task_id'):
        entries = fetch_implementation_entries(task_status['task_id'])
        ensure_impl_selection(task_status['task_id'], entries)
    with ui_state_lock:
        if ui_state['view'] != target:
            ui_state['view'] = target
            if target == 'research':
                ui_state['event_scroll'] = min(ui_state['event_scroll'], max(0, len(task_status.get('events', [])) - EVENT_DISPLAY_LIMIT))
            else:
                ui_state['impl_log_scroll'] = 0
            changed = True
    return changed


def ensure_keyboard_listener() -> None:
    global keyboard_listener_started
    if keyboard_listener_started:
        return
    with keyboard_listener_lock:
        if keyboard_listener_started:
            return
        listener = threading.Thread(target=input_listener_thread, daemon=True)
        listener.start()
        keyboard_listener_started = True


def parse_token_estimate(summary: str, content: str) -> int:
    """Extract the gitingest token estimate, falling back to content length."""
    if summary:
        match = re.search(r"Estimated tokens:\s*([0-9]+(?:\.[0-9]+)?)\s*([kKmM]?)", summary)
        if match:
            value = float(match.group(1))
            suffix = match.group(2).lower()
            if suffix == 'k':
                value *= 1_000
            elif suffix == 'm':
                value *= 1_000_000
            return int(round(value))
    return len(content) // 4


def reset_runtime_state():
    """Reset runtime state for interactive commands."""
    task_status.update({
        'connected': False,
        'task_id': None,
        'status': 'initializing',
        'current_cycle': 0,
        'max_cycles': 50,
        'current_stage': 'Initializing',
        'error': None,
        'error_code': None,
        'error_details': None,
        'start_time': None,
        'heartbeat_ok': True,
        'heartbeat_next': None,
        'can_pause': False,
        'can_resume': False,
        'token_count': 0,
        'findings_count': 0,
        'query': None,
        'cuda_device': None,
        'total_llm_tokens': 0,
        'total_prompt_tokens': 0,
        'total_completion_tokens': 0,
        'abstract': None,
        'server': None,
        'token': None,
        'dashboard_url': None,
        'completion_summary': None,
    })
    task_status['events'] = []
    task_status['broadcasts'] = deque(maxlen=5)
    control_flags.update({
        'paused': False,
        'stop_requested': False,
        'user_input_mode': False,
        'force_ui_update': False,
        'final_summary_ready': False,
    })
    with ui_state_lock:
        ui_state.update({
            'view': 'research',
            'event_scroll': 0,
            'impl_selected_index': 0,
            'impl_selected_id': None,
            'impl_log_scroll': 0,
        })
    with implementation_lock:
        implementation_registry.clear()



def create_prompt_session(prompt_name):
    """Initialise an interactive prompt session with history and clipboard support."""
    if not PROMPT_TOOLKIT_AVAILABLE or PromptSession is None or KeyBindings is None:
        return None

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    history_file = CONFIG_DIR / "cli_history.txt"
    history = None
    try:
        history_file.touch(exist_ok=True)
        history = FileHistory(str(history_file))
    except Exception:
        history = None

    key_bindings = KeyBindings()
    clipboard = None
    if PyperclipClipboard is not None:
        with suppress(Exception):
            clipboard = PyperclipClipboard()

    if clipboard is not None:
        @key_bindings.add('c-v')
        def _(event):  # type: ignore
            with suppress(Exception):
                data = clipboard.get_data()
                if data:
                    event.current_buffer.paste_clipboard_data(data)

    session_kwargs = {
        'history': history,
        'key_bindings': key_bindings,
        'enable_history_search': True,
    }

    if clipboard is not None:
        session_kwargs['clipboard'] = clipboard

    if history is not None and AutoSuggestFromHistory is not None:
        session_kwargs['auto_suggest'] = AutoSuggestFromHistory()

    session = PromptSession(**session_kwargs)
    setattr(session, 'deepscientist_clipboard_enabled', clipboard is not None)
    return session



def format_param_name(param):
    """Return a human friendly parameter name for error messaging."""
    if not param:
        return None
    name = getattr(param, 'name', None)
    if not name:
        return None
    if getattr(param, 'opts', None):
        return name if name.startswith('-') else f"--{name}"
    return name



def display_command_error(command_name, exc):
    """Render actionable error information for interactive commands."""
    console.print(f"[red]✗ {exc.format_message()}[/red]")

    hints = []
    param = getattr(exc, 'param', None)
    param_name = format_param_name(param)

    if isinstance(exc, click.MissingParameter):
        if param_name:
            hints.append(f"Provide the required value for `{param_name}`.")
        else:
            hints.append("Provide the missing required argument for this command.")
    elif isinstance(exc, click.BadParameter):
        if param_name:
            hints.append(f"Check the value supplied for `{param_name}`; it is not valid.")
    elif isinstance(exc, click.NoSuchOption):
        if param_name:
            hints.append(f"The option `{param_name}` is not recognised by `{command_name}`.")
    elif isinstance(exc, click.UsageError):
        hints.append("Review the command usage; one or more arguments look incorrect.")

    command_specific_hints = {
        'login': "Example: `login --token YOUR_TOKEN`.",
        'submit': "Example: `submit ./path/to/project`.",
    }

    specific_hint = command_specific_hints.get(command_name)
    if specific_hint and specific_hint not in hints:
        hints.append(specific_hint)

    usage_hint = f"Run `{command_name} --help` for full usage details."
    if usage_hint not in hints:
        hints.append(usage_hint)

    for hint in hints:
        console.print(f"[yellow]Hint:[/yellow] {hint}")


def run_interactive_shell(ctx):
    """Launch interactive UI when no subcommand is provided."""
    global IN_INTERACTIVE_SESSION, _INTERACTIVE_BANNER_SHOWN
    IN_INTERACTIVE_SESSION = True
    _INTERACTIVE_BANNER_SHOWN = False

    print_banner()

    prompt_name = ctx.command_path or 'deepscientist-cli'
    prompt_session = create_prompt_session(prompt_name)

    while True:
        try:
            if prompt_session:
                prompt_message = ANSI(f"\x1b[1;32m{prompt_name}> \x1b[0m") if ANSI else f"{prompt_name}> "
                user_input = prompt_session.prompt(prompt_message)
            else:
                user_input = console.input(f"[bold green]{prompt_name}> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]👋 Exiting interactive mode[/cyan]")
            break

        command_line = user_input.strip()
        if not command_line:
            continue

        lowered = command_line.lower()
        if lowered in {"exit", "quit", "q"}:
            console.print("[cyan]👋 Exiting interactive mode[/cyan]")
            break
        if lowered in {"help", "?"}:
            help_text = cli.get_help(ctx)
            console.print(Panel(help_text, title="[bold cyan]Available Commands[/bold cyan]", border_style="cyan"))
            continue
        if lowered == "clear":
            console.clear()
            _INTERACTIVE_BANNER_SHOWN = False
            print_banner()
            continue

        try:
            parts = shlex.split(command_line)
        except ValueError as exc:
            console.print(f"[red]Unable to parse command:[/red] {exc}")
            console.print("[yellow]Hint:[/yellow] Wrap arguments containing spaces in quotes.")
            continue

        command_name = parts[0]
        command = cli.get_command(ctx, command_name)
        if command is None:
            console.print(f"[red]Unknown command:[/red] {command_name}")
            available_commands = cli.list_commands(ctx)
            suggestion = difflib.get_close_matches(command_name, available_commands, n=1)
            if suggestion:
                console.print(f"[yellow]Hint:[/yellow] Did you mean `{suggestion[0]}`?")
            else:
                console.print("[yellow]Hint:[/yellow] Type `help` to list available commands.")
            continue

        reset_runtime_state()
        try:
            command.main(args=parts[1:], prog_name=f"{prompt_name} {command_name}", standalone_mode=False)
        except click.ClickException as exc:
            display_command_error(command_name, exc)
        except SystemExit as exc:
            code = getattr(exc, 'code', 0)
            if code not in (None, 0):
                console.print(f"[red]Command exited with status {code}[/red]")
        except Exception as exc:
            console.print(f"[red]Unexpected error:[/red] {exc}")
            console.print_exception(show_locals=False)
        finally:
            console.print("")

    IN_INTERACTIVE_SESSION = False
    _INTERACTIVE_BANNER_SHOWN = False


def normalize_server_url(url):
    if not url:
        return url
    cleaned = url.strip()
    if cleaned.endswith('/'):
        cleaned = cleaned.rstrip('/')
    if not cleaned.startswith('http://') and not cleaned.startswith('https://'):
        cleaned = f"http://{cleaned}"
    return cleaned


def extract_error_message(response):
    try:
        data = response.json()
    except ValueError:
        data = None

    if isinstance(data, dict):
        for key in ('error', 'message', 'detail', 'description'):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    text = (response.text or '').strip()
    if text:
        return text[:200]
    return None


def load_cli_config():
    """Load CLI configuration, merging install metadata with CLI overrides."""
    config: Dict[str, Any] = {}

    install_config = _read_install_config()
    if isinstance(install_config, dict):
        config.update(install_config)

    if CONFIG_FILE.exists():
        try:
            file_config = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
            if isinstance(file_config, dict):
                config.update(file_config)
        except Exception:
            pass

    return config


def save_cli_config(config):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding='utf-8')

    # Synchronise selected configuration keys with the install config
    install_config = _read_install_config()
    if not isinstance(install_config, dict):
        install_config = {}

    keys_to_sync = [
        'token',
        'servers',
        'default_server',
        'workspace_dir',
        'conda',
        'validation_frequency',
        'baseline_upload',
        'last_login_at',
        'claude_code_max_retries',
        'test_sh_max_retries',
        'reconnection_attempts',
        'heartbeat_interval'
    ]

    for key in keys_to_sync:
        if key in config and config[key] is not None:
            install_config[key] = config[key]
        else:
            install_config.pop(key, None)

    _write_install_config(install_config)


def is_token_configured(server: Optional[str] = None) -> bool:
    """Determine whether an authentication token is stored for the given server."""
    config = load_cli_config()
    if server:
        stored = get_saved_token(server)
        if stored:
            return True
    token = config.get('token')
    return bool(token)


def is_conda_configured(config: Optional[Dict[str, Any]] = None) -> bool:
    """Check whether a conda environment has been configured."""
    if config is None:
        config = load_cli_config()
    conda_cfg = config.get('conda')
    if not isinstance(conda_cfg, dict):
        return False
    env_name = conda_cfg.get('env_name')
    base_path = conda_cfg.get('base_path')
    return bool(env_name) and bool(base_path)


def is_validation_frequency_configured(config: Optional[Dict[str, Any]] = None) -> bool:
    """Return True if validation frequency is explicitly configured."""
    if config is None:
        config = load_cli_config()
    value = config.get('validation_frequency')
    return value in {'high', 'medium', 'low', 'auto'}


def prompt_for_token_configuration(ctx: click.Context, server: str) -> None:
    """Guide the user through interactive token configuration."""
    console.print("\n[bold cyan]🔐 Authentication Setup Required[/bold cyan]")
    console.print("─" * 60)
    console.print("[yellow]A saved API token was not found in your install config.[/yellow]")
    console.print("[dim]We'll walk through the login flow to capture your token now.[/dim]\n")

    normalized_server = normalize_server_url(server)
    try:
        ctx.invoke(login, token=None, server=normalized_server, interactive=True)
    except SystemExit:
        # Allow user to cancel without crashing the CLI shell
        console.print("[yellow]⚠ Login flow was cancelled or failed. You can rerun 'deepscientist login' later.[/yellow]")


def prompt_for_conda_configuration() -> None:
    """Prompt the user to select and store a conda environment."""
    console.print("\n[bold cyan]🐍 Conda Environment Configuration[/bold cyan]")
    console.print("─" * 60)
    console.print("[yellow]No conda environment is stored yet.[/yellow]")
    console.print("[dim]Select the environment DeepScientist should use for implementations.[/dim]\n")

    base_path = detect_and_configure_conda()
    env_names: List[str] = []

    # Attempt to retrieve available environments via conda
    try:
        result = subprocess.run(
            ['conda', 'env', 'list', '--json'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            env_paths = data.get('envs') or []
            for path in env_paths:
                try:
                    name = Path(path).name
                    if name:
                        env_names.append(name)
                except Exception:
                    continue
    except Exception:
        pass

    env_name = None
    if env_names:
        unique_envs = sorted(set(env_names))
        console.print("[bold]Available Conda environments:[/bold]")
        for idx, name in enumerate(unique_envs, start=1):
            console.print(f"  [cyan]{idx}.[/cyan] {name}")
        console.print()

        while True:
            choice = click.prompt("Enter a number or environment name", default=unique_envs[0]).strip()
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(unique_envs):
                    env_name = unique_envs[index]
                    break
            elif choice in unique_envs:
                env_name = choice
                break
            else:
                console.print("[yellow]Invalid selection. Please choose a listed environment or enter its name.[/yellow]")
    else:
        console.print("[yellow]Unable to list environments automatically.[/yellow]")
        console.print("[dim]Enter the name of the conda environment you plan to use (e.g., 'air').[/dim]")
        env_name = click.prompt("Conda environment name", default='air').strip()

    if not env_name:
        console.print("[yellow]⚠ No environment selected. Conda configuration remains unset.[/yellow]")
        return

    save_conda_config(env_name, base_path)

    global CONDA_ENV_NAME
    CONDA_ENV_NAME = env_name


def prompt_for_validation_frequency_initial() -> str:
    """Prompt the user to select and persist a validation cadence."""
    console.print("\n[bold cyan]🧪 Validation Frequency Configuration[/bold cyan]")
    console.print("─" * 60)
    console.print("[yellow]Choose how often DeepScientist validates research progress.[/yellow]")
    console.print("[dim]This preference is saved in your install config and can be changed later.[/dim]\n")

    options = {
        '1': 'high',
        'high': 'high',
        '2': 'medium',
        'medium': 'medium',
        '3': 'low',
        'low': 'low',
        '4': 'auto',
        'auto': 'auto'
    }

    console.print("  [bold green]1.[/bold green] High   - Validate every cycle (most thorough)")
    console.print("  [bold cyan]2.[/bold cyan] Medium - Validate every 3 cycles (balanced)")
    console.print("  [bold yellow]3.[/bold yellow] Low    - Validate every 10 cycles (faster)")
    console.print("  [bold blue]4.[/bold blue] Auto   - Let DeepScientist decide dynamically\n")

    selection = None
    while selection is None:
        choice = click.prompt("Your choice", default='2').strip().lower()
        selection = options.get(choice)
        if selection is None:
            console.print("[yellow]Please enter 1-4 or one of: high, medium, low, auto.[/yellow]")

    save_validation_frequency(selection)
    return selection


def ensure_initial_configuration(ctx: click.Context, default_server: str) -> None:
    """Ensure critical CLI settings are captured before entering interactive mode."""
    config = load_cli_config()

    if not is_token_configured(default_server):
        prompt_for_token_configuration(ctx, default_server)
        config = load_cli_config()

    if not is_conda_configured(config):
        prompt_for_conda_configuration()
        config = load_cli_config()

    if not is_validation_frequency_configured(config):
        prompt_for_validation_frequency_initial()


def get_conda_config():
    """Get conda configuration from config file"""
    config = load_cli_config()
    return config.get('conda', {
        'env_name': 'air',  # default
        'base_path': CONDA_BASE_PATH
    })


def get_baseline_upload_default():
    """Get default baseline upload option from config file
    Returns: 'Y', 'H', or None (ask)
    """
    config = load_cli_config()
    return config.get('baseline_upload', None)


def save_baseline_upload_default(option: str):
    """Save baseline upload default option to config file
    Args:
        option: 'Y', 'H', or 'ask'
    """
    config = load_cli_config()
    normalized = option.upper()
    if normalized in ['Y', 'H', 'ASK']:
        config['baseline_upload'] = option.upper() if option.upper() != 'ASK' else None
    else:
        config['baseline_upload'] = None
    save_cli_config(config)
    console.print(f"[green]✓ Baseline upload default saved: {option.upper()}[/green]")


def get_validation_frequency():
    """Get validation frequency setting from config file
    Returns: 'high', 'medium', 'low', 'auto', or None (default: medium)
    """
    config = load_cli_config()
    return config.get('validation_frequency', 'medium')


def save_validation_frequency(frequency: str):
    """Save validation frequency setting to config file
    Args:
        frequency: 'high', 'medium', 'low', or 'auto'
    """
    config = load_cli_config()
    if frequency.lower() in ['high', 'medium', 'low', 'auto']:
        config['validation_frequency'] = frequency.lower()
    else:
        config['validation_frequency'] = 'medium'  # default
    save_cli_config(config)
    console.print(f"[green]✓ Validation frequency saved: {frequency.lower()}[/green]")


def save_conda_config(env_name: str, base_path: str = None):
    """Save conda configuration to config file"""
    config = load_cli_config()
    if 'conda' not in config:
        config['conda'] = {}

    config['conda']['env_name'] = env_name
    if base_path:
        config['conda']['base_path'] = base_path
    else:
        config['conda']['base_path'] = CONDA_BASE_PATH

    save_cli_config(config)
    console.print(f"[green]✓ Conda config saved: env={env_name}, base={config['conda']['base_path']}[/green]")


def get_saved_token(server):
    config = load_cli_config()
    servers = config.get('servers', {})
    normalized = normalize_server_url(server)
    if normalized and normalized in servers:
        token = servers[normalized].get('token')
        if token:
            return token
    return config.get('token')


def record_login_timestamp(server, login_at: Optional[str]):
    """Persist last login timestamp for server and global config"""
    if not login_at:
        return
    config = load_cli_config()
    normalized = normalize_server_url(server)
    servers = config.setdefault('servers', {})
    if normalized:
        server_entry = servers.setdefault(normalized, {})
        server_entry['last_login_at'] = login_at
    config['last_login_at'] = login_at
    save_cli_config(config)


def store_token(token, server, login_at: Optional[str] = None):
    config = load_cli_config()
    servers = config.setdefault('servers', {})
    normalized = normalize_server_url(server)
    server_entry = servers.setdefault(normalized, {})
    server_entry['token'] = token
    server_entry['saved_at'] = datetime.utcnow().isoformat() + 'Z'
    if login_at is None:
        login_at = datetime.utcnow().isoformat() + 'Z'
    server_entry['last_login_at'] = login_at
    config['default_server'] = normalized
    config['token'] = token
    config['last_login_at'] = login_at
    save_cli_config(config)
    return CONFIG_FILE


def resolve_server(server_option):
    if server_option:
        return normalize_server_url(server_option)
    env_server = os.environ.get('DEEPSCIENTIST_SERVER')
    if env_server:
        return normalize_server_url(env_server)
    config = load_cli_config()
    stored = config.get('default_server')
    if stored:
        return normalize_server_url(stored)
    return normalize_server_url(SERVER_URL)


def get_retry_config():
    """Get retry configuration with default values.

    Returns:
        dict: {
            'claude_code_max_retries': int (default: 2),
            'test_sh_max_retries': int (default: 1)
        }
    """
    config = load_cli_config()
    return {
        'claude_code_max_retries': config.get('claude_code_max_retries', 2),
        'test_sh_max_retries': config.get('test_sh_max_retries', 1)
    }


def get_connection_config():
    """Get WebSocket connection configuration from config file

    Returns:
        dict: {
            'reconnection_attempts': int (default: 400, ~1 hour with 10s max delay),
            'heartbeat_interval': int (default: 1800, 30 minutes in seconds)
        }
    """
    config = load_cli_config()
    return {
        'reconnection_attempts': config.get('reconnection_attempts', 400),
        'heartbeat_interval': config.get('heartbeat_interval', 1800)
    }


def resolve_token(token_option, server):
    if token_option:
        return token_option, '--token option'
    env_token = os.environ.get('DEEPSCIENTIST_TOKEN')
    if env_token:
        return env_token, 'DEEPSCIENTIST_TOKEN env'
    saved = get_saved_token(server)
    if saved:
        return saved, 'saved config'
    return None, None


def mask_token(token):
    """Mask token for display - show first 8 and last 4 characters"""
    if not token:
        return "Not configured"
    if len(token) <= 12:
        return token[:4] + "***" + token[-2:]
    return token[:8] + "..." + token[-4:]


def ensure_server_available(server):
    """Fail fast if the backend cannot be reached"""
    try:
        response = requests.get(f"{server}/health", timeout=5)
    except requests.exceptions.RequestException as exc:
        console.print(f"[red]✗ Unable to reach backend:[/red] {server}")
        console.print(f"[yellow]Hint:[/yellow] Check network connectivity or verify the service is running. ({exc})")
        sys.exit(1)

    if response.status_code >= 500:
        console.print(f"[red]✗ Backend responded with {response.status_code} {response.reason or ''}[/red]")
        console.print("[yellow]Hint:[/yellow] Remote service is unavailable. Try again once the network/API recover.")
        sys.exit(1)

    if response.status_code >= 400:
        console.print(f"[red]✗ Backend health check failed (HTTP {response.status_code})[/red]")
        console.print("[yellow]Hint:[/yellow] Confirm the API endpoint is correct and accessible.")
        sys.exit(1)


def _render_broadcasts_panel(broadcasts: List[Dict[str, Any]]) -> Optional[Panel]:
    """Render all broadcasts inside a single panel."""
    if not broadcasts:
        return None

    severity_rank = {'info': 0, 'warning': 1, 'error': 2}
    level_styles = {
        'error': {
            'border': "bold red",
            'title': "bold red",
            'message': "bold red",
            'timestamp': "red dim",
            'emoji': "🚨",
        },
        'warning': {
            'border': "bold yellow",
            'title': "bold yellow",
            'message': "yellow",
            'timestamp': "yellow dim",
            'emoji': "⚠️",
        },
        'info': {
            'border': "bold cyan",
            'title': "bold cyan",
            'message': "cyan",
            'timestamp': "cyan dim",
            'emoji': "📢",
        },
    }

    # Determine overall styling by highest severity present
    chosen_level = max(broadcasts, key=lambda b: severity_rank.get((b or {}).get('level', 'info'), 0)).get('level', 'info')
    chosen_level = chosen_level if chosen_level in level_styles else 'info'
    style = level_styles[chosen_level]

    rows: List[Group] = []
    for idx, broadcast in enumerate(broadcasts, start=1):
        level = broadcast.get('level', 'info')
        level_style = level_styles.get(level, level_styles['info'])
        message = (broadcast.get('message') or '').strip() or 'No message provided'
        message = message.replace('\n', ' ')
        created_at = broadcast.get('created_at', '')
        try:
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            time_str = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            time_str = created_at or 'Unknown time'

        message_text = Text(f"{idx}. {message}", style=level_style['message'])
        timestamp_text = Text(f"Published: {time_str}", style=level_style['timestamp'])
        row = Group(message_text, Align.right(timestamp_text))
        rows.append(row)
        if idx != len(broadcasts):
            rows.append(Text("", style=""))

    panel_title = f"[{style['title']}]{style['emoji']} System Broadcasts[/{style['title']}]"
    body = Group(*rows)
    return Panel(body, title=panel_title, border_style=style['border'], box=DOUBLE)


def display_broadcasts(broadcasts: List[Dict[str, Any]]) -> None:
    """Display one or more broadcast messages."""
    if not broadcasts:
        return

    panel = _render_broadcasts_panel(broadcasts)
    if panel:
        try:
            console.print(panel)
            console.print()
        except Exception as exc:
            logging.debug("Failed to render broadcasts: %s", exc)


def update_broadcast_cache(
    broadcasts: Optional[List[Dict[str, Any]]],
    *,
    display: bool = False,
    force: bool = False
) -> None:
    """Update cached broadcasts and optionally display them."""
    global LATEST_BROADCASTS
    broadcasts = broadcasts or []
    is_new = broadcasts != LATEST_BROADCASTS
    LATEST_BROADCASTS = list(broadcasts)
    if display and broadcasts and (force or is_new):
        display_broadcasts(broadcasts)


def print_broadcast_section():
    """Render cached broadcast messages beneath the banner/help panels."""
    if not LATEST_BROADCASTS:
        return

    panel = _render_broadcasts_panel(LATEST_BROADCASTS)
    if panel:
        console.print()
        console.print(panel)
        console.print()


def format_timestamp(ts: Optional[str]) -> str:
    if not ts:
        return "N/A"
    try:
        value = ts.replace('Z', '+00:00') if ts.endswith('Z') else ts
        dt = datetime.fromisoformat(value)
        return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
    except Exception:
        return ts


def show_connection_summary(server: Optional[str], login_result: Optional[Dict[str, Any]], token: Optional[str] = None):
    if not server or server == 'Not configured':
        console.print("[yellow]⚠ No default server configured. Run `deepscientist login` to set one.[/yellow]")
        return

    summary_table = Table(show_header=False, box=None, padding=(0, 1))
    summary_table.add_column("Field", style="dim cyan")
    summary_table.add_column("Value", style="white")
    summary_table.add_row("Server", server)

    data: Optional[Dict[str, Any]] = None
    if login_result and login_result.get('success'):
        data = login_result.get('data') or LAST_LOGIN_PAYLOAD
    elif LAST_LOGIN_PAYLOAD:
        data = LAST_LOGIN_PAYLOAD

    if login_result and login_result.get('supported') is False:
        summary_table.add_row("Login Endpoint", "[yellow]Legacy (login_cli unavailable)[/yellow]")

    if data and isinstance(data, dict):
        user_info = data.get('user') or {}
        username = user_info.get('username') or 'Unknown'
        role = user_info.get('role') or user_info.get('user_type') or 'user'
        summary_table.add_row("User", f"{username} ({role})")
        login_at = data.get('login_at') or user_info.get('last_login')
        if login_at:
            summary_table.add_row("Current Login", format_timestamp(login_at))
        prev_login = data.get('previous_login_at')
        if prev_login:
            summary_table.add_row("Previous Login", format_timestamp(prev_login))
        token_exp = data.get('token_expires_at')
        if token_exp:
            summary_table.add_row("Token Expires", format_timestamp(token_exp))
        summary_table.add_row("Status", "[green]✓ Authenticated[/green]")
    else:
        if login_result and login_result.get('error'):
            status = f"[red]Error[/red]: {login_result.get('error')}"
        else:
            status = "[yellow]Auto login skipped[/yellow]"
            if token:
                status += " (token available)"
            else:
                status += " (no stored token)"
        summary_table.add_row("Status", status)

    panel = Panel(summary_table, title="[bold cyan]Connection Summary[/bold cyan]", border_style="cyan")
    console.print(panel)


def fetch_broadcasts(server: str, token: str) -> List[Dict[str, Any]]:
    """Fetch broadcast messages from server."""
    if not server or not token:
        return []

    try:
        response = requests.get(
            f"{server}/api/broadcasts",
            headers={'Authorization': f'Bearer {token}'},
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            return data.get('broadcasts', []) or []
    except Exception:
        return []
    return []


def fetch_and_display_broadcasts(server, token):
    """Fetch and display broadcast messages from the server."""
    if not server or not token:
        return []

    try:
        response = requests.get(
            f"{server}/api/broadcasts",
            headers={'Authorization': f'Bearer {token}'},
            timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            broadcasts = data.get('broadcasts', []) or []
            update_broadcast_cache(broadcasts, display=True)
            return broadcasts
    except Exception as exc:
        logging.debug("Failed to fetch broadcasts: %s", exc)
    return []


def perform_cli_login(server: str, token: str, *, reason: str = 'auto', save_token: bool = False,
                      silent: bool = False):
    """Call backend CLI login endpoint, update local config, and return response metadata."""
    global LAST_LOGIN_PAYLOAD
    if not server or not token:
        return None

    LAST_LOGIN_PAYLOAD = None

    payload = {
        'token': token,
        'cli_version': CLI_VERSION,
        'client_name': 'deepscientist-cli',
        'login_reason': reason,
    }

    try:
        response = requests.post(
            f"{server}/api/auth/login_cli",
            json=payload,
            timeout=10
        )
    except requests.exceptions.RequestException as exc:
        if not silent:
            console.print(f"[red]✗ Login handshake failed:[/red] {exc}")
        return {'supported': True, 'success': False, 'error': str(exc)}

    if response.status_code in (404, 405):
        # Older backend without CLI login endpoint
        return {'supported': False, 'success': False}

    if response.status_code in (401, 403):
        detail = extract_error_message(response)
        console.print(f"[red]✗ Token verification failed (HTTP {response.status_code}).[/red]")
        if detail:
            console.print(f"[yellow]Details:[/yellow] {detail}")
        return {'supported': True, 'success': False, 'status': response.status_code, 'error': detail}

    if response.status_code >= 400:
        detail = extract_error_message(response)
        if not silent:
            console.print(f"[red]✗ Server rejected CLI login (HTTP {response.status_code}).[/red]")
            if detail:
                console.print(f"[yellow]Details:[/yellow] {detail}")
        return {'supported': True, 'success': False, 'status': response.status_code, 'error': detail}

    try:
        data = response.json()
    except ValueError:
        if not silent:
            console.print("[red]✗ CLI login returned invalid JSON.[/red]")
        return {'supported': True, 'success': False, 'error': 'Invalid JSON response'}

    LAST_LOGIN_PAYLOAD = data

    login_at = data.get('login_at')
    broadcasts = data.get('broadcasts', [])
    config_path = None
    if save_token:
        config_path = store_token(token, server, login_at=login_at)
    else:
        record_login_timestamp(server, login_at)

    update_broadcast_cache(broadcasts, display=False)

    return {
        'supported': True,
        'success': True,
        'data': data,
        'config_path': config_path
    }


def attempt_cli_auto_login(server: str):
    """Automatically login using stored token and show broadcasts if available."""
    if not server:
        return None

    token, _token_source = resolve_token(None, server)
    if not token:
        return None
    token = token.strip()

    result = perform_cli_login(server, token, reason='auto', save_token=False, silent=True)
    if not result:
        return None

    if result.get('supported') is False:
        broadcasts = fetch_broadcasts(server, token)
        update_broadcast_cache(broadcasts, display=False)
        return result

    if result.get('success'):
        return result

    return result


def legacy_login_flow(server: str, token: str, interactive: bool):
    """Fallback login flow for legacy backends without /api/auth/login_cli."""
    global LAST_LOGIN_PAYLOAD
    with console.status("[bold cyan]Verifying token with backend...[/bold cyan]"):
        try:
            response = requests.post(
                f"{server}/api/auth/verify",
                json={'token': token},
                timeout=10
            )

            if response.status_code == 404:
                response = requests.get(
                    f"{server}/api/tasks",
                    headers={'Authorization': f'Bearer {token}'},
                    timeout=10
                )
        except requests.exceptions.Timeout:
            console.print("[red]✗ Error:[/red] Backend request timed out after 10s")
            console.print("[yellow]Hint:[/yellow] Check network connectivity and server load, then retry.")
            sys.exit(1)
        except requests.exceptions.ConnectionError as exc:
            console.print("[red]✗ Error:[/red] Unable to connect to backend server")
            console.print(f"[yellow]Details:[/yellow] {exc}")
            console.print("[yellow]Hint:[/yellow] Ensure the server address is reachable and the API service is running.")
            sys.exit(1)
        except requests.exceptions.RequestException as exc:
            console.print(f"[red]✗ Error:[/red] Request failed: {exc}")
            sys.exit(1)

    status = response.status_code
    error_detail = extract_error_message(response)

    if status == 200:
        config_path = store_token(token, server)
        broadcasts = fetch_broadcasts(server, token)
        update_broadcast_cache(broadcasts, display=True)

        try:
            data = response.json()
        except ValueError:
            data = {}
        LAST_LOGIN_PAYLOAD = data if isinstance(data, dict) else None

        if interactive:
            console.print("\n[bold green]🎉 Login Successful![/bold green]")
            console.print("─" * 60)
            console.print(f"[green]✓[/green] Configuration saved to: {config_path}")
            console.print("[green]✓[/green] You're now ready to use DeepScientist CLI\n")

            try:
                if 'user' in data:
                    user_info = data['user']
                    username = user_info.get('username', 'Unknown')
                    user_type = user_info.get('user_type', 'normal')
                    api_verified = user_info.get('api_verified', False)
                    email = user_info.get('email', 'Not set')

                    console.print("[bold]👤 User Information:[/bold]")
                    console.print(f"  Username: {username}")
                    console.print(f"  User Type: {user_type}")
                    console.print(f"  Email: {email}")
                    if api_verified:
                        console.print(f"  API Status: [green]✓ Verified[/green]")
                    else:
                        console.print(f"  API Status: [yellow]⚠ Not configured[/yellow]")

                    try:
                        tasks_response = requests.get(
                            f"{server}/api/tasks",
                            headers={'Authorization': f'Bearer {token}'},
                            timeout=5
                        )
                        if tasks_response.status_code == 200:
                            tasks_data = tasks_response.json()
                            tasks = tasks_data.get('tasks', [])
                            console.print(f"  Existing Tasks: {len(tasks)}")
                    except Exception:
                        console.print(f"  Existing Tasks: [dim]Unable to fetch[/dim]")

                    console.print("\n[bold]🚀 Next Steps:[/bold]")
                    console.print("  • [cyan]deepscientist whoami[/cyan] - Show your profile")
                    console.print("  • [cyan]deepscientist list[/cyan] - List your tasks")

                    if not api_verified:
                        console.print("  • [cyan]deepscientist verify-api[/cyan] - Configure API settings")

                    console.print("  • [cyan]deepscientist submit /path/to/repo[/cyan] - Start research")
                    console.print("  • [cyan]deepscientist --help[/cyan] - See all commands")

                    if user_type == 'admin':
                        console.print("\n[dim]Admin tips: Use [cyan]deepscientist list --all[/cyan] to see all users' tasks[/dim]")

                    console.print(f"\n[dim]Connected to: {server}[/dim]")
                else:
                    tasks = response.json().get('tasks', [])
                    console.print("[bold]👤 User Information:[/bold]")
                    console.print("  Status: [green]✓ Authenticated[/green]")
                    console.print(f"  Existing Tasks: {len(tasks)}")
            except Exception:
                console.print("[yellow]⚠ Could not fetch user details[/yellow]")
                console.print("[dim]You're logged in, but some user details are unavailable.[/dim]")

                console.print("\n[bold]🚀 Next Steps:[/bold]")
                console.print("  • [cyan]deepscientist whoami[/cyan] - Show your profile")
                console.print("  • [cyan]deepscientist list[/cyan] - List your tasks")
                console.print("  • [cyan]deepscientist submit /path/to/repo[/cyan] - Start research")
        else:
            console.print(f"\n[green]✓[/green] Token verified and saved to {config_path}")
            console.print("[green]✓[/green] You can now use CLI commands without passing --token\n")

        return True

    if status in (401, 403):
        message = error_detail or 'Unauthorized'
        console.print(f"[red]✗ Token verification failed:[/red] {message}")
        console.print("[yellow]Hint:[/yellow] Re-check the token value or generate a new one from the dashboard.")
        sys.exit(1)

    if 400 <= status < 500:
        console.print(f"[red]✗ Request rejected (HTTP {status})[/red]")
        if error_detail:
            console.print(f"[yellow]Details:[/yellow] {error_detail}")
        console.print("[yellow]Hint:[/yellow] Confirm the server URL and API path are correct.")
        sys.exit(1)

    if 500 <= status < 600:
        if status == 502:
            console.print("[red]✗ Backend unavailable (HTTP 502 Bad Gateway)[/red]")
            console.print(f"[yellow]Hint:[/yellow] {server} received the request but could not reach the API service. Check backend health or proxy configuration.")
        elif status == 503:
            console.print("[red]✗ Backend unavailable (HTTP 503 Service Unavailable)[/red]")
            console.print("[yellow]Hint:[/yellow] The API reported it is temporarily unavailable. Confirm the service is running.")
        elif status == 504:
            console.print("[red]✗ Backend unavailable (HTTP 504 Gateway Timeout)[/red]")
            console.print("[yellow]Hint:[/yellow] The reverse proxy timed out before the API responded. Investigate server performance or network latency.")
        else:
            console.print(f"[red]✗ Server error (HTTP {status})[/red]")
        if error_detail:
            console.print(f"[yellow]Details:[/yellow] {error_detail}")
        sys.exit(1)

    console.print("[red]✗ Unexpected response from backend[/red]")
    sys.exit(1)


def fetch_tasks(server, token):
    """Retrieve tasks for the authenticated user"""
    try:
        response = requests.get(
            f"{server}/api/tasks",
            headers={'Authorization': f'Bearer {token}'},
            timeout=10
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error while fetching tasks: {exc}")

    if response.status_code != 200:
        try:
            detail = response.json().get('error', '')
        except Exception:
            detail = response.text or ''
        raise RuntimeError(f"HTTP {response.status_code} {detail}".strip())

    data = response.json()
    return data.get('tasks', [])


def format_task_row(task):
    task_id = task.get('task_id', 'unknown')
    short_id = f"{task_id[:12]}…" if len(task_id) > 13 else task_id
    status = task.get('status', 'unknown')
    tokens = task.get('token_count')
    try:
        tokens = int(tokens or 0)
    except Exception:
        tokens = 0
    created = task.get('created_at') or ''
    created_display = created[:19] if created else ''
    return f"{short_id:<16}  {status:<10}  {tokens:>8,} tokens  {created_display}"


def choose_action_interactively():
    """
    Interactive action selector using arrow keys.
    Returns: '1' for monitor task, '2' for dashboard, '3' or None for exit
    """
    actions = [
        {'key': '1', 'label': 'Monitor a specific task', 'style': 'green'},
        {'key': '2', 'label': 'Start live dashboard (watch all tasks)', 'style': 'yellow'},
        {'key': '3', 'label': 'Exit', 'style': 'dim'}
    ]

    if PROMPT_TOOLKIT_AVAILABLE and KeyBindings is not None:
        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.layout import Layout
            from prompt_toolkit.layout.containers import HSplit, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.styles import Style as PTStyle
        except Exception:
            pass
        else:
            selected = {'index': 0}

            def body_text():
                lines = []
                for idx, action in enumerate(actions):
                    pointer = '➜ ' if idx == selected['index'] else '  '
                    style = 'class:pointer' if idx == selected['index'] else 'class:item'
                    lines.append((style, f"{pointer}{action['key']}. {action['label']}\n"))
                return lines

            def instructions():
                return [
                    ('class:instruction', 'Use ↑/↓ to choose, Enter to confirm, Esc to exit.')
                ]

            kb = KeyBindings()

            @kb.add('up')
            def _(event):
                selected['index'] = (selected['index'] - 1) % len(actions)
                event.app.invalidate()

            @kb.add('down')
            def _(event):
                selected['index'] = (selected['index'] + 1) % len(actions)
                event.app.invalidate()

            @kb.add('enter')
            def _(event):
                event.app.exit(result=actions[selected['index']]['key'])

            @kb.add('escape')
            @kb.add('c-c')
            def _(event):
                event.app.exit(result='3')  # Exit on Esc

            # Allow direct number input
            @kb.add('1')
            def _(event):
                event.app.exit(result='1')

            @kb.add('2')
            def _(event):
                event.app.exit(result='2')

            @kb.add('3')
            def _(event):
                event.app.exit(result='3')

            body_window = Window(content=FormattedTextControl(body_text), always_hide_cursor=True)
            instruction_window = Window(height=1, content=FormattedTextControl(instructions), always_hide_cursor=True)

            layout = Layout(HSplit([
                Window(height=1, content=FormattedTextControl(lambda: [('class:title', 'Choose an action:')]), always_hide_cursor=True),
                Window(height=1, char=' '),
                body_window,
                Window(height=1, char=' '),
                instruction_window
            ]))

            style = PTStyle.from_dict({
                'pointer': 'bold cyan',
                'item': '',
                'instruction': 'italic #64748b',
                'title': 'bold cyan'
            })

            app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False)
            try:
                return app.run()
            except Exception:
                # Fallback to simple selector
                pass

    # Fallback: simple text prompt
    console.print("\n[bold cyan]Choose an action:[/bold cyan]")
    for action in actions:
        console.print(f"  [{action['style']}]{action['key']}.[/{action['style']}] {action['label']}")

    choice = Prompt.ask(
        "\n[cyan]Your choice[/cyan]",
        choices=['1', '2', '3'],
        default='1'
    )
    return choice


def choose_task_interactively(tasks):
    if not tasks:
        return None

    if PROMPT_TOOLKIT_AVAILABLE and KeyBindings is not None:
        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.layout import Layout
            from prompt_toolkit.layout.containers import HSplit, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.styles import Style as PTStyle
        except Exception:
            pass
        else:
            selected = {'index': 0}

            def body_text():
                lines = []
                for idx, task in enumerate(tasks):
                    pointer = '➜ ' if idx == selected['index'] else '  '
                    style = 'class:pointer' if idx == selected['index'] else 'class:item'
                    lines.append((style, f"{pointer}{format_task_row(task)}\n"))
                return lines

            def instructions():
                return [
                    ('class:instruction', 'Use ↑/↓ to choose a task, Enter to view details, Esc to cancel.')
                ]

            kb = KeyBindings()

            @kb.add('up')
            def _(event):
                selected['index'] = (selected['index'] - 1) % len(tasks)
                event.app.invalidate()

            @kb.add('down')
            def _(event):
                selected['index'] = (selected['index'] + 1) % len(tasks)
                event.app.invalidate()

            @kb.add('enter')
            def _(event):
                event.app.exit(result=tasks[selected['index']]['task_id'])

            @kb.add('escape')
            @kb.add('c-c')
            def _(event):
                event.app.exit(result=None)

            body_window = Window(content=FormattedTextControl(body_text), always_hide_cursor=True)
            instruction_window = Window(height=1, content=FormattedTextControl(instructions), always_hide_cursor=True)

            layout = Layout(HSplit([
                Window(height=1, content=FormattedTextControl(lambda: [('class:title', 'Select a task to view details')]), always_hide_cursor=True),
                Window(height=1, char=' '),
                body_window,
                Window(height=1, char=' '),
                instruction_window
            ]))

            style = PTStyle.from_dict({
                'pointer': 'bold cyan',
                'item': '',
                'instruction': 'italic #64748b',
                'title': 'bold'
            })

            app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False)
            try:
                return app.run()
            except Exception:
                # Fallback to simple selector
                pass

    # Fallback: numeric selection via input
    console.print("[bold cyan]\n📋 Available Tasks[/bold cyan]")
    for idx, task in enumerate(tasks, 1):
        console.print(f"  [cyan]{idx:>2}.[/cyan] {format_task_row(task)}")

    while True:
        choice = click.prompt("Enter task number (or blank to cancel)", default='', show_default=False)
        if not choice:
            return None
        try:
            index = int(choice) - 1
        except ValueError:
            console.print("[yellow]Please enter a valid number.[/yellow]")
            continue
        if 0 <= index < len(tasks):
            return tasks[index]['task_id']
        console.print("[yellow]Number out of range, try again.[/yellow]")


def shorten_text(value, limit=160):
    """Create a short preview for long text blocks."""
    if not value:
        return None

    text = str(value).strip()
    if len(text) <= limit:
        return text

    return text[: limit - 1].rstrip() + '…'


def format_brief_duration(seconds: Optional[float]) -> str:
    """Format duration in a compact human-readable form."""
    if seconds is None:
        return 'N/A'
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return 'N/A'

    if seconds < 0:
        seconds = 0

    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return ' '.join(parts)


def build_completion_summary_panel(summary: Dict[str, Any], status: str) -> Panel:
    """Construct a Rich panel displaying final task statistics."""
    display_status = status.upper() if status else 'COMPLETED'

    info_table = Table(show_header=False, box=ROUNDED)
    info_table.add_row('Task ID', summary.get('task_id') or '—')
    info_table.add_row('Status', display_status)
    query_preview = shorten_text(summary.get('query'), 120)
    if query_preview:
        info_table.add_row('Focus', query_preview)

    runtime_table = Table(show_header=False, box=ROUNDED)
    runtime_table.add_row('Runtime', format_brief_duration(summary.get('runtime_seconds')))
    runtime_table.add_row('Started', summary.get('started_at') or '—')
    runtime_table.add_row('Completed', summary.get('completed_at') or '—')

    findings_table = Table(show_header=False, box=ROUNDED)
    findings_table.add_row('Total findings', str(summary.get('total_findings') or 0))
    findings_table.add_row('Implemented', str(summary.get('implemented_findings') or 0))
    findings_table.add_row('✅ Success', str(summary.get('successful_implementations') or 0))
    findings_table.add_row('❌ Failed', str(summary.get('failed_implementations') or 0))

    tokens_table = Table(show_header=False, box=ROUNDED)
    token_count = summary.get('token_count')
    total_llm = summary.get('total_llm_tokens')
    prompt_tokens = summary.get('total_prompt_tokens')
    completion_tokens = summary.get('total_completion_tokens')

    if token_count is not None:
        tokens_table.add_row('Submission tokens', f"{int(token_count or 0):,}")
    if prompt_tokens is not None or completion_tokens is not None:
        tokens_table.add_row('LLM prompt', f"{int(prompt_tokens or 0):,}")
        tokens_table.add_row('LLM completion', f"{int(completion_tokens or 0):,}")
    if total_llm is not None:
        tokens_table.add_row('LLM total', f"{int(total_llm or 0):,}")

    sections = [info_table, findings_table, runtime_table]
    if any(row for row in tokens_table.rows):
        sections.append(tokens_table)

    validation_frequency = summary.get('validation_frequency')
    if validation_frequency:
        info_table.add_row('Validation cadence', str(validation_frequency))

    content = Group(*sections)
    panel = Panel(
        content,
        title='Task Summary',
        border_style='cyan',
        box=ROUNDED
    )
    return panel


def format_finding_scores(finding):
    """Format the score components for display."""
    parts = []

    combined = finding.get('combined_score')
    if combined is None:
        combined = finding.get('score')
    if combined is not None:
        parts.append(f"Σ {combined}")

    for label, key in (('U', 'utility_score'), ('Q', 'quality_score'), ('E', 'exploration_score')):
        value = finding.get(key)
        if value is not None:
            parts.append(f"{label} {value}")

    return " • ".join(parts) if parts else "n/a"


def get_finding_display_title(finding):
    """Return a user-friendly title for the finding."""
    return (
        finding.get('display_title')
        or finding.get('title')
        or (finding.get('idea_id') and f"Idea {finding['idea_id']}")
        or "Untitled finding"
    )


def format_metrics_for_display(finding):
    """Return a prettified metrics block if available."""
    metrics_preview = finding.get('metrics_preview')
    if metrics_preview:
        return metrics_preview

    raw_metrics = finding.get('real_metrics')
    if not raw_metrics:
        return None

    if isinstance(raw_metrics, (dict, list)):
        try:
            return json.dumps(raw_metrics, ensure_ascii=False, indent=2)
        except Exception:
            return str(raw_metrics)

    if isinstance(raw_metrics, str):
        try:
            parsed = json.loads(raw_metrics)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return raw_metrics

    return str(raw_metrics)


def create_finding_panel(finding):
    """Build a Rich panel summarising a finding."""
    display_title = get_finding_display_title(finding)
    idea_id = finding.get('idea_id')
    scores_display = format_finding_scores(finding)

    status = (finding.get('implementation_success') or 'pending').lower()
    status_emoji = {
        'success': '✅',
        'failed': '❌',
        'pending': '⏳',
    }.get(status, 'ℹ️')

    created_at_display = finding.get('created_at_display')
    created_relative = finding.get('created_at_relative')
    timestamp_line = None
    if created_at_display:
        relative = f" ({created_relative})" if created_relative else ""
        timestamp_line = f"[bold]Logged:[/bold] {created_at_display}{relative}"

    sections = [
        f"[bold]Phase:[/bold] {finding.get('phase') or 'unknown'}",
        f"[bold]Scores:[/bold] {scores_display}",
        f"[bold]Implementation:[/bold] {status_emoji} {status}",
    ]

    if timestamp_line:
        sections.append(timestamp_line)

    code_path = finding.get('code_path')
    if code_path:
        sections.append(f"[bold]Artifacts:[/bold] {code_path}")

    summary = finding.get('summary_preview')
    if summary:
        sections.append(f"[bold]Summary:[/bold]\n{summary}")

    abstract = finding.get('abstract')
    if abstract:
        sections.append(f"[bold]Abstract:[/bold]\n{abstract}")

    methodology = finding.get('methodology')
    if methodology:
        sections.append(f"[bold]Methodology:[/bold]\n{methodology}")

    implementation_log = finding.get('implementation_log')
    if implementation_log:
        sections.append(f"[bold]Implementation Log:[/bold]\n{shorten_text(implementation_log, limit=2000)}")

    metrics_block = format_metrics_for_display(finding)
    if metrics_block:
        sections.append(f"[bold]Real Metrics:[/bold]\n{metrics_block}")

    result_text = finding.get('result')
    if result_text:
        sections.append(f"[bold]Result Notes:[/bold]\n{result_text}")

    description_text = finding.get('description')
    if description_text and description_text != abstract:
        sections.append(f"[bold]Notes:[/bold]\n{description_text}")

    full_idea = finding.get('full_idea')
    if full_idea:
        sections.append(f"[bold]Full Idea (preview):[/bold]\n{shorten_text(full_idea, limit=800)}")

    panel_title = f"[bold green]💡 {display_title}[/bold green]"
    if idea_id:
        panel_title += f" ({idea_id})"

    return Panel(
        "\n\n".join(sections),
        title=panel_title,
        border_style="green",
        box=ROUNDED
    )


def view_findings_interactively(findings):
    """Launch an interactive viewer for findings, returning True if shown."""
    if not findings:
        return False

    if not PROMPT_TOOLKIT_AVAILABLE or KeyBindings is None:
        return False

    try:
        from prompt_toolkit.application import Application
        from prompt_toolkit.layout import Layout, HSplit, VSplit
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style as PTStyle
    except Exception:
        return False

    selected = {'index': 0}
    total = len(findings)

    def current():
        if selected['index'] < 0:
            selected['index'] = 0
        if selected['index'] >= total:
            selected['index'] = total - 1
        return findings[selected['index']]

    def list_content():
        lines = []
        for idx, finding in enumerate(findings):
            pointer = '➜ ' if idx == selected['index'] else '  '
            style = 'class:list-selected' if idx == selected['index'] else 'class:list-item'
            title = get_finding_display_title(finding)
            lines.append((style, f"{pointer}{title}\n"))

            meta_parts = []
            phase = finding.get('phase') or 'unknown'
            if phase:
                meta_parts.append(f"Phase: {phase}")

            scores = format_finding_scores(finding)
            if scores:
                meta_parts.append(f"Scores: {scores}")

            when = finding.get('created_at_relative') or finding.get('created_at_display')
            if when:
                meta_parts.append(str(when))

            if meta_parts:
                meta_style = 'class:list-meta-selected' if idx == selected['index'] else 'class:list-meta'
                lines.append((meta_style, f"    {' • '.join(meta_parts)}\n"))

            summary = finding.get('summary_preview') or shorten_text(
                finding.get('abstract') or finding.get('description'),
                limit=90
            )
            if summary:
                lines.append(('class:list-summary', f"    {summary}\n"))

            lines.append(('', "\n"))

        if not lines:
            return [('class:list-empty', 'No findings available\n')]
        return lines

    def detail_content():
        finding = current()
        lines = []

        title = get_finding_display_title(finding)
        lines.append(('class:detail-title', f"{title}\n"))

        idea_id = finding.get('idea_id')
        if idea_id:
            lines.append(('class:detail-subtitle', f"Idea ID: {idea_id}\n"))

        timestamp = finding.get('created_at_display')
        relative = finding.get('created_at_relative')
        if timestamp or relative:
            rel = f" ({relative})" if relative else ''
            lines.append(('class:detail-meta', f"Logged: {timestamp or ''}{rel}\n"))

        phase = finding.get('phase') or 'unknown'
        lines.append(('class:detail-meta', f"Phase: {phase}\n"))

        scores = format_finding_scores(finding)
        lines.append(('class:detail-meta', f"Scores: {scores}\n"))

        status = (finding.get('implementation_success') or 'pending').lower()
        status_label = status.capitalize()
        status_emoji = {
            'success': '✅',
            'failed': '❌',
            'pending': '⏳',
        }.get(status, 'ℹ️')
        lines.append(('class:detail-meta', f"Implementation: {status_emoji} {status_label}\n"))

        code_path = finding.get('code_path')
        if code_path:
            lines.append(('class:detail-meta', f"Artifacts: {code_path}\n"))

        lines.append(('', '\n'))

        def add_section(title_text, value, limit=None):
            if not value:
                return
            body = str(value).strip()
            if limit and len(body) > limit:
                body = body[: limit - 1].rstrip() + '…'
            lines.append(('class:section-heading', f"{title_text}\n"))
            lines.append(('class:section-body', f"{body}\n\n"))

        summary = finding.get('summary_preview')
        if summary:
            add_section('Summary', summary)

        add_section('Abstract', finding.get('abstract'))
        add_section('Methodology', finding.get('methodology'))
        add_section('Result Notes', finding.get('result'))

        description_text = finding.get('description')
        if description_text and description_text != finding.get('abstract'):
            add_section('Notes', description_text)

        add_section('Implementation Log', finding.get('implementation_log'), limit=2000)

        metrics_block = format_metrics_for_display(finding)
        if metrics_block:
            add_section('Real Metrics', metrics_block)

        full_idea = finding.get('full_idea')
        if full_idea:
            add_section('Full Idea (preview)', full_idea, limit=1200)

        if not lines:
            lines.append(('class:detail-body', 'No data for this finding\n'))

        return lines

    def footer_content():
        instructions = '↑/↓ navigate • PgUp/PgDn jump • Home/End • Q or Esc to exit'
        return [('class:footer', instructions)]

    list_window = Window(
        content=FormattedTextControl(list_content),
        width=60,
        wrap_lines=True,
        always_hide_cursor=True
    )

    detail_window = Window(
        content=FormattedTextControl(detail_content),
        wrap_lines=True,
        always_hide_cursor=True
    )

    layout = Layout(HSplit([
        Window(height=1, content=FormattedTextControl(lambda: [('class:title', '🔬 Findings Viewer')])),
        Window(height=1, char=' '),
        VSplit([
            list_window,
            Window(width=1, char='│', style='class:divider'),
            detail_window,
        ]),
        Window(height=1, char=' '),
        Window(height=1, content=FormattedTextControl(footer_content), always_hide_cursor=True)
    ]))

    kb = KeyBindings()

    @kb.add('up')
    def _(event):
        selected['index'] = (selected['index'] - 1) % total
        event.app.invalidate()

    @kb.add('down')
    def _(event):
        selected['index'] = (selected['index'] + 1) % total
        event.app.invalidate()

    @kb.add('pageup')
    def _(event):
        selected['index'] = max(0, selected['index'] - 5)
        event.app.invalidate()

    @kb.add('pagedown')
    def _(event):
        selected['index'] = min(total - 1, selected['index'] + 5)
        event.app.invalidate()

    @kb.add('home')
    def _(event):
        selected['index'] = 0
        event.app.invalidate()

    @kb.add('end')
    def _(event):
        selected['index'] = total - 1
        event.app.invalidate()

    @kb.add('q')
    def _(event):
        event.app.exit(result=True)

    @kb.add('escape')
    def _(event):
        event.app.exit(result=True)

    @kb.add('c-c')
    def _(event):
        event.app.exit(result=True)

    @kb.add('enter')
    def _(event):
        event.app.exit(result=True)

    style = PTStyle.from_dict({
        'title': 'bold cyan',
        'divider': '#334155',
        'list-item': '',
        'list-selected': 'reverse bold',
        'list-meta': 'italic #94a3b8',
        'list-meta-selected': 'reverse italic #94a3b8',
        'list-summary': '#94a3b8',
        'list-empty': '#64748b',
        'detail-title': 'bold underline',
        'detail-subtitle': 'italic #a855f7',
        'detail-meta': '#22d3ee',
        'section-heading': 'bold #f97316',
        'section-body': '',
        'detail-body': '',
        'footer': 'italic #64748b',
    })

    app = Application(layout=layout, key_bindings=kb, style=style, full_screen=True)

    try:
        app.run()
        return True
    except Exception as exc:
        logging.debug('Interactive findings viewer failed: %s', exc)
        return False


def check_server_connection(server, token):
    """Check server connection and login status, display results"""
    console.print("[bold cyan]🔌 Connecting to Server...[/bold cyan]")
    console.print("─" * 60)

    # Test server connectivity
    try:
        response = requests.get(f"{server}/health", timeout=5)
        if response.status_code == 200:
            console.print(f"[green]✓ Server online:[/green] {server}")
        else:
            console.print(f"[yellow]⚠ Server responded but health check failed (HTTP {response.status_code})[/yellow]")
    except requests.exceptions.ConnectionError:
        console.print(f"[red]✗ Cannot connect to server:[/red] {server}")
        console.print("[yellow]Hint:[/yellow] Check if the backend service is running")
        return False
    except requests.exceptions.Timeout:
        console.print(f"[yellow]⚠ Server connection timed out:[/yellow] {server}")
        return False
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {str(e)}")
        return False

    # Test authentication if token is available
    if token:
        try:
            response = requests.post(
                f"{server}/api/auth/verify",
                headers={'Authorization': f'Bearer {token}'},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                user_info = data.get('user', {})
                username = user_info.get('username', 'Unknown')
                role = user_info.get('role', 'user')
                console.print(f"[green]✓ Authenticated as:[/green] {username} ({role})")

                # Fetch and display broadcasts
                fetch_and_display_broadcasts(server, token)
                return True
            else:
                console.print(f"[yellow]⚠ Token verification failed (HTTP {response.status_code})[/yellow]")
                console.print("[yellow]Hint:[/yellow] Run `login` command to update your token")
                return False
        except Exception as e:
            console.print(f"[yellow]⚠ Authentication check failed:[/yellow] {str(e)}")
            return False
    else:
        console.print("[yellow]⚠ No authentication token found[/yellow]")
        console.print("[yellow]Hint:[/yellow] Run `login --token <YOUR_TOKEN>` to authenticate")
        return False

    console.print()



def print_banner():
    """Print beautiful banner with system info"""
    global _INTERACTIVE_BANNER_SHOWN
    if IN_INTERACTIVE_SESSION:
        if _INTERACTIVE_BANNER_SHOWN:
            return
        _INTERACTIVE_BANNER_SHOWN = True
    ascii_logo = [
        "██████╗ ███████╗███████╗██████╗ ███████╗ ██████╗██╗███████╗███╗   ██╗████████╗██╗███████╗████████╗",
        "██╔══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔════╝██║██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝╚══██╔══╝",
        "██║  ██║█████╗  █████╗  ██████╔╝███████╗██║     ██║█████╗  ██╔██╗ ██║   ██║   ██║███████╗   ██║   ",
        "██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ╚════██║██║     ██║██╔══╝  ██║╚██╗██║   ██║   ██║╚════██║   ██║   ",
        "██████╔╝███████╗███████╗██║     ███████║╚██████╗██║███████╗██║ ╚████║   ██║   ██║███████║   ██║   ",
        "╚═════╝ ╚══════╝╚══════╝╚═╝     ╚══════╝ ╚═════╝╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚══════╝   ╚═╝   ",
    ]

    version = CURRENT_CLI_VERSION or CLI_VERSION

    title = "🔬 AI Research Platform - Enhanced CLI"
    subtitle = f"Version {version} | Crafting breakthroughs with intelligence"

    console.print("\n")
    for line in ascii_logo:
        console.print(f"  {line}", style="bold cyan")
    console.print(f"\n  {title}", style="bold yellow")
    console.print(f"  {subtitle}", style="dim white")

    # System info
    config = load_cli_config()
    server = config.get('default_server', 'Not configured')
    last_login = config.get('last_login_at', 'Never')
    token = config.get('token')

    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column("Key", style="dim cyan")
    info_table.add_column("Value", style="white")
    info_table.add_row("📡 Default Server", server)
    info_table.add_row("🔑 Config File", str(CONFIG_FILE))
    info_table.add_row("⏰ Last Login", last_login[:19] if last_login != 'Never' else 'Never')

    console.print("\n")
    console.print(info_table)
    console.print("\n")

    if UPDATE_AVAILABLE_VERSION:
        update_panel = Panel.fit(
            "🚀 New version available: "
            f"[bold]{UPDATE_AVAILABLE_VERSION}[/bold] (current {version}).\n"
            "Run `install_cli.sh` to upgrade and unlock the latest features.",
            title="[yellow]Update Available[/yellow]",
            border_style="yellow"
        )
        console.print(update_panel)
        console.print("")
    elif UPDATE_CHECK_ERROR:
        console.print(
            f"[yellow]⚠ Update check failed:[/yellow] {UPDATE_CHECK_ERROR}"
        )
        console.print("")

    # Check server connection and auth status
    if server and server != 'Not configured':
        server_normalized = normalize_server_url(server)
        check_server_connection(server_normalized, token)

    # Quick tips
    tips = Panel(
        "[bold]Quick Tips:[/bold]\n"
        "  • [cyan]login[/cyan]     - Verify and save API token\n"
        "  • [cyan]submit[/cyan]    - Submit new research task\n"
        "  • [cyan]list[/cyan]      - View your tasks\n"
        "  • [cyan]stop[/cyan]      - Stop running task(s) (--task <id> or --all)\n"
        "  • [cyan]findings[/cyan]  - View research findings\n"
        "  • [cyan]config[/cyan]    - View and manage CLI configuration\n",
        title="[bold green]Available Commands[/bold green]",
        border_style="green"
    )
    console.print(tips)
    console.print("\n")


def validate_codebase(codebase_path: str) -> dict:
    """Validate codebase with enhanced progress display"""
    path = Path(codebase_path)

    console.print("\n[bold cyan]🔍 Validating Codebase[/bold cyan]")
    console.print("─" * 60)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:

        # Check path
        task1 = progress.add_task("[cyan]Checking path...", total=None)
        if not path.exists():
            console.print(f"\n[red]✗ Error:[/red] Path does not exist: {codebase_path}")
            return None
        progress.update(task1, completed=True)
        console.print("[green]  ✓[/green] Path exists")

        # Check CLAUDE.md
        task2 = progress.add_task("[cyan]Looking for CLAUDE.md...", total=None)
        claude_md = path / "CLAUDE.md"
        if not claude_md.exists():
            progress.update(task2, completed=True)
            console.print("\n[red]  ✗ Error:[/red] CLAUDE.md not found")
            console.print("[yellow]  💡 Hint:[/yellow] Create CLAUDE.md with your research goals")
            return None
        progress.update(task2, completed=True)
        console.print("[green]  ✓[/green] CLAUDE.md found")

        # Check test.sh
        task3 = progress.add_task("[cyan]Looking for test.sh...", total=None)
        test_sh = path / "test.sh"
        if not test_sh.exists():
            progress.update(task3, completed=True)
            console.print("\n[red]  ✗ Error:[/red] test.sh not found")
            console.print("[yellow]  💡 Hint:[/yellow] Create test.sh with your test commands")
            return None
        progress.update(task3, completed=True)
        console.print("[green]  ✓[/green] test.sh found")

        # Read CLAUDE.md
        task4 = progress.add_task("[cyan]Reading CLAUDE.md...", total=None)
        claude_content = claude_md.read_text(encoding='utf-8')
        progress.update(task4, completed=True)
        console.print(f"[green]  ✓[/green] CLAUDE.md read ({len(claude_content)} chars)")

        # Ensure latex.tex exists and capture content
        task5 = progress.add_task("[cyan]Reading latex.tex...", total=None)
        latex_tex = path / "latex.tex"
        if not latex_tex.exists():
            progress.update(task5, completed=True)
            console.print("\n[red]  ✗ Error:[/red] latex.tex not found")
            console.print("[yellow]  💡 Hint:[/yellow] Prepare your paper in LaTeX or Markdown (e.g., via MinerU PDF conversion) and save it as `latex.tex` in the project root.")
            return None
        latex_content = latex_tex.read_text(encoding='utf-8')
        progress.update(task5, completed=True)
        console.print(f"[green]  ✓[/green] latex.tex read ({len(latex_content)} chars)")

        # Load exclude patterns from optional .ep file
        task6 = progress.add_task("[cyan]Loading exclude patterns...", total=None)
        exclude_patterns = set()
        ep_file = path / ".ep"
        if ep_file.exists():
            try:
                raw_patterns = ast.literal_eval(ep_file.read_text(encoding='utf-8'))
                if isinstance(raw_patterns, PATTERN_CONTAINER_TYPES):
                    patterns = [str(p).strip() for p in raw_patterns if str(p).strip()]
                    if patterns:
                        exclude_patterns.update(patterns)
                        console.print(f"[green]  ✓[/green] Loaded {len(patterns)} exclude pattern(s) from .ep")
                    else:
                        console.print("[yellow]  •[/yellow] .ep file did not contain any usable patterns; continuing")
                else:
                    console.print("[yellow]  •[/yellow] .ep file did not contain a list/set of patterns; ignoring")
            except Exception as exc:
                console.print(f"[yellow]  •[/yellow] Failed to parse .ep: {exc}")
        else:
            try:
                ep_file.write_text(DEFAULT_EP_FILE_CONTENT, encoding='utf-8')
                console.print(
                    f"[cyan]  •[/cyan] No .ep file found; auto-created with exclusions: {', '.join(DEFAULT_EP_PATTERNS)}"
                )
            except Exception as exc:
                console.print(f"[yellow]  •[/yellow] Failed to create default .ep on disk: {exc}")
                console.print(
                    f"[yellow]  •[/yellow] Using built-in exclusions in memory: {', '.join(DEFAULT_EP_PATTERNS)}"
                )
            exclude_patterns.update(DEFAULT_EP_PATTERNS)
            console.print(f"[green]  ✓[/green] Loaded {len(DEFAULT_EP_PATTERNS)} default exclude pattern(s)")

        if 'latex.tex' not in exclude_patterns:
            exclude_patterns.add('latex.tex')
            console.print("[cyan]  •[/cyan] Automatically excluding latex.tex from ingestion")
        progress.update(task6, completed=True)

        # Analyze with gitingest
        task7 = progress.add_task("[cyan]Analyzing codebase with gitingest...", total=None)
        try:
            kwargs = {'exclude_patterns': sorted(exclude_patterns)} if exclude_patterns else {}
            summary, tree, content = ingest(source=str(path), **kwargs)
            token_count = parse_token_estimate(summary, content)
            progress.update(task7, completed=True)

            # Show token count with color
            if token_count < 10000:
                color = "green"
                status = "Excellent"
            elif token_count < 30000:
                color = "cyan"
                status = "Good"
            elif token_count < 45000:
                color = "yellow"
                status = "Acceptable"
            else:
                color = "red"
                status = "Near Limit"

            console.print(f"[{color}]  ✓[/{color}] gitingest tokens: {token_count:,} ({status})")

            if token_count >= 50000:
                console.print(f"\n[red]  ✗ Error:[/red] Token count ({token_count:,}) exceeds limit (50,000)")
                console.print("[yellow]  💡 Hint:[/yellow] Create a `.ep` file in your project root with patterns to exclude large assets (datasets, checkpoints, logs) and retry. Example: \n  ['data/*', '*.pt', '*.csv']")
                return None

            console.print("\n[bold green]✓ Validation Complete[/bold green]")

            codebase_digest = f"{tree}\n{content}" if tree else content

            return {
                'codebase_digest': codebase_digest,
                'token_count': token_count,
                'claude_md': claude_content,
                'paper_content': latex_content,
                'exclude_patterns': sorted(exclude_patterns)
            }

        except Exception as e:
            progress.update(task7, completed=True)
            console.print(f"\n[red]  ✗ Error during analysis:[/red] {e}")
            return None


def generate_enhanced_ui():
    """Generate enhanced UI layout (Claude Code style)"""
    layout = Layout()

    # Calculate dynamic header size based on broadcast messages
    # Base: 2 (borders) + 1 (title) + 3 (content rows) = 6
    # Additional: +1 per broadcast line
    broadcast_lines = len(task_status['broadcasts']) if task_status.get('broadcasts') else 0
    header_size = 6 + broadcast_lines

    # Split into header, body, footer (broadcast merged into header)
    layout.split_column(
        Layout(name="header", size=header_size),
        Layout(name="body"),
        Layout(name="footer", size=1)
    )

    # Split body into events (upper 2/3) and claude_logs (lower 1/3)
    layout["body"].split_column(
        Layout(name="events", ratio=2),
        Layout(name="claude_logs", ratio=1)
    )

    # Header with detailed task info
    if task_status['task_id']:
        elapsed = ""
        remaining = ""
        if task_status['start_time']:
            elapsed_seconds = (datetime.now() - task_status['start_time']).total_seconds()
            elapsed_hours = elapsed_seconds / 3600
            remaining_hours = 24 - elapsed_hours
            elapsed = f"{elapsed_hours:.1f}h"
            remaining = f"{remaining_hours:.1f}h" if remaining_hours > 0 else "0h"

        # Status color
        status_colors = {
            'queued': 'yellow',
            'started': 'green',
            'running': 'green',  # legacy fallback
            'paused': 'yellow',
            'completed': 'cyan',
            'failed': 'red'
        }
        status_color = status_colors.get(task_status['status'], 'white')

        # Simple single-column layout, left-aligned only
        header_content = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        header_content.add_column("Info", style="cyan", justify="left", no_wrap=False)

        # Build connection status with enhanced reconnection info
        if task_status['connected']:
            heartbeat_status = "💚 OK" if task_status['heartbeat_ok'] else "❤️ Sending..."
            heartbeat_next = task_status.get('heartbeat_next', 'N/A')
            if isinstance(heartbeat_next, datetime):
                next_str = heartbeat_next.strftime('%H:%M:%S')
            else:
                next_str = 'N/A'
            connection_text = f"[green]● Connected[/green] | Heartbeat: {heartbeat_status} (next: {next_str}) | Ctrl+C: exit | Ctrl+Q: terminate"
        else:
            # Show reconnection progress
            reconnect_count = task_status.get('reconnection_count', 0)
            disconnect_start = task_status.get('disconnection_start')
            if disconnect_start and isinstance(disconnect_start, datetime):
                elapsed = (datetime.now() - disconnect_start).total_seconds()
                elapsed_str = f"{int(elapsed)}s"
            else:
                elapsed_str = "0s"
            connection_text = f"[yellow]○ Disconnected[/yellow] | Reconnecting (attempt #{reconnect_count}, elapsed: {elapsed_str}, max: 1h) | Ctrl+C: exit"

        # Broadcast row (if any) - shown first in red
        if task_status['broadcasts']:
            broadcast_text = " | ".join([f"📢 {msg}" for msg in task_status['broadcasts']])
            header_content.add_row(f"[bold red]{broadcast_text}[/bold red]")

        # Row 1: Task ID with connection status
        header_content.add_row(f"🔬 Task: {task_status['task_id']} | {connection_text}")

        # Row 2: LLM Tokens (Input, Output, Total) + Cycle and Status
        total_prompt = task_status.get('total_prompt_tokens') or 0
        total_completion = task_status.get('total_completion_tokens') or 0
        total_llm = task_status.get('total_llm_tokens') or 0
        if total_llm == 0 and (total_prompt > 0 or total_completion > 0):
            total_llm = total_prompt + total_completion
        header_content.add_row(
            f"🤖 Tokens: In {total_prompt:,} | Out {total_completion:,} | Total {total_llm:,} | "
            f"🔄 Cycle: {task_status['current_cycle']}/{task_status['max_cycles']} | Status: [{status_color}]{task_status['status'].upper()}[/{status_color}]"
        )

        # Row 3: Dashboard link (fetched from backend)
        dashboard_url = task_status.get('dashboard_url')
        if dashboard_url:
            header_content.add_row(f"🔬 ResearchStudio View Logs: {dashboard_url}")
        else:
            header_content.add_row("[dim]🔬 ResearchStudio: Dashboard link unavailable[/dim]")

        layout["header"].update(
            Panel(
                header_content,
                title="[bold cyan]Task Information[/bold cyan]",
                border_style="cyan",
                box=ROUNDED
            )
        )
    else:
        layout["header"].update(
            Panel(
                "[bold yellow]🔌 Connecting to server...[/bold yellow]",
                border_style="yellow"
            )
        )

    # Body - Display latest events with dynamic scrolling
    # REWRITTEN: Precise calculation to ensure latest events are always visible at bottom
    if task_status['events']:
        # Get all events and sort by timestamp to ensure latest are at the end
        all_events = builtins.list(task_status['events'])

        # Sort events by timestamp (oldest to newest)
        all_events.sort(key=lambda e: e.get('timestamp', ''))

        total_events = len(all_events)

        # Calculate PRECISE available height for events panel
        # This panel gets 2/3 of the body, which is shared with claude_logs (1/3)
        try:
            terminal_height = console.size.height
            # Calculate actual header size (same as above)
            broadcast_lines = len(task_status['broadcasts']) if task_status.get('broadcasts') else 0
            actual_header_size = 6 + broadcast_lines

            # Total body height calculation:
            # Terminal height - header(dynamic) - footer(1) = body
            body_height = terminal_height - actual_header_size - 1

            # Events panel gets 2/3 of body (using ratio 2:1)
            # Due to Layout ratio behavior, we need to be conservative
            events_panel_outer_height = (body_height * 2) // 3

            # Subtract panel chrome: title(1) + borders(2) + safety margin(1) = 4
            PANEL_CHROME = 4
            events_content_height = max(3, events_panel_outer_height - PANEL_CHROME)
        except:
            events_content_height = 15  # fallback

        # Get exactly the last N events that fit (ONE event = ONE line, no wrapping)
        visible_events = all_events[-events_content_height:] if total_events > events_content_height else all_events

        # Build event display - ENFORCE single line per event
        type_styles = {
            'limitations': ('🔍', 'cyan'),
            'hypothesis': ('💡', 'yellow'),
            'plan': ('📋', 'blue'),
            'activity': ('⚙️', 'white'),
            'implementation': ('🛠️', 'green'),
            'file_update': ('📝', 'yellow'),
            'terminal_output': ('💻', 'cyan'),
            'error': ('❌', 'red'),
            'system': ('🔧', 'magenta'),
            'llm_call': ('🤖', 'bright_blue'),
        }

        event_lines = []
        for idx, event in enumerate(visible_events):
            timestamp = event.get('timestamp', '')
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    time_str = timestamp[:8]
            else:
                time_str = '        '

            title = event.get('title', '')
            event_type = event.get('type', '')

            # Remove any newline characters to enforce single line
            title = title.replace('\n', ' ').replace('\r', ' ')

            # Calculate precise max title length based on terminal width
            try:
                terminal_width = console.size.width
                # Available space: terminal_width - time(8) - icon(4) - borders(4) - padding(10)
                max_title_length = max(50, terminal_width - 26)
            except:
                max_title_length = 100

            # Truncate title to fit in one line
            if len(title) > max_title_length:
                title = title[:max_title_length-3] + "..."

            icon, default_color = type_styles.get(event_type, ('📡', 'white'))

            # Check if this is the LAST event (most recent)
            is_latest = (idx == len(visible_events) - 1)

            if is_latest:
                # Latest event: BRIGHT with background to stand out
                event_line = f"[dim]{time_str}[/dim]  [bold {default_color} on black]{icon} {title}[/bold {default_color} on black]"
            else:
                # All other events: dimmed for less visual noise
                event_line = f"[dim]{time_str}  {icon} {title}[/dim]"

            event_lines.append(event_line)

        # Join events - they will display from TOP to BOTTOM (oldest to newest)
        # The LAST line will be the NEWEST event, which is what we want
        events_text = "\n".join(event_lines)

        # Display content from top, no padding needed
        # This keeps the content at the top of the panel with reserved space below

        # Simple title showing total events
        title_text = f"Research Progress (Latest {len(visible_events)}/{total_events})"

        # Create text content and disable wrapping
        text_content = Text.from_markup(events_text, overflow="ellipsis")
        text_content.no_wrap = True  # Set no_wrap as attribute instead

        layout["events"].update(
            Panel(
                text_content,
                title=f"[bold blue]{title_text}[/bold blue]",
                border_style="blue",
                box=ROUNDED
                # Note: height parameter removed - Layout ratio controls the actual height
            )
        )
    else:
        layout["events"].update(
            Panel(
                "[dim]Waiting for events...\n\nThe research agent will emit events as it:\n"
                "  🔍 Analyzes baseline limitations\n"
                "  💡 Formulates research hypotheses\n"
                "  📋 Generates experiment plans\n"
                "  ⚙️  Executes implementations\n",
                title="[bold blue]Research Progress[/bold blue]",
                border_style="dim blue"
            )
        )

    # Claude Code Logs - Display in lower 1/3
    # REWRITTEN: Precise calculation to ensure latest output is always visible at bottom
    if task_status['claude_logs']:
        try:
            # Calculate PRECISE available height for claude logs panel
            terminal_height = console.size.height
            # Calculate actual header size (same as above)
            broadcast_lines = len(task_status['broadcasts']) if task_status.get('broadcasts') else 0
            actual_header_size = 6 + broadcast_lines

            # Total body height calculation (same as events panel)
            body_height = terminal_height - actual_header_size - 1

            # Claude logs panel gets 1/3 of body (using ratio 2:1)
            claude_panel_outer_height = body_height // 3

            # Subtract panel chrome: title(1) + borders(2) + safety margin(1) = 4
            PANEL_CHROME = 4
            claude_content_height = max(3, claude_panel_outer_height - PANEL_CHROME)
        except:
            claude_content_height = 5  # fallback

        # Get all logs
        all_logs = builtins.list(task_status['claude_logs'])
        total_logs = len(all_logs)

        # Count ACTUAL lines (some log entries may be multi-line)
        all_log_lines = []
        for log in all_logs:
            formatted_text = log['formatted']
            # Split by newlines to count actual lines
            lines = formatted_text.split('\n')
            all_log_lines.extend(lines)

        total_log_lines = len(all_log_lines)

        # Get exactly the last N lines that fit
        if total_log_lines > claude_content_height:
            visible_log_lines = all_log_lines[-claude_content_height:]
        else:
            visible_log_lines = all_log_lines

        # Join visible lines
        logs_text = "\n".join(visible_log_lines)

        # Display content from top, no padding needed
        # This keeps the content at the top of the panel with reserved space below

        # Use from_ansi to convert ANSI codes to Rich markup
        logs_content = Text.from_ansi(logs_text, overflow="ellipsis")
        logs_content.no_wrap = True  # Prevent line wrapping

        layout["claude_logs"].update(
            Panel(
                logs_content,
                title=f"[bold green]Claude Code Output (Lines: {len(visible_log_lines)}/{total_log_lines})[/bold green]",
                border_style="green",
                box=ROUNDED
                # Note: height parameter removed - Layout ratio controls the actual height
            )
        )
    else:
        layout["claude_logs"].update(
            Panel(
                "[dim]No Claude Code execution yet...[/dim]",
                title="[bold green]Claude Code Output[/bold green]",
                border_style="dim green"
            )
        )

    # Footer - Hidden, reserved for user input
    layout["footer"].update("")

    return layout


def create_websocket_client(server, token):
    """Create and configure WebSocket client with robust reconnection (configurable)"""
    # Get connection configuration from config file
    conn_config = get_connection_config()
    reconnection_attempts = conn_config['reconnection_attempts']

    # Calculate expected max reconnection time based on attempts
    # With max delay of 10s, max_time ≈ attempts * 10s / 60 = minutes
    max_minutes = (reconnection_attempts * 10) // 60

    sio = socketio.Client(
        reconnection=True,
        reconnection_attempts=reconnection_attempts,  # Configurable (default: 400 for ~1 hour)
        reconnection_delay=2,        # Start with 2 seconds
        reconnection_delay_max=10    # Cap at 10 seconds between attempts
    )

    # Create Event object for task_created signal (thread-safe)
    # This replaces busy-wait polling with efficient blocking wait
    task_created_event = threading.Event()

    # Store event in control_flags for access in submit command
    control_flags['task_created_event'] = task_created_event

    @sio.on('connect')
    def on_connect():
        """Handle connection/reconnection to server"""
        was_disconnected = not task_status.get('connected', False)
        task_status['connected'] = True
        task_status['reconnection_count'] = task_status.get('reconnection_count', 0)
        task_status['disconnection_start'] = None  # Clear disconnection timestamp

        if was_disconnected and task_status.get('reconnection_count', 0) > 0:
            # This is a reconnection
            task_status['events'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'activity',
                'title': f'✅ Reconnected to server (attempt #{task_status["reconnection_count"]})'
            })

            # Automatically rejoin task room if we have a task_id
            if task_status.get('task_id'):
                try:
                    sio.emit('join_task', {
                        'task_id': task_status['task_id'],
                        'token': token
                    })
                    task_status['events'].append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'activity',
                        'title': f'🔄 Rejoined task room: {task_status["task_id"]}'
                    })
                except Exception as e:
                    task_status['events'].append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'activity',
                        'title': f'⚠️  Failed to rejoin task room: {str(e)}'
                    })
        else:
            # Initial connection
            task_status['events'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'activity',
                'title': '✅ Connected to server'
            })

        clamp_event_scroll()
        control_flags['force_ui_update'] = True

    @sio.on('disconnect')
    def on_disconnect():
        """Handle disconnection from server"""
        task_status['connected'] = False
        task_status['reconnection_count'] = task_status.get('reconnection_count', 0) + 1
        task_status['disconnection_start'] = datetime.now()

        task_status['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'activity',
            'title': '⚠️  Disconnected from server - attempting to reconnect (will retry for up to 1 hour)...'
        })
        clamp_event_scroll()
        control_flags['force_ui_update'] = True

    @sio.on('task_created')
    def on_task_created(data):
        # Debug: Log received event
        import sys
        print(f"[DEBUG] Received task_created event: {data.get('task_id', 'unknown')}", file=sys.stderr)

        task_status['task_id'] = data['task_id']
        task_status['status'] = data.get('status', 'queued')
        task_status['start_time'] = datetime.now()
        task_status['can_pause'] = True
        task_status['token_count'] = data.get('token_count', task_status['token_count']) or task_status['token_count']
        task_status['query'] = data.get('query') or task_status.get('query')
        task_status['cuda_device'] = data.get('cuda_device', task_status.get('cuda_device'))
        if data.get('abstract'):
            task_status['abstract'] = data.get('abstract')
        update_dashboard_url(data.get('dashboard_url'))
        task_status['total_prompt_tokens'] = data.get('total_prompt_tokens', task_status.get('total_prompt_tokens', 0)) or 0
        task_status['total_completion_tokens'] = data.get('total_completion_tokens', task_status.get('total_completion_tokens', 0)) or 0
        task_status['total_llm_tokens'] = data.get('total_llm_tokens', task_status.get('total_llm_tokens', 0)) or (
            (task_status['total_prompt_tokens'] or 0) + (task_status['total_completion_tokens'] or 0)
        )
        task_status['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'activity',
            'title': f'✅ Task created: {data["task_id"]}'
        })
        clamp_event_scroll()
        control_flags['force_ui_update'] = True
        with ui_state_lock:
            ui_state.update({
                'view': 'research',
                'event_scroll': 0,
                'impl_selected_index': 0,
                'impl_selected_id': None,
                'impl_log_scroll': 0,
            })
        with implementation_lock:
            implementation_registry[task_status['task_id']] = {}

        # ✅ 改进1: Signal the waiting thread that task_created was received
        task_created_event.set()

        # ✅ 改进2: Send acknowledgement to backend
        try:
            sio.emit('task_created_ack', {
                'task_id': data['task_id'],
                'timestamp': datetime.now().isoformat(),
                'client_type': 'deepscientist-cli'
            })
            print(f"[DEBUG] Sent task_created_ack for task {data['task_id']}", file=sys.stderr)
        except Exception as ack_error:
            print(f"[WARNING] Failed to send task_created_ack: {ack_error}", file=sys.stderr)

    @sio.on('event')
    def on_event(data):
        # Update task status based on event
        event_type = data.get('type', '') or 'activity'
        title = data.get('title', '')
        cycle = data.get('cycle_number')

        # Optional metadata updates embedded in event payloads
        for key in ('token_count', 'total_prompt_tokens', 'total_completion_tokens', 'total_llm_tokens'):
            if data.get(key) is not None:
                task_status[key] = int(data.get(key) or 0)

        if data.get('query'):
            task_status['query'] = data['query']
        if data.get('cuda_device') is not None:
            task_status['cuda_device'] = data['cuda_device']

        if isinstance(cycle, int) and cycle > 0:
            task_status['current_cycle'] = cycle

        stage_map = {
            'init': 'Initializing workspace',
            'main_cycle': 'Coordinating islands',
            'island_cycle': 'Research cycle',
            'explorer_generating': 'Explorer',
            'evaluation': 'Evaluator',
            'evaluation_result': 'Evaluation result',
            'strategist_decision': 'Strategist',
            'island_final_answer': 'Final answer',
            'migration_trigger': 'Migration check',
            'migration_complete': 'Migration complete',
            'implementation_start': 'Implementer',
            'implementation_success': 'Implementation success',
            'post_implementation_decision': 'Post-implementation',
            'instruction_start': 'Instruction',
            'instruction_complete': 'Instruction complete',
            'tool_update': 'Tool',
            'llm_call': 'LLM',
            'activity': 'Activity',
        }

        stage_label = data.get('stage_label') or stage_map.get(event_type)
        if stage_label:
            task_status['current_stage'] = f"[{stage_label}]"

        event_record = {
            'timestamp': data.get('timestamp', datetime.now().isoformat()),
            'type': event_type,
            'title': title,
            'cycle_number': cycle,
            'island_id': data.get('island_id'),
            'idea_id': data.get('idea_id'),
            'stage_label': stage_label,
            'model': data.get('model'),
            'prompt_tokens': data.get('prompt_tokens'),
            'completion_tokens': data.get('completion_tokens'),
            'total_tokens': data.get('total_tokens'),
            'result_preview': data.get('result_preview'),
        }

        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = event_record.get(key)
            if value is not None:
                try:
                    event_record[key] = int(value)
                except (TypeError, ValueError):
                    pass

        # Event deduplication: merge consecutive identical LLM events
        events_list = task_status['events']
        should_merge = False

        if events_list and event_type == 'llm_call':
            last_event = events_list[-1]
            # Check if last event is the same type and has the same base title (without +N suffix)
            last_title = last_event.get('title', '')
            last_agent = last_event.get('agent_name', '')
            current_agent = data.get('agent_name', '')

            # Extract base title (remove +N suffix if exists)
            last_base_title = re.sub(r'\s*\+\d+$', '', last_title)
            current_base_title = re.sub(r'\s*\+\d+$', '', title)

            if (last_event.get('type') == 'llm_call' and
                last_base_title == current_base_title and
                last_agent == current_agent):
                should_merge = True

                # Extract current count from last event title
                match = re.search(r'\+(\d+)$', last_title)
                if match:
                    count = int(match.group(1)) + 1
                else:
                    count = 2  # First merge

                # Update the last event's title with new count
                last_event['title'] = f"{current_base_title} +{count}"
                last_event['timestamp'] = event_record['timestamp']  # Update to latest timestamp

                # Update token counts if present
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    if event_record.get(key) is not None and last_event.get(key) is not None:
                        last_event[key] = event_record[key]

        if not should_merge:
            # Store agent_name in event record for future deduplication checks
            if data.get('agent_name'):
                event_record['agent_name'] = data.get('agent_name')
            events_list.append(event_record)

        clamp_event_scroll()
        control_flags['force_ui_update'] = True

    @sio.on('task_terminated')
    def on_task_terminated(data):
        """Handle server-initiated task termination"""
        message = data.get('message', 'Task has been terminated by server')
        task_id = data.get('task_id')
        retry_num = data.get('retry', 0)

        task_status['terminated_by_server'] = True
        task_status['termination_message'] = message
        task_status['status'] = 'terminated'

        # Add termination event to event log
        task_status['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'system',
            'title': '🛑 Task Terminated',
            'description': message,
        })

        clamp_event_scroll()
        control_flags['force_ui_update'] = True
        control_flags['should_exit'] = True  # Signal to exit main loop

        # Log termination
        task_status['claude_logs'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'system',
            'formatted': f'[red]🛑 Task Terminated: {message}[/red]'
        })

        # Send confirmation to backend (only on first receipt, not retries)
        if retry_num == 0:
            try:
                sio.emit('task_termination_confirmed', {
                    'task_id': task_id or task_status.get('task_id'),
                    'token': token,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"✅ Sent termination confirmation to backend for task {task_id}")
            except Exception as e:
                print(f"⚠️ Failed to send termination confirmation: {e}")

    @sio.on('task_update')
    def on_task_update(data):
        status = data.get('status') or task_status.get('status')
        task_status['status'] = status

        error_message = data.get('error')
        if error_message:
            task_status['error'] = error_message

        summary = data.get('summary')
        if summary:
            task_status['completion_summary'] = summary
            findings_total = summary.get('total_findings')
            if findings_total is not None:
                try:
                    task_status['findings_count'] = int(findings_total)
                except (TypeError, ValueError):
                    pass

            for key in ('total_llm_tokens', 'total_prompt_tokens', 'total_completion_tokens', 'token_count'):
                if summary.get(key) is not None:
                    try:
                        task_status[key] = int(summary.get(key) or 0)
                    except (TypeError, ValueError):
                        task_status[key] = summary.get(key)

            started_at = summary.get('started_at')
            if started_at:
                try:
                    task_status['start_time'] = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                except Exception:
                    pass

        if status == 'completed':
            task_status['events'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'activity',
                'title': '🎉 Task marked as completed'
            })
            clamp_event_scroll()
            control_flags['force_ui_update'] = True
            control_flags['final_summary_ready'] = True
            control_flags['stop_requested'] = True
        else:
            # Non-final updates still refresh the UI
            control_flags['force_ui_update'] = True

    @sio.on('broadcast')
    def on_broadcast(data):
        message = data.get('message', '')
        task_status['broadcasts'].append(message)
        task_status['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'activity',
            'title': f'📢 ADMIN: {message}'
        })
        clamp_event_scroll()
        control_flags['force_ui_update'] = True

    @sio.on('execute_idea')
    def on_execute_idea(data):
        """
        Handle implementation request from backend.
        Execute Claude Code locally with the provided prompt.
        """
        task_id = data.get('task_id')
        idea_id = data.get('idea_id')
        idea = data.get('idea', {})
        baseline_path = data.get('baseline_path', '')
        claude_prompt = data.get('claude_prompt', '')
        cuda_device = data.get('cuda_device', '0')

        idea_title = idea.get('title', idea_id)

        task_status['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'implementation',
            'title': f'🛠️  Executing: {idea_title}'
        })
        control_flags['force_ui_update'] = True

        # Run implementation in background thread
        import threading

        def run_implementation():
            try:
                from pathlib import Path
                import subprocess
                import json

                # Get CLI workspace directory - use workspace_dir from config
                workspace_base = get_workspace_dir()
                task_workspace = workspace_base / task_id
                task_workspace.mkdir(parents=True, exist_ok=True)

                # Create implementation directory named IDEA_{idea_id}
                impl_dir = task_workspace / f"IDEA_{idea_id}"
                if impl_dir.exists():
                    shutil.rmtree(impl_dir)
                impl_dir.mkdir(parents=True, exist_ok=True)

                # Save the complete execute_idea data to JSON file
                taskstart_file = task_workspace / f"taskstart_{idea_id}.json"
                with open(taskstart_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"📄 Saved task start data to: {taskstart_file}")

                # Copy from baseline_code in task workspace
                baseline_backup = task_workspace / "baseline_code"
                if baseline_backup.exists():
                    baseline_src = baseline_backup
                elif task_status.get('baseline_backup') and Path(task_status['baseline_backup']).exists():
                    baseline_src = Path(task_status['baseline_backup'])
                elif task_status.get('codebase_path') and Path(task_status['codebase_path']).exists():
                    # Fallback to original codebase path
                    baseline_src = Path(task_status['codebase_path'])
                else:
                    raise FileNotFoundError(f"Baseline code not found. Checked: {baseline_backup}, {task_status.get('baseline_backup')}, {task_status.get('codebase_path')}")

                shutil.copytree(baseline_src, impl_dir, dirs_exist_ok=True)

                # Save idea to method.json
                method_file = impl_dir / 'method.json'
                with open(method_file, 'w', encoding='utf-8') as f:
                    json.dump(idea, f, ensure_ascii=False, indent=2)

                # Create runlog and messages files in the task_workspace directory (not in IDEA_xxx)
                runlog_file = task_workspace / f"{idea_id}.runlog"
                messages_file = task_workspace / f"{idea_id}.messages"

                # Create CombinedStreamLogger
                combined_logger = CombinedStreamLogger(
                    str(runlog_file),
                    idea_id
                )

                # Register implementation metadata
                idea_summary = idea.get('claim', '') or idea.get('theory_and_method', '')[:200] if idea.get('theory_and_method') else ''
                register_implementation_metadata(
                    task_id=task_id,
                    idea_id=idea_id,
                    title=idea_title,
                    summary=idea_summary,
                    log_path=str(runlog_file)
                )

                # Build CLI's own prompt with conda environment configuration
                conda_config = get_conda_config()
                conda_env = conda_config.get('env_name', 'air')

                # Build implementation context from idea
                implementation_context = f"""
## IMPLEMENTATION TASK

### IDEA INFORMATION
**IDEA ID**: {idea_id}
**Title**: {idea.get('title', idea_id)}

### CLAIM
{idea.get('claim', 'N/A')}

### THEORY AND METHOD
{idea.get('theory_and_method', 'N/A')}

### IMPLEMENTATION REQUIREMENTS
{idea.get('implementation_requirements', 'N/A')}

### NOVELTY
{idea.get('novelty', 'N/A')}

### RELATION TO LITERATURE
{idea.get('relation_to_literature', 'N/A')}

### EVIDENCE
{idea.get('evidence', 'N/A')}
"""

                backend_prompt = data.get('claude_prompt') or ''
                if backend_prompt.strip():
                    prompt_prefix = backend_prompt.strip() + "\n\n" + implementation_context.lstrip('\n')
                else:
                    prompt_prefix = implementation_context.lstrip('\n')

                # Build CUDA instruction with conda environment
                if cuda_device is not None:
                    cuda_instruction = f'Use conda activate {conda_env}, must to run the test.sh code, do not skip or stop. if stop, please restart the process.'
                else:
                    cuda_instruction = f'Use conda activate {conda_env}. Before running any CUDA operations, please use nvidia-smi to check available GPUs and automatically select an available GPU by setting CUDA_VISIBLE_DEVICES=<selected_gpu_id>. Must run the test.sh code, do not skip or stop. if stop, please restart the process.'

                # Get retry configuration
                retry_config = get_retry_config()
                claude_code_max_retries = retry_config.get('claude_code_max_retries', 2)
                test_sh_max_retries = retry_config.get('test_sh_max_retries', 1)

                # Build prompts for two-phase execution
                base_directive = (
                    '\nPlease implement the IDEA in the codebase. \n\n\n'
                    "Your main focus should be on implementing the code based on THEORY AND METHOD. You have ample time to modify test.sh and the codebase. As a professional code engineer, your primary objective is to improve the performance of the current method by strictly executing the provided commands."
                    "If the initial results do not surpass the baseline, you will have one opportunity to further modify the codebase along the current direction. This requires you to first modify the `method file`, and subsequently update and re-run the experiment using `test.sh`. Should the performance remain unsatisfactory after this attempt, you are expected to analyze, debug, and iteratively correct the method code, re-running the experiment until the desired performance is achieved."
                    f'You only need to run the both datasets in test.sh, change the codebase and the method file, do not change the test.sh. {cuda_instruction}'
                )

                # Phase 1: Implementation without test.sh requirement
                phase1_prompt = (
                    prompt_prefix +
                    base_directive +
                    '\n\n## PHASE 1: IMPLEMENTATION INSTRUCTIONS\n\n' +
                    '**IMPORTANT - TASK MANAGEMENT**: You MUST:\n' +
                    '- Create a TODO list at the beginning of this phase listing ALL implementation tasks\n' +
                    '- Break down the implementation into clear, manageable steps\n' +
                    '- Mark each task as completed only when truly finished\n' +
                    '- Use the TODO list to track your progress and ensure nothing is missed\n\n' +
                    'Please implement the IDEA in the codebase following these requirements:\n\n' +
                    '**1. IMPLEMENTATION FOCUS**: Your main focus should be on implementing the code based on THEORY AND METHOD. As a professional code engineer, your primary objective is to implement the new approach correctly.\n\n' +
                    '**2. UNIT TESTING**: Ensure the implementation can run successfully with basic unit tests (if available in the codebase).\n\n' +
                    '**3. RESULT DOCUMENTATION (MANDATORY)**: Create or update a file named `Result.md` in the codebase directory. This file MUST contain:\n' +
                    '   - A summary of what was implemented\n' +
                    '   - Key code changes made (with file names and brief descriptions)\n' +
                    '   - Any important findings or notes during implementation\n' +
                    '   - Current status of the implementation\n\n' +
                    f'**4. ENVIRONMENT**: {cuda_instruction}\n\n' +
                    '**REMEMBER - PHASE 1 REQUIREMENTS**: \n' +
                    '- **CREATE A TODO LIST** at the start with all implementation tasks\n' +
                    '- The focus of this phase is correct implementation and documentation\n' +
                    '- Do NOT run test.sh yet - that will come in Phase 2\n' +
                    '- Mark tasks in your TODO list as completed only when truly done\n' +
                    '- Ensure Result.md is created and properly documents all changes\n' +
                    '- The better you document in Result.md, the easier Phase 2 will be'
                )

                # Phase 2: Method renaming and test.sh execution
                phase2_prompt = (
                    prompt_prefix +
                    base_directive +
                    '\n\n--------********-------\n\n' +
                    '## PHASE 2: FINALIZATION INSTRUCTIONS\n\n' +
                    'The IDEA implementation is now halfway complete. You MUST thoroughly review and verify Phase 1 work before proceeding.\n\n' +
                    '**IMPORTANT - SUCCESS CRITERIA**: This phase is ONLY considered successful when test.sh runs successfully without errors. You MUST:\n' +
                    '- Create a TODO list at the beginning of this phase listing ALL required tasks\n' +
                    '- Mark each task as completed only when truly finished\n' +
                    '- Do NOT end this session until test.sh executes successfully\n' +
                    '- The TODO list helps prevent premature termination and ensures all tasks are completed\n\n' +
                    'Follow these steps in order:\n\n' +
                    '**1. COMPREHENSIVE CODE REVIEW & VERIFICATION (MANDATORY)**:\n' +
                    '   a) **Read Result.md thoroughly**: Understand exactly what was supposed to be implemented in Phase 1\n' +
                    '   b) **Read all modified code files**: For each file mentioned in Result.md, carefully read the actual code to verify:\n' +
                    '      - Is the implementation correct according to THEORY AND METHOD?\n' +
                    '      - Are there any bugs, errors, or incomplete implementations?\n' +
                    '      - Does the code match what is described in Result.md?\n' +
                    '   c) **Cross-check implementation against requirements**: Compare the actual code with:\n' +
                    '      - The CLAIM section above\n' +
                    '      - The THEORY AND METHOD section above\n' +
                    '      - The IMPLEMENTATION REQUIREMENTS section above\n' +
                    '   d) **Fix any issues found**: If you discover ANY of the following, you MUST fix them immediately:\n' +
                    '      - Incorrect implementation that does not match THEORY AND METHOD\n' +
                    '      - Bugs or errors in the code\n' +
                    '      - Incomplete implementation (missing parts)\n' +
                    '      - Code that contradicts the IDEA requirements\n' +
                    '   e) **Update Result.md if fixes were made**: Document any corrections you made during this review step\n\n' +
                    '**CRITICAL**: Do NOT proceed to step 2 until you have verified that Phase 1 implementation is correct and complete. If you find errors, fix them first!\n\n' +
                    '**2. METHOD RENAMING (MANDATORY)**: You MUST rename the new method to reflect the new approach. This is critical:\n' +
                    '   - Choose a descriptive name that clearly indicates this is the new method\n' +
                    '   - Update all references in the codebase\n' +
                    '   - Update the `method` file if it exists\n\n' +
                    '**3. TEST.SH MODIFICATION (MANDATORY)**: Modify test.sh to use the new method name. Make sure test.sh calls the renamed method.\n\n' +
                    '**4. TEST.SH EXECUTION (MANDATORY)**: Execute test.sh successfully AT LEAST ONCE. The execution must complete without errors. Record the output.\n\n' +
                    '**5. UPDATE RESULT.MD**: Add to Result.md:\n' +
                    '   - Any corrections made during code review (Step 1) - if applicable\n' +
                    '   - The new method name you chose\n' +
                    '   - Changes made to test.sh\n' +
                    '   - Complete test.sh execution output and results\n' +
                    '   - Final status and any performance metrics\n\n' +
                    f'**6. ENVIRONMENT**: {cuda_instruction}\n\n' +
                    '**REMEMBER - CRITICAL SUCCESS REQUIREMENTS**: \n' +
                    '- **CREATE A TODO LIST** at the start of Phase 2 with all tasks (review code, fix issues, rename method, modify test.sh, run test.sh)\n' +
                    '- Code review and verification (Step 1) is the FIRST priority. Fix any issues before proceeding.\n' +
                    '- Method renaming and successful test.sh execution are NON-NEGOTIABLE requirements.\n' +
                    '- **DO NOT END THIS SESSION** until test.sh runs successfully without errors.\n' +
                    '- Mark tasks in your TODO list as completed only when truly done.\n' +
                    '- The implementation is NOT complete until ALL steps are done correctly, test.sh succeeds, and everything is documented in Result.md.\n' +
                    '- **SUCCESS = test.sh runs without errors**. Nothing less is acceptable.'
                )

                print(f"\n📋 Using two-phase Claude Code execution")
                print(f"   Phase 1 prompt: {len(phase1_prompt)} chars")
                print(f"   Phase 2 prompt: {len(phase2_prompt)} chars")
                print(f"   Conda environment: {conda_env}")
                print(f"   CUDA device: {cuda_device}")

                # Set environment variables
                import os
                env = os.environ.copy()
                if cuda_device is not None and cuda_device != '':
                    env['CUDA_VISIBLE_DEVICES'] = cuda_device

                # Log initial messages
                combined_logger.log_message(f"Starting two-phase implementation for IDEA: {idea_id}", "SYSTEM")
                combined_logger.log_message(f"Implementation directory: {impl_dir}", "SYSTEM")
                combined_logger.log_message(f"CUDA device: {cuda_device}", "SYSTEM")

                # ==================================================================
                # FIRST CLAUDE CODE CALL - Implementation without test.sh
                # ==================================================================
                task_status['events'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'implementation',
                    'title': f'⚙️  Phase 1/2: Initial Implementation - {idea_id}'
                })
                control_flags['force_ui_update'] = True
                clamp_event_scroll()

                print(f"\n{'='*80}")
                print(f"🛠️  PHASE 1/2: Initial Implementation (without test.sh) - {idea_id}")
                print(f"   Max retries: {claude_code_max_retries} (total {claude_code_max_retries + 1} attempts)")
                print(f"{'='*80}\n")

                combined_logger.log_message("="*80, "SYSTEM")
                combined_logger.log_message("=== CLAUDE CODE CALL 1/2: INITIAL IMPLEMENTATION ===", "SYSTEM")
                combined_logger.log_message(f"Max retries configured: {claude_code_max_retries}", "SYSTEM")
                combined_logger.log_message("="*80, "SYSTEM")

                # Clear claude_logs at the start
                task_status['claude_logs'].clear()
                control_flags['force_ui_update'] = True
                clamp_event_scroll()

                # Retry loop for Phase 1
                phase1_return_code = 1
                phase1_attempt = 0

                while phase1_attempt <= claude_code_max_retries:
                    phase1_attempt += 1

                    # Determine if this is a retry attempt
                    is_retry = phase1_attempt > 1

                    # Build current attempt prompt (add retry reminder for retries)
                    current_phase1_prompt = phase1_prompt
                    if is_retry:
                        retry_reminder = (
                            "\n\n--------********-------\n\n"
                            "**IMPORTANT - THIS IS A RETRY ATTEMPT**: "
                            "This is a retry attempt. The previous execution encountered an unknown error and was terminated. "
                            "Please continue with the current task. First, make a complete plan for the current task. "
                            "If you find some code has already been modified, you can skip it and proceed with further plans. "
                            "Only stop when all tasks in the current command are completed."
                        )
                        current_phase1_prompt = phase1_prompt + retry_reminder

                    # Log attempt information
                    attempt_msg = f"Attempt {phase1_attempt}/{claude_code_max_retries + 1}" + (" (RETRY)" if is_retry else " (INITIAL)")
                    print(f"\n{'─'*60}")
                    print(f"🔄 {attempt_msg}")
                    print(f"{'─'*60}\n")

                    combined_logger.log_message("="*60, "SYSTEM")
                    combined_logger.log_message(f"=== Phase 1 {attempt_msg} ===", "SYSTEM")
                    combined_logger.log_message("="*60, "SYSTEM")

                    # Build command for Phase 1
                    phase1_command = [
                        'claude',
                        '--output-format', 'stream-json',
                        '-p',
                        '--dangerously-skip-permissions',
                        '--verbose',
                        current_phase1_prompt
                    ]

                    # Create manager for Phase 1
                    manager1 = ClaudeCodeStreamManager(
                        show_types={'assistant', 'user', 'result', 'system'},
                        show_tools=True,
                        cwd=str(impl_dir),
                        cuda_device=cuda_device
                    )
                    manager1.processor = ClaudeCodeStreamProcessor(
                        cwd=str(impl_dir),
                        cuda_device=cuda_device,
                        combined_logger=combined_logger
                    )

                    try:
                        manager1.process_claude_stream(phase1_command)
                        # Success!
                        phase1_return_code = 0
                        print(f"\n✅ Phase 1 {attempt_msg} completed successfully")
                        combined_logger.log_message(f"=== Phase 1 {attempt_msg} SUCCEEDED ===", "SYSTEM")
                        break  # Exit retry loop on success

                    except KeyboardInterrupt:
                        # User requested stop via /q command - propagate immediately
                        print(f"\n⚠️  Phase 1 stopped by user request (/q command)")
                        combined_logger.log_message("="*60, "SYSTEM")
                        combined_logger.log_message("Phase 1 stopped by user request (/q command)", "SYSTEM")
                        combined_logger.log_message("="*60, "SYSTEM")
                        raise  # Re-raise to stop execution completely

                    except Exception as e:
                        import traceback as tb
                        error_str = str(e)

                        # Log the error
                        if "Pre-flight check" in error_str or "BashTool" in error_str:
                            print(f"\n❌ Claude Code API pre-flight check failed!")
                            print(f"   This usually means:")
                            print(f"   1. ANTHROPIC_API_KEY is not set or invalid")
                            print(f"   2. Network connection to api.anthropic.com is blocked")
                            print(f"   3. Claude API service is experiencing issues")
                        else:
                            print(f"❌ Phase 1 {attempt_msg} error: {e}")

                        phase1_return_code = 1
                        combined_logger.log_message("="*60, "ERROR")
                        combined_logger.log_message(f"ERROR: Phase 1 {attempt_msg} failed", "ERROR")
                        combined_logger.log_message(f"Exception: {e}", "ERROR")
                        combined_logger.log_message(f"Traceback:\n{tb.format_exc()}", "ERROR")
                        combined_logger.log_message("="*60, "ERROR")

                        # Check if we should retry
                        if phase1_attempt <= claude_code_max_retries:
                            retry_msg = f"Will retry... ({claude_code_max_retries - phase1_attempt + 1} retries remaining)"
                            print(f"\n🔄 {retry_msg}")
                            combined_logger.log_message(retry_msg, "SYSTEM")
                        else:
                            # All retries exhausted
                            final_msg = f"All {claude_code_max_retries + 1} attempts failed. No more retries available."
                            print(f"\n❌ {final_msg}")
                            combined_logger.log_message(final_msg, "ERROR")

                combined_logger.log_message("="*80, "SYSTEM")
                combined_logger.log_message(f"=== PHASE 1/2 COMPLETED (return code: {phase1_return_code}) ===", "SYSTEM")
                combined_logger.log_message("="*80, "SYSTEM")

                # ==================================================================
                # SECOND CLAUDE CODE CALL - Method renaming and test.sh execution
                # ==================================================================
                task_status['events'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'implementation',
                    'title': f'⚙️  Phase 2/2: Finalization & Testing - {idea_id}'
                })
                control_flags['force_ui_update'] = True
                clamp_event_scroll()

                print(f"\n{'='*80}")
                print(f"🛠️  PHASE 2/2: Finalization & Testing (method rename + test.sh) - {idea_id}")
                print(f"   Max retries: {claude_code_max_retries} (total {claude_code_max_retries + 1} attempts)")
                print(f"{'='*80}\n")

                combined_logger.log_message("\n\n--------********-------\n", "SYSTEM")
                combined_logger.log_message("="*80, "SYSTEM")
                combined_logger.log_message("=== CLAUDE CODE CALL 2/2: FINALIZATION & TESTING ===", "SYSTEM")
                combined_logger.log_message(f"Max retries configured: {claude_code_max_retries}", "SYSTEM")
                combined_logger.log_message("="*80, "SYSTEM")

                # Retry loop for Phase 2
                phase2_return_code = 1
                phase2_attempt = 0

                while phase2_attempt <= claude_code_max_retries:
                    phase2_attempt += 1

                    # Determine if this is a retry attempt
                    is_retry = phase2_attempt > 1

                    # Build current attempt prompt (add retry reminder for retries)
                    current_phase2_prompt = phase2_prompt
                    if is_retry:
                        retry_reminder = (
                            "\n\n--------********-------\n\n"
                            "**IMPORTANT - THIS IS A RETRY ATTEMPT**: "
                            "This is a retry attempt. The previous execution encountered an unknown error and was terminated. "
                            "Please continue with the current task. First, make a complete plan for the current task. "
                            "If you find some code has already been modified, you can skip it and proceed with further plans. "
                            "Only stop when all tasks in the current command are completed."
                        )
                        current_phase2_prompt = phase2_prompt + retry_reminder

                    # Log attempt information
                    attempt_msg = f"Attempt {phase2_attempt}/{claude_code_max_retries + 1}" + (" (RETRY)" if is_retry else " (INITIAL)")
                    print(f"\n{'─'*60}")
                    print(f"🔄 {attempt_msg}")
                    print(f"{'─'*60}\n")

                    combined_logger.log_message("="*60, "SYSTEM")
                    combined_logger.log_message(f"=== Phase 2 {attempt_msg} ===", "SYSTEM")
                    combined_logger.log_message("="*60, "SYSTEM")

                    # Build command for Phase 2
                    phase2_command = [
                        'claude',
                        '--output-format', 'stream-json',
                        '-p',
                        '--dangerously-skip-permissions',
                        '--verbose',
                        current_phase2_prompt
                    ]

                    # Create manager for Phase 2
                    manager2 = ClaudeCodeStreamManager(
                        show_types={'assistant', 'user', 'result', 'system'},
                        show_tools=True,
                        cwd=str(impl_dir),
                        cuda_device=cuda_device
                    )
                    manager2.processor = ClaudeCodeStreamProcessor(
                        cwd=str(impl_dir),
                        cuda_device=cuda_device,
                        combined_logger=combined_logger
                    )

                    try:
                        manager2.process_claude_stream(phase2_command)
                        # Success!
                        phase2_return_code = 0
                        print(f"\n✅ Phase 2 {attempt_msg} completed successfully")
                        combined_logger.log_message(f"=== Phase 2 {attempt_msg} SUCCEEDED ===", "SYSTEM")
                        break  # Exit retry loop on success

                    except KeyboardInterrupt:
                        # User requested stop via /q command - propagate immediately
                        print(f"\n⚠️  Phase 2 stopped by user request (/q command)")
                        combined_logger.log_message("="*60, "SYSTEM")
                        combined_logger.log_message("Phase 2 stopped by user request (/q command)", "SYSTEM")
                        combined_logger.log_message("="*60, "SYSTEM")
                        raise  # Re-raise to stop execution completely

                    except Exception as e:
                        import traceback as tb
                        print(f"❌ Phase 2 {attempt_msg} error: {e}")
                        phase2_return_code = 1
                        combined_logger.log_message("="*60, "ERROR")
                        combined_logger.log_message(f"ERROR: Phase 2 {attempt_msg} failed", "ERROR")
                        combined_logger.log_message(f"Exception: {e}", "ERROR")
                        combined_logger.log_message(f"Traceback:\n{tb.format_exc()}", "ERROR")
                        combined_logger.log_message("="*60, "ERROR")

                        # Check if we should retry
                        if phase2_attempt <= claude_code_max_retries:
                            retry_msg = f"Will retry... ({claude_code_max_retries - phase2_attempt + 1} retries remaining)"
                            print(f"\n🔄 {retry_msg}")
                            combined_logger.log_message(retry_msg, "SYSTEM")
                        else:
                            # All retries exhausted
                            final_msg = f"All {claude_code_max_retries + 1} attempts failed. No more retries available."
                            print(f"\n❌ {final_msg}")
                            combined_logger.log_message(final_msg, "ERROR")

                combined_logger.log_message("="*80, "SYSTEM")
                combined_logger.log_message(f"=== PHASE 2/2 COMPLETED (return code: {phase2_return_code}) ===", "SYSTEM")
                combined_logger.log_message("="*80, "SYSTEM")

                # Overall return code
                claude_return_code = max(phase1_return_code, phase2_return_code)

                # Phase 2: Run test.sh with retry logic
                test_script = impl_dir / 'test.sh'
                test_output_lines = []

                print(f"\n{'='*60}")
                print(f"🧪 Running test.sh - {idea_id}")
                print(f"   Max retries: {test_sh_max_retries} (total {test_sh_max_retries + 1} attempts)")
                print(f"{'='*60}\n")

                combined_logger.log_message("="*80, "SYSTEM")
                combined_logger.log_message("=== Running test.sh ===", "SYSTEM")
                combined_logger.log_message(f"Test script path: {test_script}", "SYSTEM")
                combined_logger.log_message(f"Working directory: {impl_dir}", "SYSTEM")
                combined_logger.log_message(f"Max retries configured: {test_sh_max_retries}", "SYSTEM")
                combined_logger.log_message("="*80, "SYSTEM")

                success = False
                test_return_code = 1

                if test_script.exists():
                    # Retry loop for test.sh
                    test_attempt = 0

                    while test_attempt <= test_sh_max_retries:
                        test_attempt += 1

                        # Clear output lines for new attempt
                        test_output_lines = []

                        # Determine if this is a retry attempt
                        is_retry = test_attempt > 1

                        # Log attempt information
                        attempt_msg = f"Attempt {test_attempt}/{test_sh_max_retries + 1}" + (" (RETRY)" if is_retry else " (INITIAL)")
                        print(f"\n{'─'*60}")
                        print(f"🔄 test.sh {attempt_msg}")
                        print(f"{'─'*60}\n")

                        combined_logger.log_message("="*60, "SYSTEM")
                        combined_logger.log_message(f"=== test.sh {attempt_msg} ===", "SYSTEM")
                        combined_logger.log_message("="*60, "SYSTEM")

                        task_status['events'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'implementation',
                            'title': f'🧪 Running test.sh for {idea_id} - {attempt_msg}'
                        })
                        control_flags['force_ui_update'] = True
                        clamp_event_scroll()

                        # Run test.sh - capture both stdout and stderr
                        # Use pre-configured conda activation command
                        conda_activate = get_conda_activate_command()
                        test_command = f"{conda_activate} && bash test.sh"
                        try:
                            test_process = subprocess.Popen(
                                ['bash', '-c', test_command],
                                cwd=impl_dir,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True,
                                text=True,
                                env=env,
                                bufsize=1
                            )

                            # Read stdout
                            for line in test_process.stdout:
                                test_output_lines.append(line.rstrip())
                                print(line, end='', flush=True)  # Display in real-time
                                combined_logger.log_message(line.rstrip(), "TEST", print_to_console=False)

                            test_process.wait()
                            test_return_code = test_process.returncode

                            # Read stderr after process completes
                            stderr_output = test_process.stderr.read()
                            if stderr_output:
                                combined_logger.log_message("--- STDERR OUTPUT ---", "TEST")
                                combined_logger.log_message(stderr_output, "TEST", print_to_console=False)
                                combined_logger.log_message("--- END STDERR ---", "TEST")
                                test_output_lines.append("--- STDERR OUTPUT ---")
                                test_output_lines.extend(stderr_output.strip().split('\n'))
                                test_output_lines.append("--- END STDERR ---")
                                print(f"\n[STDERR]\n{stderr_output}")

                            combined_logger.log_message(f"=== test.sh {attempt_msg} completed (return code: {test_return_code}) ===", "SYSTEM")

                            if test_return_code != 0:
                                combined_logger.log_message(f"WARNING: test.sh exited with non-zero code: {test_return_code}", "WARNING")
                                success = False

                                # Check if we should retry
                                if test_attempt <= test_sh_max_retries:
                                    retry_msg = f"test.sh failed. Will retry... ({test_sh_max_retries - test_attempt + 1} retries remaining)"
                                    print(f"\n🔄 {retry_msg}")
                                    combined_logger.log_message(retry_msg, "SYSTEM")
                                else:
                                    # All retries exhausted
                                    final_msg = f"test.sh failed after all {test_sh_max_retries + 1} attempts. No more retries available."
                                    print(f"\n❌ {final_msg}")
                                    combined_logger.log_message(final_msg, "ERROR")
                            else:
                                # Success!
                                success = True
                                print(f"\n✅ test.sh {attempt_msg} completed successfully")
                                combined_logger.log_message(f"=== test.sh {attempt_msg} SUCCEEDED ===", "SYSTEM")
                                break  # Exit retry loop on success

                        except Exception as test_error:
                            import traceback as tb
                            error_msg = f"ERROR: Exception while running test.sh: {test_error}"
                            combined_logger.log_message(error_msg, "ERROR")
                            combined_logger.log_message(f"Traceback: {tb.format_exc()}", "ERROR")
                            test_output_lines.append(error_msg)
                            print(f"\n❌ {error_msg}")
                            success = False
                            test_return_code = 1

                            # Check if we should retry
                            if test_attempt <= test_sh_max_retries:
                                retry_msg = f"test.sh exception occurred. Will retry... ({test_sh_max_retries - test_attempt + 1} retries remaining)"
                                print(f"\n🔄 {retry_msg}")
                                combined_logger.log_message(retry_msg, "SYSTEM")
                            else:
                                # All retries exhausted
                                final_msg = f"test.sh failed with exceptions after all {test_sh_max_retries + 1} attempts. No more retries available."
                                print(f"\n❌ {final_msg}")
                                combined_logger.log_message(final_msg, "ERROR")

                else:
                    error_msg = "ERROR: test.sh not found in implementation directory"
                    combined_logger.log_message(error_msg, "ERROR")
                    test_output_lines = [error_msg]
                    print(f"\n❌ {error_msg}")
                    success = False

                combined_logger.log_message("=== IMPLEMENTATION COMPLETE ===", "SYSTEM")

                # Save messages to .messages file
                combined_logger.save_messages(str(messages_file))
                print(f"💾 Saved messages to: {messages_file}")

                # Read full runlog and messages
                with open(runlog_file, 'r', encoding='utf-8') as f:
                    logfile_content = f.read()

                with open(messages_file, 'r', encoding='utf-8') as f:
                    messages_content = f.read()

                # Send result back to backend
                test_output = '\n'.join(test_output_lines[-500:])  # Last 500 lines

                print(f"\n📤 Sending implementation result to backend via HTTP POST...")
                print(f"   Task ID: {task_id}")
                print(f"   Idea ID: {idea_id}")
                print(f"   Success: {success}")
                print(f"   Log size: {len(logfile_content)} bytes")
                print(f"   Messages size: {len(messages_content)} bytes")

                # Prepare payload for backend acknowledgement loop
                token = task_status.get('token')
                server = task_status.get('server_url', SERVER_URL)

                if not token:
                    raise Exception("No authentication token available")

                payload = {
                    'idea_id': idea_id,
                    'success': success,
                    'test_output': test_output,
                    'logfile': logfile_content,
                    'messages': messages_content,
                    'workspace_path': str(impl_dir)
                }

                acknowledgement = None
                ack_received = False
                send_attempt = 0
                retry_delay_seconds = 30
                max_retries = 5  # Maximum number of retry attempts before auto-termination

                while not ack_received and not control_flags['stop_requested']:
                    send_attempt += 1

                    task_status['events'].append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'implementation',
                        'title': f'📤 Sending result to backend (attempt {send_attempt}/{max_retries}): {idea_id} ({"Success" if success else "Failed"})'
                    })
                    control_flags['force_ui_update'] = True
                    clamp_event_scroll()

                    try:
                        response = requests.post(
                            f"{server}/api/tasks/{task_id}/implementation_result",
                            headers={'Authorization': f'Bearer {token}'},
                            json=payload,
                            timeout=500  # 500 second timeout for LLM summary generation in background
                        )

                        if response.status_code != 200:
                            raise Exception(f"HTTP {response.status_code}: {response.text}")

                        try:
                            acknowledgement = response.json()
                        except Exception:
                            acknowledgement = {}

                        if acknowledgement.get('status') == 'success' and acknowledgement.get('acknowledged') is not False:
                            ack_received = True
                            ack_message = acknowledgement.get('message', 'Backend acknowledged implementation result')
                            print(f"✅ Result acknowledged by backend via HTTP POST")

                            task_status['events'].append({
                                'timestamp': datetime.now().isoformat(),
                                'type': 'implementation',
                                'title': f'✅ Backend acknowledged result: {idea_id} ({ack_message})'
                            })
                            control_flags['force_ui_update'] = True
                            clamp_event_scroll()
                        else:
                            raise Exception(f"Acknowledgement missing or invalid: {acknowledgement}")

                    except Exception as send_error:
                        error_msg = f"Failed to send result (attempt {send_attempt}): {send_error}"
                        print(f"❌ {error_msg}")

                        task_status['events'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'error',
                            'title': f'❌ Send attempt {send_attempt} failed for {idea_id}: {str(send_error)[:120]}'
                        })
                        control_flags['force_ui_update'] = True
                        clamp_event_scroll()

                        if control_flags['stop_requested']:
                            break

                        # Check if max retries reached
                        if send_attempt >= max_retries:
                            task_status['events'].append({
                                'timestamp': datetime.now().isoformat(),
                                'type': 'error',
                                'title': f'❌ Max retries ({max_retries}) reached for {idea_id}. Auto-terminating task...'
                            })
                            control_flags['force_ui_update'] = True
                            clamp_event_scroll()

                            # Trigger auto-termination
                            print(f"\n{'='*60}")
                            print(f"⚠️  AUTO-TERMINATION: Failed to submit results after {max_retries} attempts")
                            print(f"Stopping backend task and exiting CLI...")
                            print(f"{'='*60}\n")

                            # Set stop flag to terminate monitoring loop
                            control_flags['stop_requested'] = True
                            break

                        # Inform user about upcoming retry
                        task_status['events'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'implementation',
                            'title': f'⏳ Will retry sending result for {idea_id} in {retry_delay_seconds} seconds'
                        })
                        control_flags['force_ui_update'] = True
                        clamp_event_scroll()

                        for _ in range(retry_delay_seconds):
                            if control_flags['stop_requested']:
                                break
                            time.sleep(1)

                # Decide final status based on acknowledgement outcome
                if ack_received:
                    result_summary = f"Success: {success}. Test output: {test_output[:100]}..." if len(test_output) > 100 else f"Success: {success}. {test_output}"
                    update_implementation_status(
                        task_id=task_id,
                        idea_id=idea_id,
                        status='success' if success else 'failed',
                        result_summary=result_summary
                    )

                    task_status['events'].append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'implementation',
                        'title': f'✅ Completed: {idea_title} - Ready for next cycle' if success else f'❌ Failed: {idea_title} - Backend will continue next cycle'
                    })
                    control_flags['force_ui_update'] = True
                    clamp_event_scroll()
                else:
                    pending_summary = (
                        "Success acknowledged" if success else "Implementation reported failure"
                    )
                    pending_summary += ", but backend confirmation was not received before stopping."
                    update_implementation_status(
                        task_id=task_id,
                        idea_id=idea_id,
                        status='failed',
                        result_summary=pending_summary
                    )

                    task_status['events'].append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'error',
                        'title': f'⚠️ Backend acknowledgement missing for {idea_title}. Manual intervention required.'
                    })
                    control_flags['force_ui_update'] = True
                    clamp_event_scroll()

            except KeyboardInterrupt:
                # User or server requested stop - stop immediately
                print(f"\n⚠️  Implementation stopped (user/server request)")

                # Set stop flag to ensure clean exit
                control_flags['stop_requested'] = True
                control_flags['should_exit'] = True

                # Log the termination
                task_status['events'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'system',
                    'title': f'🛑 Implementation {idea_id} stopped by termination request'
                })
                control_flags['force_ui_update'] = True
                clamp_event_scroll()

                # Send termination result to backend
                try:
                    sio.emit('idea_result', {
                        'task_id': task_id,
                        'idea_id': idea_id,
                        'success': False,
                        'test_output': 'Implementation stopped by termination request',
                        'error': 'Task terminated',
                        'logfile': 'Implementation was terminated before completion'
                    })
                except Exception as send_error:
                    print(f"Warning: Could not send termination result: {send_error}")

                # Don't re-raise - let the function complete normally
                return

            except Exception as e:
                import traceback
                error_msg = f"Error executing idea: {e}\n{traceback.format_exc()}"
                print(f"\n❌ FATAL ERROR during implementation: {e}")
                print(error_msg)

                # Try to write error to log file - use correct path matching main flow
                runlog_file = None
                try:
                    workspace_base = get_workspace_dir()
                    task_workspace = workspace_base / task_id
                    # runlog file is in task_workspace, not in IDEA_xxx directory
                    runlog_file = task_workspace / f"{idea_id}.runlog"

                    if runlog_file.exists():
                        with open(runlog_file, 'a', encoding='utf-8') as log_f:
                            log_f.write(f"\n{'='*60}\n")
                            log_f.write(f"FATAL ERROR: Unexpected exception during implementation\n")
                            log_f.write(f"{'='*60}\n")
                            log_f.write(error_msg)
                            log_f.write(f"\n{'='*60}\n")
                    else:
                        # Create log file if it doesn't exist
                        task_workspace.mkdir(parents=True, exist_ok=True)
                        with open(runlog_file, 'w', encoding='utf-8') as log_f:
                            log_f.write(f"{'='*60}\n")
                            log_f.write(f"FATAL ERROR: Implementation failed to start\n")
                            log_f.write(f"{'='*60}\n")
                            log_f.write(error_msg)
                            log_f.write(f"\n{'='*60}\n")
                except Exception as log_error:
                    print(f"Warning: Could not write error to log file: {log_error}")

                # Update implementation status to failed
                try:
                    update_implementation_status(
                        task_id=task_id,
                        idea_id=idea_id,
                        status='failed',
                        result_summary=f"Exception: {str(e)[:200]}"
                    )
                except Exception as status_error:
                    print(f"Warning: Could not update implementation status: {status_error}")

                # Read log file content if available
                logfile_content = error_msg
                try:
                    if runlog_file and runlog_file.exists():
                        with open(runlog_file, 'r', encoding='utf-8') as f:
                            logfile_content = f.read()
                except Exception as read_error:
                    print(f"Warning: Could not read log file: {read_error}")

                # Send error result to backend
                print(f"\n📤 Sending error result to backend...")
                print(f"   Task ID: {task_id}")
                print(f"   Idea ID: {idea_id}")
                print(f"   Error: {str(e)[:100]}")

                try:
                    sio.emit('idea_result', {
                        'task_id': task_id,
                        'idea_id': idea_id,
                        'success': False,
                        'test_output': error_msg,
                        'logfile': logfile_content,
                        'workspace_path': str(impl_dir) if 'impl_dir' in locals() else ''
                    })
                    print(f"✅ Error result sent to backend")
                except Exception as emit_error:
                    print(f"❌ Failed to send error result to backend: {emit_error}")

                # Update task events
                task_status['events'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'implementation',
                    'title': f'❌ Failed: {idea_title} - {str(e)[:100]}'
                })
                control_flags['force_ui_update'] = True
                clamp_event_scroll()

                task_status['events'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'error',
                    'title': f'❌ Error: {idea_title}'
                })
                control_flags['force_ui_update'] = True
                clamp_event_scroll()

                print(f"\n❌ Fatal error during implementation:")
                print(error_msg)

        # Start implementation in background
        thread = threading.Thread(target=run_implementation, daemon=True)
        thread.start()

    @sio.on('task_paused')
    def on_task_paused(data):
        task_status['status'] = 'paused'
        task_status['can_pause'] = False
        task_status['can_resume'] = True
        control_flags['paused'] = True
        task_status['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'activity',
            'title': '⏸️  Task paused'
        })

    @sio.on('task_resumed')
    def on_task_resumed(data):
        task_status['status'] = 'started'
        task_status['can_pause'] = True
        task_status['can_resume'] = False
        control_flags['paused'] = False
        task_status['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'activity',
            'title': '▶️  Task resumed'
        })

    @sio.on('stop_task')
    def on_stop_task(data):
        reason = data.get('reason', 'unknown')
        task_status['status'] = 'stopped'
        control_flags['stop_requested'] = True
        task_status['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'title': f'🛑 Task stopped: {reason}'
        })

    @sio.on('error')
    def on_error(data):
        message = data.get('message', 'Unknown error')
        task_status['error'] = message
        task_status['error_code'] = data.get('code')
        task_status['error_details'] = data.get('details')
        task_status['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'title': f'❌ Error: {message}'
        })

    @sio.on('task_status_update')
    def on_task_status_update(data):
        """Handle task status updates from backend"""
        new_status = data.get('status')
        if new_status:
            task_status['status'] = new_status
            control_flags['force_ui_update'] = True
            clamp_event_scroll()

            if new_status == 'terminated':
                control_flags['stop_requested'] = True
                task_status['events'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'system',
                    'title': '🛑 Task terminated'
                })

    @sio.on('stop_confirmed')
    def on_stop_confirmed(data):
        """Handle stop confirmation from backend"""
        task_status['status'] = 'terminated'
        control_flags['stop_requested'] = True
        task_status['events'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'system',
            'title': '✓ Task termination confirmed'
        })
        control_flags['force_ui_update'] = True

    return sio


def heartbeat_thread(sio, token):
    """Background thread for sending heartbeats (interval configurable, default 30 minutes)"""
    # Get heartbeat interval from config (default: 1800 seconds = 30 minutes)
    conn_config = get_connection_config()
    heartbeat_interval = conn_config['heartbeat_interval']

    while task_status['connected'] and not control_flags['stop_requested']:
        time.sleep(heartbeat_interval)  # Configurable interval (default: 30 minutes)
        if task_status['connected'] and task_status['task_id']:
            try:
                task_status['heartbeat_ok'] = False
                sio.emit('heartbeat', {
                    'task_id': task_status['task_id'],
                    'token': token
                })
                task_status['heartbeat_ok'] = True
                # Calculate next heartbeat time using configured interval (in seconds)
                task_status['heartbeat_next'] = datetime.now() + timedelta(seconds=heartbeat_interval)
            except:
                task_status['heartbeat_ok'] = False


def input_listener_thread():
    """Background thread for listening to user input (arrows, controls)."""
    import sys
    import select
    import tty
    import termios

    def handle_sequence(sequence: str) -> bool:
        if not sequence:
            return False

        lower_seq = sequence.lower()
        if lower_seq == 'q':
            control_flags['stop_requested'] = True
            return True
        if lower_seq == 's':
            control_flags['stop_requested'] = True
            return True
        if lower_seq == 'p' and task_status.get('can_pause'):
            control_flags['user_input_mode'] = True
            return True
        if lower_seq == 'r' and task_status.get('can_resume'):
            control_flags['user_input_mode'] = True
            return True

        view = ui_state.get('view', 'research')
        task_id = task_status.get('task_id')

        # Tab key to switch between implementers
        if sequence == '\t':
            if view == 'implementer' and task_id:
                return change_impl_selection(task_id, +1)
            return False

        arrow_sequences = {
            '\x1b[A', '\x1bOA',  # Up
            '\x1b[B', '\x1bOB',  # Down
            '\x1b[C', '\x1bOC',  # Right
            '\x1b[D', '\x1bOD',  # Left
            '\x1b[5~',  # Page Up
            '\x1b[6~',  # Page Down
        }

        if sequence not in arrow_sequences:
            return False

        if sequence in {'\x1b[A', '\x1bOA'}:  # Up
            if view == 'research':
                if modify_event_scroll(+1):
                    control_flags['force_ui_update'] = True
                clamp_event_scroll()
                return True
            elif task_id:
                # In implementer view, up arrow scrolls log up
                if modify_impl_log_scroll(task_id, +1):
                    control_flags['force_ui_update'] = True
                clamp_event_scroll()
                return True
            return False

        if sequence in {'\x1b[B', '\x1bOB'}:  # Down
            if view == 'research':
                if modify_event_scroll(-1):
                    control_flags['force_ui_update'] = True
                clamp_event_scroll()
                return True
            elif task_id:
                # In implementer view, down arrow scrolls log down
                if modify_impl_log_scroll(task_id, -1):
                    control_flags['force_ui_update'] = True
                clamp_event_scroll()
                return True
            return False

        if sequence in {'\x1b[C', '\x1bOC'}:  # Right
            if switch_view('implementer'):
                control_flags['force_ui_update'] = True
                clamp_event_scroll()
                return True
            return False

        if sequence in {'\x1b[D', '\x1bOD'}:  # Left
            if switch_view('research'):
                control_flags['force_ui_update'] = True
                clamp_event_scroll()
                return True
            return False

        if sequence == '\x1b[5~':  # Page Up
            if view == 'research':
                if modify_event_scroll(+5):
                    control_flags['force_ui_update'] = True
                clamp_event_scroll()
                return True
            elif task_id:
                if modify_impl_log_scroll(task_id, +10):
                    control_flags['force_ui_update'] = True
                clamp_event_scroll()
                return True
            return False

        if sequence == '\x1b[6~':  # Page Down
            if view == 'research':
                if modify_event_scroll(-5):
                    control_flags['force_ui_update'] = True
                clamp_event_scroll()
                return True
            elif task_id:
                if modify_impl_log_scroll(task_id, -10):
                    control_flags['force_ui_update'] = True
                clamp_event_scroll()
                return True
            return False

        return False

    while not control_flags['stop_requested']:
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        except Exception:
            time.sleep(0.1)
            continue

        if not rlist:
            continue

        try:
            old_settings = termios.tcgetattr(sys.stdin)
        except termios.error:
            time.sleep(0.1)
            continue

        sequence = ''
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                sequence += ch
                while True:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.01)
                    if ready:
                        next_ch = sys.stdin.read(1)
                        sequence += next_ch
                        if next_ch.isalpha() or next_ch == '~':
                            break
                    else:
                        break
            else:
                sequence = ch
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        try:
            if handle_sequence(sequence):
                control_flags['force_ui_update'] = True
                clamp_event_scroll()
        except Exception:
            # Ensure robust behaviour even if sequence parsing fails
            pass



# ============================================================================
# CLI Commands
# ============================================================================

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """DeepScientistCLI Enhanced - Professional Research Platform"""
    initialize_cli_version_metadata()
    default_server = resolve_server(None)
    check_for_cli_update(default_server)
    # Initialize conda configuration on startup
    conda_path = detect_and_configure_conda()
    if ctx.invoked_subcommand is None:
        # Show conda info in interactive mode
        console.print(f"[dim]Conda detected at: {conda_path}[/dim]")
        run_interactive_shell(ctx)
        return


@cli.command()
@click.option('--token', help='API authentication token')
@click.option('--server', default=None, help=f'Backend server URL (default: {SERVER_URL})')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode with guided setup')
def login(token, server, interactive):
    """Verify API token and save configuration"""
    print_banner()

    # Interactive mode setup
    if interactive:
        console.print("[bold cyan]🚀 DeepScientist Interactive Setup[/bold cyan]")
        console.print("─" * 60)
        console.print("[dim]This wizard will help you configure your CLI connection.[/dim]\n")

        # Server configuration with common options
        console.print("[bold]1. Server Configuration[/bold]")
        console.print("Select your server:")
        console.print("  [1] localhost:5000 (default development)")
        console.print("  [2] Custom server URL")

        server_choice = click.prompt("\nChoose option [1-2]", type=int, default=1)

        if server_choice == 1:
            server = "http://localhost:5000"
            console.print(f"[green]✓[/green] Selected: {server}")
        else:
            server = click.prompt("Enter server URL", type=str, default=server or SERVER_URL)
            # Validate server URL format
            if not server.startswith(('http://', 'https://')):
                server = 'http://' + server
            console.print(f"[green]✓[/green] Server: {server}")

        # Check server availability with progress
        with console.status("[bold cyan]Checking server availability...[/bold cyan]"):
            try:
                ensure_server_available(server)
                console.print("[green]✓[/green] Server is reachable")
            except Exception as e:
                console.print(f"[red]✗[/red] Server unavailable: {e}")
                if not click.confirm("Continue anyway?"):
                    sys.exit(1)

        console.print()

        # Token input with enhanced guidance
        console.print("[bold]2. Authentication Token[/bold]")
        console.print("[dim]You can find your API token in the DeepScientist dashboard.[/dim]")
        console.print("[dim]It's typically a long string starting with 'sk-' or similar prefix.[/dim]\n")

        # Try to get token from environment first
        env_token = os.getenv('DEEPSCIENTIST_TOKEN')
        if env_token:
            console.print(f"[dim]Found token in environment variable (DEEPSCIENTIST_TOKEN)[/dim]")
            if click.confirm("Use the token from environment?"):
                token = env_token
                console.print("[green]✓[/green] Using environment token")

        if not token:
            token = click.prompt('Enter your API token', hide_input=True, confirmation_prompt=True)
            console.print("[green]✓[/green] Token entered")

        console.print()
    else:
        # Non-interactive mode
        server = resolve_server(server)
        ensure_server_available(server)

        console.print("[bold cyan]🔐 Login & Token Verification[/bold cyan]")
        console.print("─" * 60)
        console.print(f"Server: {server}\n")

        if not token:
            token = click.prompt('API token', hide_input=True)
    token = token.strip()
    server = resolve_server(server)

    with console.status("[bold cyan]Verifying token with backend...[/bold cyan]"):
        result = perform_cli_login(
            server,
            token,
            reason='interactive' if interactive else 'manual',
            save_token=True,
            silent=False
        )

    if result and result.get('supported') is False:
        legacy_login_flow(server, token, interactive)
        return

    if result and result.get('success'):
        data = result.get('data') or {}
        config_path = result.get('config_path')
        login_at = data.get('login_at')
        previous_login = data.get('previous_login_at')

        def format_ts(value):
            if not value:
                return "N/A"
            try:
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M:%S UTC')
            except Exception:
                return value

        if interactive:
            console.print("\n[bold green]🎉 Login Successful![/bold green]")
            console.print("─" * 60)
            if config_path:
                console.print(f"[green]✓[/green] Configuration saved to: {config_path}")
            if login_at:
                console.print(f"[green]✓[/green] Login recorded at: {format_ts(login_at)}")
            console.print("[green]✓[/green] You're now ready to use DeepScientist CLI\n")

            user_info = data.get('user', {})
            if user_info:
                username = user_info.get('username', 'Unknown')
                user_type = user_info.get('user_type', 'normal')
                api_verified = user_info.get('api_verified', False)
                email = user_info.get('email', 'Not set')

                console.print("[bold]👤 User Information:[/bold]")
                console.print(f"  Username: {username}")
                console.print(f"  User Type: {user_type}")
                console.print(f"  Email: {email}")
                if previous_login:
                    console.print(f"  Previous Login: {format_ts(previous_login)}")
                if api_verified:
                    console.print(f"  API Status: [green]✓ Verified[/green]")
                else:
                    console.print(f"  API Status: [yellow]⚠ Not configured[/yellow]")

                console.print("\n[bold]🚀 Next Steps:[/bold]")
                console.print("  • [cyan]deepscientist whoami[/cyan] - Show your profile")
                console.print("  • [cyan]deepscientist list[/cyan] - List your tasks")
                if not api_verified:
                    console.print("  • [cyan]deepscientist verify-api[/cyan] - Configure API settings")
                console.print("  • [cyan]deepscientist submit /path/to/repo[/cyan] - Start research")
                console.print("  • [cyan]deepscientist --help[/cyan] - See all commands")

                if user_type == 'admin':
                    console.print("\n[dim]Admin tips: Use [cyan]deepscientist list --all[/cyan] to see all users' tasks[/dim]")

            console.print(f"\n[dim]Connected to: {server}[/dim]")
        else:
            console.print(f"\n[green]✓[/green] Token verified and saved to {config_path}")
            if login_at:
                console.print(f"[green]✓[/green] Login recorded at: {format_ts(login_at)}")
            if previous_login:
                console.print(f"[dim]Previous login:[/dim] {format_ts(previous_login)}")
            console.print("[green]✓[/green] You can now use CLI commands without passing --token\n")

        update_broadcast_cache(data.get('broadcasts'), display=True)
        return

    sys.exit(1)


def _parse_utc_iso(timestamp: Optional[str]) -> Optional[datetime]:
    """Parse a UTC ISO timestamp with optional Z suffix."""
    if not timestamp:
        return None
    try:
        value = timestamp.rstrip('Z') + ('+00:00' if timestamp.endswith('Z') else '')
        return datetime.fromisoformat(value)
    except Exception:
        return None


def fetch_task_usage_limits(server: str, token: str) -> Optional[Dict[str, Any]]:
    """Fetch current task usage statistics for the authenticated user."""
    try:
        response = requests.get(
            f"{server}/api/tasks/usage_limits",
            headers={'Authorization': f'Bearer {token}'},
            timeout=10
        )
    except requests.RequestException as exc:
        console.print(f"[yellow]⚠ Could not fetch usage limits: {exc}[/yellow]")
        return None

    if response.status_code == 200:
        return response.json()

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    error_detail = payload.get('error') or response.text
    console.print(f"[yellow]⚠ Failed to fetch usage limits (HTTP {response.status_code}): {error_detail}[/yellow]")
    return None


@cli.command()
@click.option('--server', default=None, help=f'Backend server URL (default: {SERVER_URL})')
@click.option('--token', default=None, help='API authentication token (overrides stored config)')
@click.option('--configure', is_flag=True, help='Configure API settings if verification fails')
def verify_api(server, token, configure):
    """Verify API configuration and show connection status"""
    print_banner()

    server = resolve_server(server)

    # Load token from config if not provided
    if not token:
        config = load_cli_config()
        if server in config.get('servers', {}):
            token = config['servers'][server].get('token')
        if not token:
            console.print("[red]✗ No token found. Please run 'deepscientist login' first.[/red]")
            sys.exit(1)

    console.print("[bold cyan]🔍 API Configuration Verification[/bold cyan]")
    console.print("─" * 60)
    console.print(f"Server: {server}")
    console.print(f"Token: {'*' * 20}{token[-4:] if len(token) > 4 else token}\n")

    # Check server availability
    with console.status("[bold cyan]Checking server availability...[/bold cyan]"):
        try:
            response = requests.get(f"{server}/health", timeout=5)
            if response.status_code == 200:
                console.print("[green]✓[/green] Server is reachable")
            else:
                console.print(f"[yellow]⚠ Server responded with HTTP {response.status_code}[/yellow]")
        except requests.exceptions.RequestException as e:
            console.print(f"[red]✗ Server unavailable:[/red] {e}")
            return

    # Verify API configuration
    with console.status("[bold cyan]Verifying API configuration...[/bold cyan]"):
        try:
            response = requests.post(
                f"{server}/api/auth/verify",
                json={'token': token},
                timeout=10
            )
        except requests.exceptions.RequestException as e:
            console.print(f"[red]✗ API verification failed:[/red] {e}")
            return

    status = response.status_code
    error_detail = extract_error_message(response)

    if status == 200:
        try:
            data = response.json()
            user_info = data.get('user', {})

            console.print("\n[bold green]✓ API Configuration Valid[/bold green]")
            console.print("─" * 60)

            # User information
            console.print("[bold]👤 User Account:[/bold]")
            console.print(f"  Username: {user_info.get('username', 'Unknown')}")
            console.print(f"  User Type: {user_info.get('user_type', 'normal')}")
            console.print(f"  Email: {user_info.get('email', 'Not set')}")

            # API status
            api_verified = user_info.get('api_verified', False)
            if api_verified:
                console.print(f"  API Status: [green]✓ Verified and configured[/green]")
            else:
                console.print(f"  API Status: [yellow]⚠ Not configured[/yellow]")

            # Task limits and usage
            console.print("\n[bold]📊 Usage Information:[/bold]")
            try:
                usage_info = fetch_task_usage_limits(server, token)
                if usage_info:
                    limits = usage_info.get('limits', {})
                    usage = usage_info.get('usage', {})

                    console.print(f"  Task Limit: {limits.get('max_tasks_per_user', 'Unlimited')}")
                    console.print(f"  Current Tasks: {usage.get('active_tasks', 'Unknown')}")
                    console.print(f"  Total Submitted: {usage.get('total_submitted', 'Unknown')}")
                else:
                    console.print("  [dim]Usage information unavailable[/dim]")
            except Exception as e:
                console.print(f"  [dim]Could not fetch usage info: {e}[/dim]")

            # Connection details
            console.print(f"\n[bold]🔗 Connection Details:[/bold]")
            console.print(f"  Endpoint: {server}")
            console.print(f"  Authentication: [green]✓ Valid[/green]")
            console.print(f"  Last Verified: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Recommendations
            if not api_verified:
                console.print(f"\n[bold]💡 Recommendations:[/bold]")
                console.print("  • Configure API settings to enable LLM features")
                console.print("  • Use [cyan]deepscientist verify-api --configure[/cyan] to set up API")
                console.print("  • Or configure through the web dashboard")

        except Exception as e:
            console.print(f"[yellow]⚠ Unexpected response format: {e}[/yellow]")
            console.print("[green]✓ Token appears valid, but user details unavailable[/green]")

    elif status in (401, 403):
        console.print(f"[red]✗ Authentication failed:[/red] {error_detail or 'Invalid token'}")
        console.print("[yellow]Suggestions:[/yellow]")
        console.print("  • Check if the token is correct")
        console.print("  • Run [cyan]deepscientist login[/cyan] to update your token")
        console.print("  • Contact your administrator if the problem persists")

        if configure:
            console.print(f"\n[dim]To configure new API settings, use:[/dim]")
            console.print(f"  [cyan]deepscientist login --interactive[/cyan]")

    elif 400 <= status < 500:
        console.print(f"[red]✗ Request error (HTTP {status}):[/red] {error_detail}")
        console.print("[yellow]Suggestions:[/yellow]")
        console.print("  • Check the server URL")
        console.print("  • Verify the API endpoint is correct")
        console.print("  • Check if the server is running the correct version")

    else:
        console.print(f"[red]✗ Server error (HTTP {status}):[/red] {error_detail}")
        console.print("[yellow]Suggestions:[/yellow]")
        console.print("  • The server may be experiencing issues")
        console.print("  • Try again later")
        console.print("  • Contact the system administrator")


@cli.command()
@click.option('--server', default=None, help=f'Backend server URL (default: {SERVER_URL})')
@click.option('--token', default=None, help='API authentication token (overrides stored config)')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def whoami(server, token, output_format):
    """Display current user information and permissions"""
    print_banner()

    server = resolve_server(server)

    # Load token from config if not provided
    if not token:
        config = load_cli_config()
        if server in config.get('servers', {}):
            token = config['servers'][server].get('token')
        if not token:
            console.print("[red]✗ No token found. Please run 'deepscientist login' first.[/red]")
            sys.exit(1)

    console.print("[bold cyan]👤 User Profile Information[/bold cyan]")
    console.print("─" * 60)

    # Get user information
    with console.status("[bold cyan]Fetching user profile...[/bold cyan]"):
        try:
            response = requests.post(
                f"{server}/api/auth/verify",
                json={'token': token},
                timeout=10
            )
        except requests.exceptions.RequestException as e:
            console.print(f"[red]✗ Failed to fetch user information:[/red] {e}")
            return

    if response.status_code != 200:
        error_detail = extract_error_message(response)
        console.print(f"[red]✗ Authentication failed:[/red] {error_detail}")
        return

    try:
        data = response.json()
        user_info = data.get('user', {})
    except Exception:
        console.print("[red]✗ Unexpected response format[/red]")
        return

    if output_format == 'json':
        # Output as JSON
        console.print(json.dumps({
            'user': {
                'username': user_info.get('username', 'Unknown'),
                'user_type': user_info.get('user_type', 'normal'),
                'email': user_info.get('email', 'Not set'),
                'api_verified': user_info.get('api_verified', False),
                'created_at': user_info.get('created_at'),
                'last_login': user_info.get('last_login'),
                'role': user_info.get('role', 'user')
            },
            'server': server,
            'token_last_characters': token[-4:] if len(token) > 4 else token
        }, indent=2))
        return

    # Display as table format
    from rich.table import Table
    from rich.panel import Panel

    # Basic information table
    basic_table = Table(title="Basic Information", show_header=False, box=None)
    basic_table.add_column("Field", style="bold cyan")
    basic_table.add_column("Value")

    basic_table.add_row("Username", user_info.get('username', 'Unknown'))
    basic_table.add_row("User Type", user_info.get('user_type', 'normal'))
    basic_table.add_row("Email", user_info.get('email', 'Not set'))
    basic_table.add_row("Role", user_info.get('role', 'user'))

    console.print(basic_table)
    console.print()

    # API configuration table
    api_table = Table(title="API Configuration", show_header=False, box=None)
    api_table.add_column("Field", style="bold cyan")
    api_table.add_column("Value")

    api_verified = user_info.get('api_verified', False)
    api_status = "[green]✓ Verified and configured[/green]" if api_verified else "[yellow]⚠ Not configured[/yellow]"
    api_table.add_row("API Status", api_status)

    console.print(api_table)
    console.print()

    # Permissions table
    permissions_table = Table(title="Permissions & Capabilities", show_header=False, box=None)
    permissions_table.add_column("Permission", style="bold cyan")
    permissions_table.add_column("Status")

    user_type = user_info.get('user_type', 'normal')
    role = user_info.get('role', 'user')

    # User type permissions
    permissions_table.add_row("Submit Research Tasks", "[green]✓ Yes[/green]")
    permissions_table.add_row("View Own Tasks", "[green]✓ Yes[/green]")
    permissions_table.add_row("Manage Own Tasks", "[green]✓ Yes[/green]")

    if user_type == 'supported':
        permissions_table.add_row("Custom API Configuration", "[green]✓ Yes[/green]")
    else:
        permissions_table.add_row("Custom API Configuration", "[red]✗ No[/red]")

    if role == 'admin':
        permissions_table.add_row("View All Users' Tasks", "[green]✓ Yes[/green]")
        permissions_table.add_row("Manage All Tasks", "[green]✓ Yes[/green]")
        permissions_table.add_row("Create/Manage Users", "[green]✓ Yes[/green]")
        permissions_table.add_row("System Administration", "[green]✓ Yes[/green]")
    else:
        permissions_table.add_row("View All Users' Tasks", "[red]✗ No[/red]")
        permissions_table.add_row("Manage All Tasks", "[red]✗ No[/red]")
        permissions_table.add_row("Create/Manage Users", "[red]✗ No[/red]")
        permissions_table.add_row("System Administration", "[red]✗ No[/red]")

    console.print(permissions_table)
    console.print()

    # Usage information
    with console.status("[bold cyan]Fetching usage statistics...[/bold cyan]"):
        try:
            usage_info = fetch_task_usage_limits(server, token)
        except Exception:
            usage_info = None

    if usage_info:
        usage_table = Table(title="Usage Statistics", show_header=False, box=None)
        usage_table.add_column("Metric", style="bold cyan")
        usage_table.add_column("Value")

        limits = usage_info.get('limits', {})
        usage = usage_info.get('usage', {})

        usage_table.add_row("Task Limit", str(limits.get('max_tasks_per_user', 'Unlimited')))
        usage_table.add_row("Current Tasks", str(usage.get('active_tasks', 'Unknown')))
        usage_table.add_row("Total Submitted", str(usage.get('total_submitted', 'Unknown')))
        usage_table.add_row("Total Completed", str(usage.get('total_completed', 'Unknown')))

        console.print(usage_table)
        console.print()

    # Connection information
    connection_table = Table(title="Connection Information", show_header=False, box=None)
    connection_table.add_column("Field", style="bold cyan")
    connection_table.add_column("Value")

    connection_table.add_row("Server", server)
    connection_table.add_row("Token", f"{'*' * 20}{token[-4:] if len(token) > 4 else token}")
    connection_table.add_row("Config File", os.path.expanduser('~/.deepscientist/config.json'))

    console.print(connection_table)

    # Account status message
    console.print()
    if api_verified:
        console.print("[bold green]✓ Your account is fully configured and ready to use![/bold green]")
    else:
        console.print("[bold yellow]⚠ Your account needs API configuration[/bold yellow]")
        console.print("[dim]Run [cyan]deepscientist verify-api[/cyan] to check your API status[/dim]")
        console.print("[dim]Run [cyan]deepscientist login --interactive[/cyan] to configure API settings[/dim]")


@cli.command()
@click.option('--server', default=None, help=f'Backend server URL (default: {SERVER_URL})')
@click.option('--token', default=None, help='API authentication token (overrides stored config)')
@click.option('--detailed', is_flag=True, help='Show detailed system information')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
def status(server, token, detailed, output_format):
    """Check connection status and system health"""
    print_banner()

    server = resolve_server(server)

    # Load token from config if not provided (optional for status command)
    if not token:
        config = load_cli_config()
        if server in config.get('servers', {}):
            token = config['servers'][server].get('token')

    console.print("[bold cyan]🔍 System Status & Health Check[/bold cyan]")
    console.print("─" * 60)
    console.print(f"Checking server: {server}\n")

    # Collect status information
    status_info = {
        'server': server,
        'server_reachable': False,
        'api_available': False,
        'authenticated': False,
        'user_info': None,
        'response_times': {},
        'system_info': None,
        'timestamp': datetime.now().isoformat()
    }

    # Check server reachability
    with console.status("[bold cyan]Checking server connectivity...[/bold cyan]"):
        try:
            start_time = time.time()
            response = requests.get(f"{server}/health", timeout=5)
            response_time = time.time() - start_time
            status_info['response_times']['health'] = round(response_time * 1000, 2)

            if response.status_code == 200:
                status_info['server_reachable'] = True
                console.print(f"[green]✓[/green] Server reachable ({response_time:.2f}s)")

                # Parse health response
                try:
                    health_data = response.json()
                    status_info['system_info'] = health_data
                except:
                    health_data = None

                if health_data:
                    console.print(f"[green]✓[/green] Health endpoint available")
                    if 'version' in health_data:
                        console.print(f"[dim]  Version: {health_data['version']}[/dim]")
                    if 'status' in health_data:
                        console.print(f"[dim]  System Status: {health_data['status']}[/dim]")
                else:
                    console.print(f"[yellow]⚠[/yellow] Health endpoint returned invalid data")
            else:
                console.print(f"[yellow]⚠[/yellow] Server responded with HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            console.print(f"[red]✗[/red] Server unreachable: {e}")

    # Check API availability
    console.print()
    with console.status("[bold cyan]Checking API availability...[/bold cyan]"):
        try:
            start_time = time.time()
            response = requests.get(f"{server}/api/auth/verify", timeout=5)
            response_time = time.time() - start_time
            status_info['response_times']['api'] = round(response_time * 1000, 2)

            if response.status_code in [200, 401, 403]:
                status_info['api_available'] = True
                console.print(f"[green]✓[/green] API available ({response_time:.2f}s)")
            else:
                console.print(f"[yellow]⚠[/yellow] API responded with HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            console.print(f"[red]✗[/red] API unavailable: {e}")

    # Check authentication if token is available
    if token:
        console.print()
        with console.status("[bold cyan]Checking authentication...[/bold cyan]"):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{server}/api/auth/verify",
                    json={'token': token},
                    timeout=10
                )
                response_time = time.time() - start_time
                status_info['response_times']['auth'] = round(response_time * 1000, 2)

                if response.status_code == 200:
                    status_info['authenticated'] = True
                    console.print(f"[green]✓[/green] Authentication valid ({response_time:.2f}s)")

                    try:
                        data = response.json()
                        user_info = data.get('user', {})
                        status_info['user_info'] = user_info

                        username = user_info.get('username', 'Unknown')
                        user_type = user_info.get('user_type', 'normal')
                        console.print(f"[dim]  User: {username} ({user_type})[/dim]")

                        if user_info.get('api_verified'):
                            console.print(f"[dim]  API Status: Configured[/dim]")
                        else:
                            console.print(f"[dim]  API Status: Not configured[/dim]")
                    except:
                        console.print(f"[dim]  User info unavailable[/dim]")
                else:
                    console.print(f"[red]✗[/red] Authentication failed (HTTP {response.status_code})")
            except requests.exceptions.RequestException as e:
                console.print(f"[red]✗[/red] Authentication check failed: {e}")
    else:
        console.print()
        console.print("[dim]ℹ️  No token provided - authentication check skipped[/dim]")
        console.print("[dim]   Use --token to check authentication status[/dim]")

    # Detailed information
    if detailed:
        console.print()
        console.print("[bold]📊 Detailed System Information[/bold]")
        console.print("─" * 40)

        from rich.table import Table

        # Response times table
        times_table = Table(title="Response Times", show_header=True, box=None)
        times_table.add_column("Endpoint", style="bold cyan")
        times_table.add_column("Time (ms)")
        times_table.add_column("Status")

        for endpoint, time_ms in status_info['response_times'].items():
            status_color = "[green]Good[/green]" if time_ms < 1000 else "[yellow]Slow[/green]" if time_ms < 3000 else "[red]Very Slow[/red]"
            times_table.add_row(endpoint.capitalize(), str(time_ms), status_color)

        console.print(times_table)

        # System information table
        if status_info['system_info']:
            console.print()
            system_table = Table(title="System Information", show_header=False, box=None)
            system_table.add_column("Field", style="bold cyan")
            system_table.add_column("Value")

            for key, value in status_info['system_info'].items():
                system_table.add_row(key.replace('_', ' ').title(), str(value))

            console.print(system_table)

        # Connection summary
        console.print()
        summary_table = Table(title="Connection Summary", show_header=False, box=None)
        summary_table.add_column("Component", style="bold cyan")
        summary_table.add_column("Status")

        summary_table.add_row("Server", "[green]✓ Reachable[/green]" if status_info['server_reachable'] else "[red]✗ Unreachable[/red]")
        summary_table.add_row("API", "[green]✓ Available[/green]" if status_info['api_available'] else "[red]✗ Unavailable[/red]")
        summary_table.add_row("Authentication", "[green]✓ Valid[/green]" if status_info['authenticated'] else "[red]✗ Invalid/Not checked[/red]")

        console.print(summary_table)

    # JSON output
    if output_format == 'json':
        console.print()
        console.print("[bold]JSON Output:[/bold]")
        console.print(json.dumps(status_info, indent=2))
        return

    # Overall status summary
    console.print()
    if status_info['server_reachable'] and status_info['api_available']:
        if status_info['authenticated']:
            console.print("[bold green]✅ System Status: All components operational[/bold green]")
        else:
            console.print("[bold yellow]⚠️  System Status: Operational, authentication needed[/bold yellow]")
    elif status_info['server_reachable']:
        console.print("[bold red]❌ System Status: Server reachable, API issues detected[/bold red]")
    else:
        console.print("[bold red]❌ System Status: Server unreachable[/bold red]")

    # Recommendations
    console.print()
    console.print("[bold]💡 Recommendations:[/bold]")

    if not status_info['server_reachable']:
        console.print("  • Check if the server URL is correct")
        console.print("  • Verify network connectivity")
        console.print("  • Ensure the server is running")
    elif not status_info['api_available']:
        console.print("  • Check if the backend service is properly configured")
        console.print("  • Verify API endpoints are accessible")
    elif not status_info['authenticated']:
        console.print("  • Run [cyan]deepscientist login[/cyan] to authenticate")
        console.print("  • Check if your token is valid")
    else:
        console.print("  • System is functioning normally")
        if token:
            console.print("  • Run [cyan]deepscientist whoami[/cyan] for detailed user information")
            console.print("  • Run [cyan]deepscientist verify-api[/cyan] for API configuration status")


@cli.command()
@click.option('--task-id', help='Export specific task (exports all if not provided)')
@click.option('--format', 'export_format', type=click.Choice(['json', 'csv', 'markdown']), default='json', help='Export format')
@click.option('--output', '-o', type=click.Path(), help='Output file (prints to stdout if not provided)')
@click.option('--include-findings/--no-include-findings', default=True, help='Include findings in export (default: include)')
@click.option('--include-events/--no-include-events', default=False, help='Include detailed event logs (default: exclude)')
@click.option('--server', default=None, help=f'Backend server URL (default: {SERVER_URL})')
@click.option('--token', default=None, help='API authentication token (overrides stored config)')
def export(task_id, export_format, output, include_findings, include_events, server, token):
    """Export task results and findings to various formats"""
    print_banner()

    server = resolve_server(server)

    # Load token from config if not provided
    if not token:
        config = load_cli_config()
        if server in config.get('servers', {}):
            token = config['servers'][server].get('token')
        if not token:
            console.print("[red]✗ No token found. Please run 'deepscientist login' first.[/red]")
            sys.exit(1)

    console.print("[bold cyan]📤 Export Task Results[/bold cyan]")
    console.print("─" * 60)

    # Fetch data
    with console.status("[bold cyan]Fetching task data...[/bold cyan]"):
        try:
            if task_id:
                # Fetch specific task
                tasks_response = requests.get(
                    f"{server}/api/tasks/{task_id}",
                    headers={'Authorization': f'Bearer {token}'},
                    timeout=15
                )
                if tasks_response.status_code == 200:
                    tasks_data = {'tasks': [tasks_response.json()]}
                else:
                    console.print(f"[red]✗ Task {task_id} not found (HTTP {tasks_response.status_code})[/red]")
                    return
            else:
                # Fetch all tasks
                tasks_response = requests.get(
                    f"{server}/api/tasks",
                    headers={'Authorization': f'Bearer {token}'},
                    timeout=15
                )
                if tasks_response.status_code == 200:
                    tasks_data = tasks_response.json()
                else:
                    console.print(f"[red]✗ Failed to fetch tasks (HTTP {tasks_response.status_code})[/red]")
                    return

            # Fetch findings if requested
            if include_findings:
                if task_id:
                    findings_response = requests.get(
                        f"{server}/api/tasks/{task_id}/findings",
                        headers={'Authorization': f'Bearer {token}'},
                        timeout=15
                    )
                else:
                    findings_response = requests.get(
                        f"{server}/api/findings",
                        headers={'Authorization': f'Bearer {token}'},
                        timeout=15
                    )

                findings_data = []
                if findings_response.status_code == 200:
                    if task_id:
                        findings_data = findings_response.json()
                    else:
                        findings_data = findings_response.json().get('findings', [])
                else:
                    console.print(f"[yellow]⚠ Could not fetch findings (HTTP {findings_response.status_code})[/yellow]")
            else:
                findings_data = []

            # Fetch events if requested
            events_data = []
            if include_events:
                if task_id:
                    events_response = requests.get(
                        f"{server}/api/tasks/{task_id}/events",
                        headers={'Authorization': f'Bearer {token}'},
                        timeout=15
                    )
                    if events_response.status_code == 200:
                        events_data = events_response.json()
                    else:
                        console.print(f"[yellow]⚠ Could not fetch events (HTTP {events_response.status_code})[/yellow]")

        except requests.exceptions.RequestException as e:
            console.print(f"[red]✗ Failed to fetch data:[/red] {e}")
            return

    console.print(f"[green]✓[/green] Fetched {len(tasks_data.get('tasks', []))} task(s)")
    if include_findings:
        console.print(f"[green]✓[/green] Fetched {len(findings_data)} finding(s)")
    if include_events:
        console.print(f"[green]✓[/green] Fetched {len(events_data)} event(s)")

    # Process and format data
    export_data = {
        'export_metadata': {
            'timestamp': datetime.now().isoformat(),
            'server': server,
            'user': token[:8] + '...' if len(token) > 8 else token,
            'task_id': task_id,
            'format': export_format,
            'include_findings': include_findings,
            'include_events': include_events
        },
        'tasks': tasks_data.get('tasks', []),
        'findings': findings_data,
        'events': events_data
    }

    # Generate output based on format
    if export_format == 'json':
        output_content = json.dumps(export_data, indent=2, default=str)
    elif export_format == 'csv':
        output_content = format_as_csv(export_data, include_findings, include_events)
    elif export_format == 'markdown':
        output_content = format_as_markdown(export_data, include_findings, include_events)

    # Write output
    if output:
        try:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_content)
            console.print(f"\n[green]✓[/green] Export saved to: {output}")
        except Exception as e:
            console.print(f"[red]✗ Failed to write to file:[/red] {e}")
            console.print("\n[dim]Output content:[/dim]")
            print(output_content)
    else:
        console.print(f"\n[bold]Export Results ({export_format.upper()}):[/bold]")
        console.print("─" * 40)
        # Use print() instead of console.print() for raw content to avoid markdown parsing
        print(output_content)


def format_as_csv(data, include_findings, include_events):
    """Format export data as CSV"""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    # Write metadata as comments
    writer.writerow(['# Export Metadata'])
    metadata = data['export_metadata']
    writer.writerow(['# Timestamp', metadata['timestamp']])
    writer.writerow(['# Server', metadata['server']])
    writer.writerow(['# Task ID', metadata.get('task_id', 'All')])
    writer.writerow([])

    # Write tasks
    if data['tasks']:
        writer.writerow(['# Tasks'])
        # Get all possible task fields
        task_fields = set()
        for task in data['tasks']:
            task_fields.update(task.keys())
        task_fields = sorted(task_fields)

        writer.writerow(task_fields)
        for task in data['tasks']:
            row = []
            for field in task_fields:
                value = task.get(field, '')
                # Handle nested objects
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, separators=(',', ':'))
                row.append(str(value))
            writer.writerow(row)
        writer.writerow([])

    # Write findings
    if include_findings and data['findings']:
        writer.writerow(['# Findings'])
        # Get all possible finding fields
        finding_fields = set()
        for finding in data['findings']:
            finding_fields.update(finding.keys())
        finding_fields = sorted(finding_fields)

        writer.writerow(finding_fields)
        for finding in data['findings']:
            row = []
            for field in finding_fields:
                value = finding.get(field, '')
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, separators=(',', ':'))
                row.append(str(value))
            writer.writerow(row)
        writer.writerow([])

    # Write events
    if include_events and data['events']:
        writer.writerow(['# Events'])
        # Get all possible event fields
        event_fields = set()
        for event in data['events']:
            event_fields.update(event.keys())
        event_fields = sorted(event_fields)

        writer.writerow(event_fields)
        for event in data['events']:
            row = []
            for field in event_fields:
                value = event.get(field, '')
                if isinstance(value, (dict, list)):
                    value = json.dumps(value, separators=(',', ':'))
                row.append(str(value))
            writer.writerow(row)

    return output.getvalue()


def format_as_markdown(data, include_findings, include_events):
    """Format export data as Markdown"""
    output = []
    metadata = data['export_metadata']

    # Header and metadata
    output.append(f"# DeepScientist Export Report")
    output.append("")
    output.append("## Export Information")
    output.append("")
    output.append(f"- **Timestamp:** {metadata['timestamp']}")
    output.append(f"- **Server:** {metadata['server']}")
    output.append(f"- **Task ID:** {metadata.get('task_id', 'All tasks')}")
    output.append(f"- **Format:** {metadata['format']}")
    output.append(f"- **Include Findings:** {metadata['include_findings']}")
    output.append(f"- **Include Events:** {metadata['include_events']}")
    output.append("")

    # Tasks section
    if data['tasks']:
        output.append("## Tasks")
        output.append("")
        for i, task in enumerate(data['tasks'], 1):
            output.append(f"### Task {i}: {task.get('title', 'Untitled')}")
            output.append("")
            output.append(f"**ID:** `{task.get('id', 'N/A')}`")
            output.append(f"**Status:** {task.get('status', 'Unknown')}")
            output.append(f"**Created:** {task.get('created_at', 'Unknown')}")
            output.append(f"**Updated:** {task.get('updated_at', 'Unknown')}")

            if task.get('description'):
                output.append(f"**Description:** {task['description']}")

            if task.get('research_query'):
                output.append(f"**Research Query:** {task['research_query']}")

            output.append("")

    # Findings section
    if include_findings and data['findings']:
        output.append("## Findings")
        output.append("")
        for i, finding in enumerate(data['findings'], 1):
            output.append(f"### Finding {i}: {finding.get('title', 'Untitled')}")
            output.append("")
            output.append(f"**Task ID:** `{finding.get('task_id', 'N/A')}`")
            output.append(f"**Score:** {finding.get('score', 'N/A')}")
            output.append(f"**Cycle:** {finding.get('cycle_number', 'N/A')}")
            output.append(f"**Created:** {finding.get('created_at', 'Unknown')}")

            if finding.get('content'):
                output.append("")
                output.append("**Content:**")
                output.append("")
                content = finding['content']
                # Convert to markdown-friendly format
                if isinstance(content, dict):
                    content = json.dumps(content, indent=2)
                output.append(content)

            output.append("")

    # Events section
    if include_events and data['events']:
        output.append("## Event Timeline")
        output.append("")
        for i, event in enumerate(data['events'], 1):
            output.append(f"### Event {i}: {event.get('title', 'Untitled')}")
            output.append("")
            output.append(f"**Type:** {event.get('event_type', 'Unknown')}")
            output.append(f"**Timestamp:** {event.get('timestamp', 'Unknown')}")
            output.append(f"**Cycle:** {event.get('cycle_number', 'N/A')}")

            if event.get('content'):
                output.append("")
                output.append("**Details:**")
                output.append("")
                content = event['content']
                if isinstance(content, dict):
                    content = json.dumps(content, indent=2)
                output.append(content)

            output.append("")

    return "\n".join(output)


@cli.command()
@click.option('--server', default=None, help=f'Backend server URL (default: {SERVER_URL})')
@click.option('--token', default=None, help='API authentication token (overrides stored config)')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.option('--period', default='30d', help='Time period for stats (e.g., 7d, 30d, 90d)')
@click.option('--detailed', is_flag=True, help='Show detailed breakdown by task status')
def stats(server, token, output_format, period, detailed):
    """Display user statistics and task analysis"""
    print_banner()

    server = resolve_server(server)

    # Load token from config if not provided
    if not token:
        config = load_cli_config()
        if server in config.get('servers', {}):
            token = config['servers'][server].get('token')
        if not token:
            console.print("[red]✗ No token found. Please run 'deepscientist login' first.[/red]")
            sys.exit(1)

    console.print("[bold cyan]📊 User Statistics & Task Analysis[/bold cyan]")
    console.print("─" * 60)

    # Parse period
    days = 30  # default
    if period.endswith('d'):
        try:
            days = int(period[:-1])
        except ValueError:
            console.print(f"[yellow]⚠ Invalid period format: {period}. Using 30d.[/yellow]")
    else:
        console.print(f"[yellow]⚠ Invalid period format: {period}. Using 30d.[/yellow]")

    from_date = datetime.now() - timedelta(days=days)
    console.print(f"Analysis period: {from_date.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')} ({days} days)\n")

    # Collect statistics
    stats_data = {
        'period_days': days,
        'from_date': from_date.isoformat(),
        'to_date': datetime.now().isoformat(),
        'tasks': {},
        'findings': {},
        'usage': None,
        'user_info': None
    }

    # Get user information
    with console.status("[bold cyan]Fetching user information...[/bold cyan]"):
        try:
            user_response = requests.post(
                f"{server}/api/auth/verify",
                json={'token': token},
                timeout=10
            )
            if user_response.status_code == 200:
                user_data = user_response.json()
                stats_data['user_info'] = user_data.get('user', {})
        except requests.exceptions.RequestException:
            console.print("[yellow]⚠ Could not fetch user information[/yellow]")

    # Get tasks
    with console.status("[bold cyan]Fetching task statistics...[/bold cyan]"):
        try:
            tasks_response = requests.get(
                f"{server}/api/tasks",
                headers={'Authorization': f'Bearer {token}'},
                timeout=15
            )
            if tasks_response.status_code == 200:
                all_tasks = tasks_response.json().get('tasks', [])

                # Filter tasks by date if needed
                if days > 0:
                    tasks = []
                    for task in all_tasks:
                        created_at = task.get('created_at')
                        if created_at:
                            try:
                                task_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                if task_date >= from_date:
                                    tasks.append(task)
                            except:
                                # If date parsing fails, include the task
                                tasks.append(task)
                else:
                    tasks = all_tasks

                # Analyze tasks
                stats_data['tasks'] = analyze_tasks(tasks, detailed)

        except requests.exceptions.RequestException as e:
            console.print(f"[yellow]⚠ Could not fetch tasks: {e}[/yellow]")

    # Get findings
    with console.status("[bold cyan]Fetching findings statistics...[/bold cyan]"):
        try:
            findings_response = requests.get(
                f"{server}/api/findings",
                headers={'Authorization': f'Bearer {token}'},
                timeout=15
            )
            if findings_response.status_code == 200:
                all_findings = findings_response.json().get('findings', [])

                # Filter findings by date if needed
                if days > 0:
                    findings = []
                    for finding in all_findings:
                        created_at = finding.get('created_at')
                        if created_at:
                            try:
                                finding_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                if finding_date >= from_date:
                                    findings.append(finding)
                            except:
                                findings.append(finding)
                else:
                    findings = all_findings

                # Analyze findings
                stats_data['findings'] = analyze_findings(findings)

        except requests.exceptions.RequestException as e:
            console.print(f"[yellow]⚠ Could not fetch findings: {e}[/yellow]")

    # Get usage information
    with console.status("[bold cyan]Fetching usage statistics...[/bold cyan]"):
        try:
            usage_info = fetch_task_usage_limits(server, token)
            if usage_info:
                stats_data['usage'] = usage_info
        except Exception:
            console.print("[yellow]⚠ Could not fetch usage information[/yellow]")

    # Output results
    if output_format == 'json':
        console.print(json.dumps(stats_data, indent=2, default=str))
        return

    # Display as tables
    from rich.table import Table

    # User information
    if stats_data['user_info']:
        user_info = stats_data['user_info']
        console.print("[bold]👤 User Overview[/bold]")
        user_table = Table(show_header=False, box=None)
        user_table.add_column("Field", style="bold cyan")
        user_table.add_column("Value")

        user_table.add_row("Username", user_info.get('username', 'Unknown'))
        user_table.add_row("User Type", user_info.get('user_type', 'normal'))
        user_table.add_row("Email", user_info.get('email', 'Not set'))
        user_table.add_row("API Status", "[green]✓ Configured[/green]" if user_info.get('api_verified') else "[yellow]⚠ Not configured[/yellow]")

        console.print(user_table)
        console.print()

    # Task statistics
    tasks_stats = stats_data['tasks']
    console.print("[bold]📋 Task Statistics[/bold]")

    task_summary_table = Table(show_header=True, box=None)
    task_summary_table.add_column("Metric", style="bold cyan")
    task_summary_table.add_column("Count", justify="right")
    task_summary_table.add_column("Percentage", justify="right")

    total_tasks = tasks_stats.get('total', 0)
    task_summary_table.add_row("Total Tasks", str(total_tasks), "100.0%")

    if total_tasks > 0:
        for status, count in tasks_stats.get('by_status', {}).items():
            percentage = (count / total_tasks) * 100
            task_summary_table.add_row(status.capitalize(), str(count), f"{percentage:.1f}%")

    console.print(task_summary_table)

    # Detailed task breakdown
    if detailed and tasks_stats.get('by_status'):
        console.print()
        console.print("[bold]Task Status Breakdown[/bold]")

        for status, count in tasks_stats.get('by_status', {}).items():
            if count > 0:
                console.print(f"  {status.capitalize()}: {count} tasks")

    console.print()

    # Findings statistics
    findings_stats = stats_data['findings']
    console.print("[bold]🔬 Findings Statistics[/bold]")

    findings_table = Table(show_header=True, box=None)
    findings_table.add_column("Metric", style="bold cyan")
    findings_table.add_column("Value", justify="right")

    findings_table.add_row("Total Findings", str(findings_stats.get('total', 0)))
    findings_table.add_row("Average Score", f"{findings_stats.get('avg_score', 0):.2f}")
    findings_table.add_row("Highest Score", f"{findings_stats.get('max_score', 0):.2f}")
    findings_table.add_row("Unique Tasks", str(findings_stats.get('unique_tasks', 0)))

    console.print(findings_table)

    console.print()

    # Usage limits
    if stats_data['usage']:
        usage = stats_data['usage']
        limits = usage.get('limits', {})
        current_usage = usage.get('usage', {})

        console.print("[bold]⚖️ Usage Limits[/bold]")

        usage_table = Table(show_header=True, box=None)
        usage_table.add_column("Metric", style="bold cyan")
        usage_table.add_column("Current", justify="right")
        usage_table.add_column("Limit", justify="right")
        usage_table.add_column("Usage", justify="right")

        max_tasks = limits.get('max_tasks_per_user', 'Unlimited')
        current_tasks = current_usage.get('active_tasks', 0)

        if max_tasks == 'Unlimited':
            usage_table.add_row("Active Tasks", str(current_tasks), "Unlimited", "N/A")
        else:
            usage_percent = (current_tasks / max_tasks) * 100 if max_tasks > 0 else 0
            usage_style = "[green]" if usage_percent < 80 else "[yellow]" if usage_percent < 95 else "[red]"
            usage_table.add_row("Active Tasks", str(current_tasks), str(max_tasks), f"{usage_style}{usage_percent:.1f}%[/{usage_style}]")

        usage_table.add_row("Total Submitted", str(current_usage.get('total_submitted', 0)), "N/A", "N/A")
        usage_table.add_row("Total Completed", str(current_usage.get('total_completed', 0)), "N/A", "N/A")

        console.print(usage_table)

    # Recommendations
    console.print()
    console.print("[bold]💡 Insights & Recommendations[/bold]")

    recommendations = generate_recommendations(stats_data)
    for i, recommendation in enumerate(recommendations, 1):
        console.print(f"  {i}. {recommendation}")


def analyze_tasks(tasks, detailed):
    """Analyze task data and return statistics"""
    analysis = {
        'total': len(tasks),
        'by_status': {},
        'by_date': {},
        'recent': []
    }

    if not tasks:
        return analysis

    # Count by status
    for task in tasks:
        status = task.get('status', 'unknown')
        analysis['by_status'][status] = analysis['by_status'].get(status, 0) + 1

        # Group by date (created_at)
        created_at = task.get('created_at')
        if created_at:
            try:
                task_date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).date().isoformat()
                analysis['by_date'][task_date] = analysis['by_date'].get(task_date, 0) + 1
            except:
                pass

    # Sort recent tasks
    analysis['recent'] = sorted(
        tasks,
        key=lambda x: x.get('created_at', ''),
        reverse=True
    )[:5] if detailed else []

    return analysis


def analyze_findings(findings):
    """Analyze findings data and return statistics"""
    analysis = {
        'total': len(findings),
        'avg_score': 0,
        'max_score': 0,
        'min_score': float('inf'),
        'unique_tasks': set(),
        'by_score_range': {
            'High (0.8-1.0)': 0,
            'Medium (0.5-0.8)': 0,
            'Low (0.0-0.5)': 0
        }
    }

    if not findings:
        analysis['min_score'] = 0
        return analysis

    scores = []
    for finding in findings:
        score = finding.get('score', 0)
        scores.append(score)

        analysis['max_score'] = max(analysis['max_score'], score)
        analysis['min_score'] = min(analysis['min_score'], score)

        task_id = finding.get('task_id')
        if task_id:
            analysis['unique_tasks'].add(task_id)

        # Score ranges
        if score >= 0.8:
            analysis['by_score_range']['High (0.8-1.0)'] += 1
        elif score >= 0.5:
            analysis['by_score_range']['Medium (0.5-0.8)'] += 1
        else:
            analysis['by_score_range']['Low (0.0-0.5)'] += 1

    if scores:
        analysis['avg_score'] = sum(scores) / len(scores)

    analysis['unique_tasks'] = len(analysis['unique_tasks'])

    return analysis


def generate_recommendations(stats_data):
    """Generate insights and recommendations based on statistics"""
    recommendations = []

    tasks_stats = stats_data['tasks']
    findings_stats = stats_data['findings']
    usage_info = stats_data['usage']

    # Task recommendations
    total_tasks = tasks_stats.get('total', 0)
    if total_tasks == 0:
        recommendations.append("Start your first research task with [cyan]deepscientist submit /path/to/repo[/cyan]")
    else:
        completed_tasks = tasks_stats.get('by_status', {}).get('completed', 0)
        running_tasks = tasks_stats.get('by_status', {}).get('running', 0)

        if completed_tasks == 0 and running_tasks > 0:
            recommendations.append("You have running tasks. Monitor their progress with [cyan]deepscientist list[/cyan]")
        elif completed_tasks > 0:
            recommendations.append(f"Great! You've completed {completed_tasks} task(s). Consider exporting results with [cyan]deepscientist export[/cyan]")

    # Findings recommendations
    total_findings = findings_stats.get('total', 0)
    if total_findings > 0:
        avg_score = findings_stats.get('avg_score', 0)
        if avg_score < 0.5:
            recommendations.append("Consider refining your research queries to improve finding quality")
        else:
            recommendations.append(f"Excellent! Your average finding score is {avg_score:.2f}")

    # Usage recommendations
    if usage_info:
        limits = usage_info.get('limits', {})
        current_usage = usage_info.get('usage', {})
        max_tasks = limits.get('max_tasks_per_user')
        current_tasks = current_usage.get('active_tasks', 0)

        if max_tasks and isinstance(max_tasks, int):
            usage_percent = (current_tasks / max_tasks) * 100 if max_tasks > 0 else 0
            if usage_percent > 80:
                recommendations.append("You're approaching your task limit. Consider completing or pausing some tasks")

    # API configuration
    user_info = stats_data.get('user_info', {})
    if user_info and not user_info.get('api_verified'):
        recommendations.append("Configure your API settings to enable full research capabilities")

    if not recommendations:
        recommendations.append("Continue exploring! Use [cyan]deepscientist --help[/cyan] to see all available commands")

    return recommendations


def terminate_all_tasks_for_user(server: str, token: str) -> Optional[Dict[str, Any]]:
    """Request backend to terminate all active tasks for the current user."""
    try:
        response = requests.post(
            f"{server}/api/tasks/terminate_all",
            headers={'Authorization': f'Bearer {token}'},
            timeout=20
        )
    except requests.RequestException as exc:
        console.print(f"[red]✗ Failed to terminate tasks:[/red] {exc}")
        return None

    if response.status_code == 200:
        return response.json()

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    error_detail = payload.get('error') or response.text
    console.print(f"[red]✗ Termination request failed (HTTP {response.status_code}):[/red] {error_detail}")
    return None


@cli.command()
@click.argument('codebase_path', type=click.Path(exists=True))
@click.option('--token', help='API authentication token')
@click.option('--server', default=None, help='Backend server URL')
@click.option('--query', '-q', default=None, help='Specify research exploration direction (optional)')
@click.option('--gpu', default='0', help='GPU device ID for CUDA operations (default: 0)')
@click.option('--baseline', '-b', default=None, type=click.Choice(['H', 'Y'], case_sensitive=False),
              help='Baseline code upload: H=already uploaded, Y=upload (default: ask or use config)')
def submit(codebase_path, token, server, query, gpu, baseline):
    """Submit a new research task"""
    import sys  # Local import to ensure availability even in embedded copies

    print_banner()

    # Show helpful info about configuration
    console.print("[dim]💡 Tip: You can view/manage your settings with 'deepscientist-cli config'[/dim]")
    console.print()

    # Validate codebase
    validation = validate_codebase(codebase_path)
    if not validation:
        console.print("[red]✗ Submission aborted due to validation errors.[/red]")
        sys.exit(1)

    task_status['token_count'] = validation['token_count']
    task_status['query'] = query
    task_status['cuda_device'] = gpu
    task_status['codebase_path'] = str(Path(codebase_path).resolve())
    task_status['error'] = None
    task_status['error_code'] = None
    task_status['error_details'] = None

    # Get or ask for validation frequency setting
    validation_frequency = get_validation_frequency()

    # If not set, ask user with interactive menu
    if validation_frequency is None:
        console.print("\n" + "─" * 80)
        console.print("[bold cyan]🔍 Validation Frequency Setting[/bold cyan]")
        console.print("─" * 80)

        # Try interactive selection with prompt_toolkit
        freq_options = [
            {'value': 'high', 'label': 'High   - Validate every cycle (most thorough)', 'color': 'green'},
            {'value': 'medium', 'label': 'Medium - Validate every 3 cycles (balanced, default)', 'color': 'cyan'},
            {'value': 'low', 'label': 'Low    - Validate every 10 cycles (faster)', 'color': 'yellow'},
            {'value': 'auto', 'label': 'Auto   - Let the AI decide when to validate', 'color': 'blue'}
        ]

        selected_freq = None

        if PROMPT_TOOLKIT_AVAILABLE and KeyBindings is not None:
            try:
                from prompt_toolkit.application import Application
                from prompt_toolkit.layout import Layout
                from prompt_toolkit.layout.containers import HSplit, Window
                from prompt_toolkit.layout.controls import FormattedTextControl
                from prompt_toolkit.styles import Style as PTStyle

                selected = {'index': 1}  # Default to medium

                def body_text():
                    lines = []
                    for idx, opt in enumerate(freq_options):
                        pointer = '➜ ' if idx == selected['index'] else '  '
                        style = f'class:pointer_{opt["color"]}' if idx == selected['index'] else f'class:item_{opt["color"]}'
                        lines.append((style, f"{pointer}{opt['label']}\n"))
                    return lines

                def instructions():
                    return [
                        ('class:instruction', 'Use ↑/↓ to choose, Enter to confirm, Esc to use default (medium).')
                    ]

                kb = KeyBindings()

                @kb.add('up')
                def _(event):
                    selected['index'] = (selected['index'] - 1) % len(freq_options)
                    event.app.invalidate()

                @kb.add('down')
                def _(event):
                    selected['index'] = (selected['index'] + 1) % len(freq_options)
                    event.app.invalidate()

                @kb.add('enter')
                def _(event):
                    event.app.exit(result=freq_options[selected['index']]['value'])

                @kb.add('escape')
                @kb.add('c-c')
                def _(event):
                    event.app.exit(result='medium')  # Default

                body_window = Window(content=FormattedTextControl(body_text), always_hide_cursor=True)
                instruction_window = Window(height=1, content=FormattedTextControl(instructions), always_hide_cursor=True)

                layout = Layout(HSplit([
                    Window(height=1, content=FormattedTextControl(lambda: [('class:title', 'Choose validation frequency:')]), always_hide_cursor=True),
                    Window(height=1, char=' '),
                    body_window,
                    Window(height=1, char=' '),
                    instruction_window
                ]))

                style = PTStyle.from_dict({
                    'pointer_green': 'bold green',
                    'pointer_cyan': 'bold cyan',
                    'pointer_yellow': 'bold yellow',
                    'pointer_blue': 'bold blue',
                    'item_green': 'green',
                    'item_cyan': 'cyan',
                    'item_yellow': 'yellow',
                    'item_blue': 'blue',
                    'instruction': 'italic #64748b',
                    'title': 'bold'
                })

                app = Application(layout=layout, key_bindings=kb, style=style, full_screen=False)
                selected_freq = app.run()
            except Exception as e:
                # Fallback to simple selection
                console.print(f"[dim]Interactive menu unavailable, using text input[/dim]")
                selected_freq = None

        # Fallback: simple text selection
        if selected_freq is None:
            console.print("Choose how often to validate research ideas:")
            console.print("  [bold green]1.[/bold green] High   - Validate every cycle (most thorough)")
            console.print("  [bold cyan]2.[/bold cyan] Medium - Validate every 3 cycles (balanced, default)")
            console.print("  [bold yellow]3.[/bold yellow] Low    - Validate every 10 cycles (faster)")
            console.print("  [bold blue]4.[/bold blue] Auto   - Let the AI decide when to validate\n")

            while True:
                freq_choice = click.prompt("Your choice", type=str, default='2').strip()
                if freq_choice in ['1', 'high']:
                    selected_freq = 'high'
                    break
                elif freq_choice in ['2', 'medium']:
                    selected_freq = 'medium'
                    break
                elif freq_choice in ['3', 'low']:
                    selected_freq = 'low'
                    break
                elif freq_choice in ['4', 'auto']:
                    selected_freq = 'auto'
                    break
                else:
                    console.print("[yellow]Invalid choice. Please enter 1-4 or high/medium/low/auto.[/yellow]")

        validation_frequency = selected_freq

        # Ask if user wants to remember this choice
        remember = click.confirm("\nRemember this setting for future tasks?", default=True)
        if remember:
            save_validation_frequency(validation_frequency)
            console.print(f"[green]✓ Validation frequency saved as default: {validation_frequency}[/green]")
    else:
        console.print(f"[cyan]ℹ Using validation frequency: {validation_frequency}[/cyan]")

    task_status['validation_frequency'] = validation_frequency

    # Determine baseline upload decision BEFORE compression
    upload_baseline = False
    baseline_code_zip = None
    zip_size_mb = 0

    # Determine upload decision based on --baseline parameter or config
    if baseline:
        baseline_option = baseline.upper()
    else:
        # Check config default
        baseline_option = get_baseline_upload_default()

    if baseline_option:
        baseline_option = baseline_option.upper()
        if baseline_option == 'N':
            console.print("[dim]ℹ Baseline upload option 'N' is deprecated; treating as 'H'.[/dim]")
            baseline_option = 'H'
    else:
        baseline_option = None

    if baseline_option == 'H':
        # Already uploaded for this task
        upload_baseline = False
        task_status['claude_logs'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'system',
            'formatted': '📋 Baseline code already uploaded for this task'
        })
    elif baseline_option == 'Y':
        # User agreed to upload
        upload_baseline = True
        task_status['claude_logs'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'system',
            'formatted': '✓ Will upload baseline code to help the community'
        })
    else:
        # Ask user interactively
        console.print("\n" + "─" * 80)
        console.print("[bold cyan]📊 Baseline Code Upload[/bold cyan]")
        console.print("─" * 80)
        console.print("Would you like to share your baseline code with the research community?")
        console.print("This will help improve the platform and benefit other researchers.\n")
        console.print("Options:")
        console.print("  [bold green]Y[/bold green] - Yes, upload the baseline code")
        console.print("  [bold blue]H[/bold blue] - Already uploaded for this task\n")

        while True:
            choice = click.prompt("Your choice", type=str, default='H').upper()
            if choice in ['Y', 'H']:
                break
            console.print("[yellow]Invalid choice. Please enter Y or H.[/yellow]")

        if choice == 'Y':
            upload_baseline = True
            task_status['claude_logs'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'system',
                'formatted': '✓ User agreed to upload baseline code'
            })
        elif choice == 'H':
            upload_baseline = False
            task_status['claude_logs'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'system',
                'formatted': '📋 Baseline code already uploaded for this task'
            })

        console.print("─" * 80 + "\n")

    # Only compress if user agreed to upload
    if upload_baseline:
        import zipfile
        import tempfile
        import os

        # Add compression log to Claude Code Output
        task_status['claude_logs'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'system',
            'formatted': '📦 Compressing codebase...'
        })

        # Create temporary zip file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        zip_path = Path(temp_zip.name)
        temp_zip.close()

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                codebase_path_obj = Path(codebase_path)
                for root, dirs, files in os.walk(codebase_path):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(codebase_path_obj)
                        zipf.write(file_path, arcname)

            # Check zip file size
            zip_size = zip_path.stat().st_size
            zip_size_mb = zip_size / (1024 * 1024)

            if zip_size_mb <= 500:
                baseline_code_zip = zip_path
                task_status['claude_logs'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'system',
                    'formatted': f'✓ Codebase compressed successfully ({zip_size_mb:.2f} MB)'
                })
            else:
                zip_path.unlink()
                baseline_code_zip = None
                upload_baseline = False
                task_status['claude_logs'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'system',
                    'formatted': f'⚠ Codebase too large ({zip_size_mb:.2f} MB > 500 MB), skipping upload'
                })
        except Exception as e:
            task_status['claude_logs'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'system',
                'formatted': f'⚠ Failed to compress codebase: {e}'
            })
            if zip_path.exists():
                zip_path.unlink()
            baseline_code_zip = None
            upload_baseline = False

    # Resolve server and token
    server = resolve_server(server)
    ensure_server_available(server)
    token, token_source = resolve_token(token, server)

    if not token:
        console.print("\n[red]✗ Error:[/red] No authentication token available")
        console.print("[yellow]💡 Hint:[/yellow] Run `deepscientist-cli login --token <TOKEN>`")
        sys.exit(1)

    console.print(f"\n[cyan]Using token from {token_source}[/cyan]")

    usage_limits = fetch_task_usage_limits(server, token)
    if usage_limits:
        def _as_int(value, default=0):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        max_concurrent = max(_as_int(usage_limits.get('max_concurrent_tasks'), 1), 1)
        started_tasks_current = _as_int(usage_limits.get('started_tasks'), None)
        legacy_running_tasks = _as_int(usage_limits.get('legacy_running_tasks'), 0)
        active_tasks = _as_int(usage_limits.get('active_tasks'), 0)
        if started_tasks_current is None:
            started_tasks_current = active_tasks - legacy_running_tasks
            if started_tasks_current < 0:
                started_tasks_current = active_tasks
        max_daily = max(_as_int(usage_limits.get('max_daily_tasks'), 1), 1)
        tasks_today = _as_int(usage_limits.get('tasks_started_today'), 0)
        next_allowed_display = usage_limits.get('next_allowed_time')
        parsed_next_allowed = _parse_utc_iso(next_allowed_display)
        if parsed_next_allowed:
            next_allowed_display = parsed_next_allowed.strftime('%Y-%m-%d %H:%M UTC')

        if tasks_today >= max_daily:
            console.print(
                f"[red]✗ Daily task limit reached: {tasks_today}/{max_daily} submissions today.[/red]"
            )
            if next_allowed_display:
                console.print(f"[cyan]Next allowed submission window: {next_allowed_display}[/cyan]")
            else:
                console.print("[cyan]Daily reset uses AOE (UTC-12) midnight.[/cyan]")
            sys.exit(1)

        if active_tasks >= max_concurrent:
            console.print(
                f"[yellow]⚠ Concurrent task limit reached: {started_tasks_current}/{max_concurrent} started tasks.[/yellow]"
            )
            console.print("[dim]You can terminate all active tasks to free capacity.[/dim]")

            while True:
                choice = click.prompt(
                    "Terminate all active tasks now? (Y/N)",
                    default='N'
                ).strip().upper()
                if choice in {'Y', 'N'}:
                    break
                console.print("[yellow]Please enter Y or N.[/yellow]")

            if choice == 'Y':
                termination = terminate_all_tasks_for_user(server, token)
                if not termination:
                    console.print("[red]✗ Could not terminate tasks automatically. Submission cancelled.[/red]")
                    sys.exit(1)

                terminated_count = _as_int(termination.get('terminated_count'), 0)
                console.print(
                    f"[cyan]Termination request processed ({terminated_count} tasks affected).[/cyan]"
                )

                usage_limits = fetch_task_usage_limits(server, token)
                if not usage_limits:
                    console.print("[yellow]⚠ Could not verify updated usage. Please retry submission.[/yellow]")
                    sys.exit(1)

                max_concurrent = max(_as_int(usage_limits.get('max_concurrent_tasks'), 1), 1)
                legacy_running_tasks = _as_int(usage_limits.get('legacy_running_tasks'), 0)
                active_tasks = _as_int(usage_limits.get('active_tasks'), 0)
                started_tasks_current = _as_int(usage_limits.get('started_tasks'), None)
                if started_tasks_current is None:
                    started_tasks_current = active_tasks - legacy_running_tasks
                    if started_tasks_current < 0:
                        started_tasks_current = active_tasks
                max_daily = max(_as_int(usage_limits.get('max_daily_tasks'), 1), 1)
                tasks_today = _as_int(usage_limits.get('tasks_started_today'), 0)
                next_allowed_display = usage_limits.get('next_allowed_time')
                parsed_next_allowed = _parse_utc_iso(next_allowed_display)
                if parsed_next_allowed:
                    next_allowed_display = parsed_next_allowed.strftime('%Y-%m-%d %H:%M UTC')

                if tasks_today >= max_daily:
                    console.print(
                        f"[red]✗ Daily task limit reached: {tasks_today}/{max_daily} submissions today.[/red]"
                    )
                    if next_allowed_display:
                        console.print(f"[cyan]Next allowed submission window: {next_allowed_display}[/cyan]")
                    sys.exit(1)

                if active_tasks >= max_concurrent:
                    console.print(
                        f"[red]✗ Concurrent limit still reached ({started_tasks_current}/{max_concurrent} started). Please wait or terminate tasks manually.[/red]"
                    )
                    sys.exit(1)
            else:
                console.print("[yellow]Submission cancelled. Terminate tasks manually to free capacity.[/yellow]")
                sys.exit(1)

    # Store server and token in task_status for UI rendering
    task_status['server'] = server
    task_status['token'] = token

    # Connect to server
    console.print(f"\n[bold cyan]🔌 Connecting to {server}[/bold cyan]")

    sio = create_websocket_client(server, token)

    try:
        sio.connect(server, transports=['websocket', 'polling'])
        time.sleep(1)

        if not task_status['connected']:
            console.print("[red]✗ Failed to connect to server[/red]")
            sys.exit(1)

        console.print("[green]✓ Connected successfully[/green]")

        # Submit task
        console.print("\n[bold cyan]📤 Submitting Task[/bold cyan]")
        console.print("─" * 60)

        if query:
            console.print(f"[cyan]Research Direction: {query}[/cyan]")
        console.print(f"[cyan]GPU Device: {gpu}[/cyan]")
        console.print(f"[cyan]Validation Frequency: {validation_frequency}[/cyan]\n")

        sio.emit('submit_task', {
            'token': token,
            'codebase_digest': validation['codebase_digest'],
            'token_count': validation['token_count'],
            'claude_md': validation['claude_md'],
            'paper_content': validation['paper_content'],
            'query': query,
            'cuda_device': gpu,
            'validation_frequency': validation_frequency
        })

        # ✅ 改进3: Use threading.Event.wait() instead of busy-wait polling
        # This is more efficient and reliable than sleep-based polling
        max_wait = 10  # Wait up to 10 seconds
        task_created_event = control_flags.get('task_created_event')

        if task_created_event:
            # Block until task_created event is received or timeout
            event_received = task_created_event.wait(timeout=max_wait)

            if not event_received:
                # Timeout occurred - task_created event not received
                console.print(f"[yellow]⚠ Timeout waiting for task_created event ({max_wait}s)[/yellow]")
        else:
            # Fallback to old polling method if Event not available (backward compatibility)
            console.print("[dim]Using legacy polling mode (Event not available)[/dim]")
            waited = 0
            while waited < max_wait and not task_status['task_id'] and not task_status.get('error'):
                time.sleep(0.5)
                waited += 0.5

        if not task_status['task_id']:
            error_msg = task_status.get('error', 'Unknown error')
            error_code = task_status.get('error_code')
            details = task_status.get('error_details') or {}

            console.print(f"[red]✗ Task submission failed[/red]")

            # Check if this is a timeout issue vs actual error
            if not error_msg or error_msg == 'Unknown error':
                console.print(f"[yellow]⚠ Timeout waiting for server response ({max_wait}s)[/yellow]")
                console.print(f"[dim]This may indicate:[/dim]")
                console.print(f"[dim]  1. WebSocket connection issue - check network[/dim]")
                console.print(f"[dim]  2. Backend processing delay - task may still be created[/dim]")
                console.print(f"[dim]  3. Backend not sending task_created event[/dim]")
                console.print(f"\n[cyan]💡 Try checking:[/cyan]")
                console.print(f"[cyan]  - Backend logs: tail -f /home/air/DeepScientist/log/backend.log[/cyan]")
                console.print(f"[cyan]  - Task status via: deepscientist-cli list[/cyan]")
            elif error_msg:
                console.print(f"[yellow]Error: {error_msg}[/yellow]")

            if error_code == 'CONCURRENT_LIMIT_EXCEEDED':
                limit = details.get('limit')
                active = details.get('active_tasks')
                if limit is not None and active is not None:
                    console.print(f"[yellow]Active tasks: {active}/{limit}[/yellow]")
                while True:
                    choice = click.prompt(
                        "Terminate all active tasks now? (Y/N)",
                        default='N'
                    ).strip().upper()
                    if choice in {'Y', 'N'}:
                        break
                    console.print("[yellow]Please enter Y or N.[/yellow]")

                if choice == 'Y':
                    termination = terminate_all_tasks_for_user(server, token)
                    if termination:
                        terminated_count = termination.get('terminated_count', 0)
                        console.print(
                            f"[cyan]Termination request processed ({terminated_count} tasks affected).[/cyan]"
                        )
                        console.print("[yellow]Please rerun the submission to start a new task.[/yellow]")
                    else:
                        console.print("[red]✗ Failed to terminate tasks automatically.[/red]")
                else:
                    console.print("[yellow]Submission cancelled. Existing tasks remain active.[/yellow]")
            elif error_code == 'DAILY_LIMIT_EXCEEDED':
                next_allowed = details.get('next_allowed_time')
                parsed_next = _parse_utc_iso(next_allowed)
                if parsed_next:
                    next_allowed = parsed_next.strftime('%Y-%m-%d %H:%M UTC')
                if next_allowed:
                    console.print(f"[cyan]Next allowed submission window: {next_allowed}[/cyan]")
                else:
                    console.print("[cyan]Daily limit resets at AOE (UTC-12) midnight.[/cyan]")
            else:
                console.print("[dim]Check backend logs for details[/dim]")

            sio.disconnect()
            sys.exit(1)

        # Join task room
        sio.emit('join_task', {
            'task_id': task_status['task_id'],
            'token': token
        })

        console.print(f"[green]✓ Task submitted: {task_status['task_id']}[/green]")
        console.print(f"[cyan]Token count: {validation['token_count']:,}[/cyan]\n")

        dashboard_url = task_status.get('dashboard_url')
        if not dashboard_url:
            dashboard_url = compute_dashboard_url(
                task_id=task_status['task_id'],
                prompt=task_status.get('query') or query,
                token=task_status.get('token') or token,
                server=task_status.get('server') or server,
            )
            if dashboard_url:
                task_status['dashboard_url'] = dashboard_url
                control_flags['force_ui_update'] = True
                clamp_event_scroll()

        # Copy baseline code to workspace for future implementations
        try:
            workspace_base = get_workspace_dir()
            task_workspace = workspace_base / task_status['task_id']
            task_workspace.mkdir(parents=True, exist_ok=True)

            baseline_backup = task_workspace / "baseline_code"
            if baseline_backup.exists():
                shutil.rmtree(baseline_backup)

            console.print(f"[cyan]📁 Creating baseline code backup...[/cyan]")
            shutil.copytree(codebase_path, baseline_backup, dirs_exist_ok=True)
            console.print(f"[green]✓ Baseline code backed up to workspace[/green]\n")

            # Update task_status with baseline backup path
            task_status['baseline_backup'] = str(baseline_backup)

            # Upload compressed zip if upload_baseline is True
            if upload_baseline and baseline_code_zip and baseline_code_zip.exists():
                task_status['claude_logs'].append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'system',
                    'formatted': f'📤 Uploading baseline code to backend ({zip_size_mb:.2f} MB)...'
                })

                try:
                    from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

                    zip_size = baseline_code_zip.stat().st_size

                    with Progress(
                        TextColumn("[bold blue]{task.description}"),
                        BarColumn(complete_style="green", finished_style="bold green"),
                        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                        TimeRemainingColumn(),
                        console=console
                    ) as progress:
                        upload_task = progress.add_task(
                            f"Uploading baseline_code.zip",
                            total=zip_size
                        )

                        # Custom upload with progress callback
                        class ProgressFileReader:
                            def __init__(self, file_path, progress, task_id):
                                self.file_path = file_path
                                self.progress = progress
                                self.task_id = task_id
                                self.file = open(file_path, 'rb')

                            def read(self, size=-1):
                                chunk = self.file.read(size)
                                if chunk:
                                    self.progress.update(self.task_id, advance=len(chunk))
                                return chunk

                            def __len__(self):
                                return self.file_path.stat().st_size

                            def close(self):
                                self.file.close()

                            def __enter__(self):
                                return self

                            def __exit__(self, *args):
                                self.close()

                        # Upload file
                        upload_url = f"{server}/api/tasks/{task_status['task_id']}/upload_baseline"

                        task_status['claude_logs'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'system',
                            'formatted': f'🔗 Upload URL: {upload_url}'
                        })

                        task_status['claude_logs'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'system',
                            'formatted': f'📂 Local file: {baseline_code_zip.absolute()} (exists: {baseline_code_zip.exists()})'
                        })

                        with ProgressFileReader(baseline_code_zip, progress, upload_task) as reader:
                            task_status['claude_logs'].append({
                                'timestamp': datetime.now().isoformat(),
                                'type': 'system',
                                'formatted': f'🚀 Starting upload request...'
                            })
                            response = requests.post(
                                upload_url,
                                files={'baseline_code': ('baseline_code.zip', reader, 'application/zip')},
                                headers={'Authorization': f'Bearer {token}'},
                                timeout=300
                            )
                            task_status['claude_logs'].append({
                                'timestamp': datetime.now().isoformat(),
                                'type': 'system',
                                'formatted': f'✓ Upload request completed'
                            })

                        task_status['claude_logs'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'system',
                            'formatted': f'📊 Response status: {response.status_code}'
                        })

                        if response.status_code == 200:
                            result = response.json()
                            relative_path = result.get('relative_path', 'unknown')
                            backend_size_mb = result.get('file_size_mb', 0)
                            task_status['claude_logs'].append({
                                'timestamp': datetime.now().isoformat(),
                                'type': 'system',
                                'formatted': f'✓ Baseline code uploaded successfully ({zip_size_mb:.2f} MB)'
                            })
                            task_status['claude_logs'].append({
                                'timestamp': datetime.now().isoformat(),
                                'type': 'system',
                                'formatted': f'📁 Saved to: {relative_path}'
                            })
                        else:
                            task_status['claude_logs'].append({
                                'timestamp': datetime.now().isoformat(),
                                'type': 'system',
                                'formatted': f'⚠ Upload failed (HTTP {response.status_code}): {response.text}'
                            })

                except Exception as upload_error:
                    task_status['claude_logs'].append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'system',
                        'formatted': f'⚠ Failed to upload baseline code: {upload_error}'
                    })
                finally:
                    # Clean up temporary zip file
                    if baseline_code_zip and baseline_code_zip.exists():
                        temp_path = str(baseline_code_zip.absolute())
                        baseline_code_zip.unlink()
                        task_status['claude_logs'].append({
                            'timestamp': datetime.now().isoformat(),
                            'type': 'system',
                            'formatted': f'🗑️  Cleaned up temporary file: {temp_path}'
                        })

        except Exception as e:
            console.print(f"[yellow]⚠ Warning: Failed to backup baseline code: {e}[/yellow]")
            console.print(f"[yellow]  Implementations will use original path: {codebase_path}[/yellow]\n")
        finally:
            # Ensure cleanup of temporary zip file if it wasn't uploaded
            if not upload_baseline and baseline_code_zip and baseline_code_zip.exists():
                try:
                    baseline_code_zip.unlink()
                except:
                    pass

        # Start heartbeat thread
        heartbeat_t = threading.Thread(target=heartbeat_thread, args=(sio, token), daemon=True)
        heartbeat_t.start()
        # Set initial heartbeat_next using configured interval (in seconds)
        conn_config = get_connection_config()
        task_status['heartbeat_next'] = datetime.now() + timedelta(seconds=conn_config['heartbeat_interval'])

        # Start monitoring
        console.print("[bold green]✓ Now monitoring task in real-time[/bold green]")
        console.print("[dim]Task will continue until completion or timeout (configured by admin)[/dim]")
        console.print("[dim]Type '/q' or '/quit' + Enter to terminate | Ctrl+C to exit monitoring[/dim]\n")

        # UI monitoring with command input support
        termination_requested = threading.Event()
        termination_confirmed = threading.Event()
        input_queue = queue.Queue()
        last_ui_render = None
        keyboard_interrupt_flag = threading.Event()

        def input_listener():
            """Listen for user commands in a separate thread"""
            while not control_flags['stop_requested'] and not termination_requested.is_set() and not control_flags.get('should_exit', False):
                try:
                    # Use a non-blocking approach with select on Unix-like systems
                    if hasattr(sys.stdin, 'fileno'):
                        import select
                        # Check if input is available without blocking
                        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                        if ready:
                            user_input = sys.stdin.readline().strip()
                            if user_input:
                                input_queue.put(user_input)
                    else:
                        # Fallback for Windows or non-terminal
                        time.sleep(0.5)
                except (EOFError, KeyboardInterrupt):
                    # Signal that Ctrl+C was pressed
                    keyboard_interrupt_flag.set()
                    break
                except Exception as e:
                    # Silently ignore other errors to avoid spam
                    break

        # Start input listener thread
        input_thread = threading.Thread(target=input_listener, daemon=True)
        input_thread.start()

        try:
            last_update = time.time()

            while not control_flags['stop_requested'] and not control_flags['should_exit']:
                current_time = time.time()

                # Check if server terminated the task
                if control_flags['should_exit'] or task_status.get('terminated_by_server'):
                    console.print("\n\n[red]🛑 Task has been terminated by server[/red]")
                    if task_status.get('termination_message'):
                        console.print(f"[yellow]{task_status['termination_message']}[/yellow]")
                    console.print("\n[dim]Exiting to main menu...[/dim]")
                    time.sleep(2)
                    break

                # Check if Ctrl+C was pressed in input listener
                if keyboard_interrupt_flag.is_set():
                    raise KeyboardInterrupt()

                # Check for user input commands
                try:
                    user_input = input_queue.get_nowait()
                    if user_input in ['/q', '/quit']:
                        termination_requested.set()
                    elif user_input:
                        # Don't clear screen when showing command response
                        console.print(f"\n[yellow]Unknown command: {user_input}[/yellow]")
                        console.print(f"[dim]Available commands: /q or /quit to terminate[/dim]\n")
                except queue.Empty:
                    pass

                # Check for termination request
                if termination_requested.is_set() and not termination_confirmed.is_set():
                    console.print("\n\n[yellow]⏳ Termination requested - Stopping task...[/yellow]")
                    console.print("[cyan]Sending stop signal to all agents...[/cyan]\n")

                    # IMMEDIATELY set stop_requested flag to stop all agents (including implementer)
                    # This behaves like Ctrl+C, stopping all running processes immediately
                    control_flags['stop_requested'] = True

                    # Send stop request to backend
                    try:
                        sio.emit('stop_task', {
                            'task_id': task_status['task_id'],
                            'token': token,
                            'graceful': True
                        })

                        console.print("[cyan]Waiting for backend confirmation...[/cyan]\n")

                        # Wait for confirmation (max 30 seconds)
                        wait_start = time.time()
                        while time.time() - wait_start < 30:
                            if task_status.get('status') == 'terminated':
                                termination_confirmed.set()
                                console.print("[green]✓ Task terminated successfully[/green]")
                                break
                            time.sleep(0.5)

                        if not termination_confirmed.is_set():
                            console.print("[yellow]⚠ Termination request sent, but confirmation timeout[/yellow]")
                            console.print("[yellow]Task may still be terminating in background[/yellow]")
                    except Exception as e:
                        console.print(f"[red]✗ Error sending termination request: {e}[/red]")

                    break

                # Update UI every 5 seconds or immediately when forced by socketio events
                if current_time - last_update >= 5.0 or control_flags['force_ui_update']:
                    # Clear screen and render UI
                    console.clear()
                    console.print(generate_enhanced_ui())

                    # Show input prompt after UI update
                    console.print("\n[dim]Commands: /q or /quit to terminate | Ctrl+C to exit monitoring[/dim]")
                    console.print("[bold cyan]>[/bold cyan] ", end='')

                    last_update = current_time
                    control_flags['force_ui_update'] = False  # Reset flag

                time.sleep(0.5)

                # No client-side timeout check - backend handles all timeouts
        except KeyboardInterrupt:
            # Restore terminal to normal state
            console.print("\n\n[yellow]⚠️  Monitoring stopped (task continues in background)[/yellow]")
            console.print(f"Task ID: {task_status['task_id']}")
            console.print("Use 'deepscientist-cli list' to reconnect\n")
        else:
            if control_flags.get('final_summary_ready') and task_status.get('completion_summary'):
                console.print("\n[bold cyan]🎉 Task completed. Final summary:[/bold cyan]\n")
                try:
                    summary_panel = build_completion_summary_panel(
                        task_status['completion_summary'],
                        task_status.get('status') or 'completed'
                    )
                    console.print(summary_panel)
                except Exception as summary_exc:
                    console.print(f"[yellow]⚠ Unable to render summary: {summary_exc}[/yellow]")
                    console.print(json.dumps(task_status['completion_summary'], indent=2, ensure_ascii=False))
                finally:
                    control_flags['final_summary_ready'] = False
        finally:
            # ALWAYS restore terminal state when exiting
            import os
            import sys
            # Reset stdin to blocking mode
            if hasattr(sys.stdin, 'fileno'):
                try:
                    import fcntl
                    flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
                    fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
                except:
                    pass
            # Reset terminal settings
            if hasattr(os, 'system'):
                os.system('stty sane 2>/dev/null || true')

    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if sio.connected:
            sio.disconnect()


def start_live_dashboard(tasks, token, server):
    """
    Start a live dashboard showing all tasks in real-time.
    Updates every few seconds to show current status of all tasks.
    """
    from rich.live import Live
    from rich.layout import Layout
    from datetime import datetime
    import time

    console.print("\n[bold cyan]🚀 Starting Live Dashboard...[/bold cyan]")
    console.print("[dim]Press Ctrl+C to exit[/dim]\n")
    time.sleep(1)

    def generate_dashboard():
        """Generate the dashboard layout"""
        layout = Layout()

        # Fetch latest task data with full details
        try:
            # Fetch all tasks with their latest status
            response = requests.get(
                f"{server}/api/tasks",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10
            )
            if response.status_code == 200:
                current_tasks = response.json().get('tasks', [])
            else:
                current_tasks = tasks  # Fallback to cached tasks
        except Exception as e:
            current_tasks = tasks  # Fallback to cached tasks

        # Create header
        header_text = Text()
        header_text.append("📊 ", style="bold cyan")
        header_text.append("DeepScientist Live Dashboard", style="bold white")
        header_text.append(f" | Last Update: {datetime.now().strftime('%H:%M:%S')}", style="dim")

        header_panel = Panel(
            header_text,
            style="bold cyan",
            border_style="cyan"
        )

        # Create tasks table with detailed information
        tasks_table = Table(
            show_header=True,
            header_style="bold white on blue",
            box=box.ROUNDED,
            expand=True,
            border_style="cyan",
            show_lines=True
        )

        tasks_table.add_column("№", style="dim", width=3, justify="right")
        tasks_table.add_column("Task ID", style="cyan", width=36)
        tasks_table.add_column("Status", width=14, justify="center")
        tasks_table.add_column("Latest Activity", width=50)
        tasks_table.add_column("Tokens", justify="right", width=11)
        tasks_table.add_column("GPU", style="magenta", width=7, justify="center")

        for idx, task in enumerate(current_tasks, 1):
            tid = task.get('task_id', 'N/A')
            status = task.get('status', 'unknown').lower()

            # Get latest event information
            events = task.get('events', [])
            latest_activity = "—"
            if events and len(events) > 0:
                latest_event = events[-1]
                event_title = latest_event.get('title', '')
                event_time = latest_event.get('timestamp', '')

                # Format time if available
                time_str = ""
                if event_time:
                    try:
                        from datetime import datetime
                        event_dt = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                        time_str = f"[dim]{event_dt.strftime('%H:%M:%S')}[/dim] "
                    except:
                        pass

                # Truncate long event titles
                if len(event_title) > 42:
                    event_title = event_title[:39] + '...'

                latest_activity = f"{time_str}{event_title}"

            # Token count display
            tokens = task.get('token_count', 0)
            total_llm = task.get('total_llm_tokens')
            if total_llm is None:
                total_prompt = task.get('total_prompt_tokens') or 0
                total_completion = task.get('total_completion_tokens') or 0
                total_llm = total_prompt + total_completion

            # Show both codebase tokens and LLM tokens
            tokens_display = f"[cyan]{tokens:,}[/cyan]"
            if total_llm > 0:
                tokens_display += f"\n[dim]LLM:{total_llm//1000}K[/dim]"

            cuda_device = task.get('cuda_device')
            cuda_display = f"cuda:{cuda_device}" if cuda_device not in (None, '') else '—'

            # Status with animation and color
            status_display = {
                'queued': '[yellow]⏳ QUEUE[/yellow]',
                'started': '[green]▶️  START[/green]',
                'running': '[green bold]🟢 RUN[/green bold]',
                'paused': '[yellow]⏸️  PAUSE[/yellow]',
                'completed': '[cyan]✅ DONE[/cyan]',
                'failed': '[red]❌ FAIL[/red]',
                'terminated': '[dim]⭕ STOP[/dim]'
            }.get(status, f'[dim]{status.upper()}[/dim]')

            tasks_table.add_row(
                f"{idx}",
                tid[:34] + ".." if len(tid) > 34 else tid,
                status_display,
                latest_activity,
                tokens_display,
                cuda_display
            )

        tasks_panel = Panel(
            tasks_table,
            title="[bold white]Active Research Tasks[/bold white]",
            border_style="blue"
        )

        # Create statistics panel
        status_counts = {}
        for task in current_tasks:
            status = task.get('status', 'unknown').lower()
            status_counts[status] = status_counts.get(status, 0) + 1

        prompt_sum = sum((task.get('total_prompt_tokens') or 0) for task in current_tasks)
        completion_sum = sum((task.get('total_completion_tokens') or 0) for task in current_tasks)

        stats_text = f"""[bold]Total Tasks:[/bold] {len(current_tasks)}

[bold]Status Breakdown:[/bold]
  🟢 Running: [green bold]{status_counts.get('running', 0)}[/green bold]
  ⏳ Queued: [yellow]{status_counts.get('queued', 0)}[/yellow]
  ✅ Completed: [cyan]{status_counts.get('completed', 0)}[/cyan]
  ⏸️  Paused: [yellow]{status_counts.get('paused', 0)}[/yellow]
  ❌ Failed: [red]{status_counts.get('failed', 0)}[/red]

[bold]LLM Token Usage:[/bold]
  Prompt: [green]{prompt_sum:,}[/green]
  Completion: [blue]{completion_sum:,}[/blue]
  Total: [bold]{prompt_sum + completion_sum:,}[/bold]"""

        stats_panel = Panel(
            stats_text,
            title="[bold white]📈 Statistics[/bold white]",
            border_style="green"
        )

        # Create footer
        footer_text = "[dim]Press [bold]Ctrl+C[/bold] to exit dashboard • Auto-refresh every 5 seconds[/dim]"
        footer_panel = Panel(footer_text, style="dim", border_style="dim")

        # Split layout
        layout.split_column(
            Layout(header_panel, size=3),
            Layout(tasks_panel),
            Layout(stats_panel, size=12),
            Layout(footer_panel, size=3)
        )

        return layout

    # Run live dashboard
    try:
        with Live(generate_dashboard(), refresh_per_second=0.2, console=console, screen=True) as live:
            while True:
                time.sleep(5)  # Update every 5 seconds
                live.update(generate_dashboard())
    except KeyboardInterrupt:
        console.print("\n[cyan]Dashboard closed[/cyan]")


def monitor_task_impl(task_id, token, server):
    """Internal implementation for monitoring a task"""
    # Reset runtime state before loading
    reset_runtime_state()

    console.print(f"\n[bold cyan]🔌 Connecting to {server}[/bold cyan]")

    # Set task ID, server and token
    task_status['task_id'] = task_id
    task_status['server'] = server
    task_status['token'] = token
    update_dashboard_url()

    # Load task information
    console.print("[cyan]Loading task information...[/cyan]")
    try:
        response = requests.get(
            f"{server}/api/tasks",
            headers={'Authorization': f'Bearer {token}'},
            timeout=10
        )
        if response.status_code == 200:
            tasks_data = response.json().get('tasks', [])
            current_task = next((t for t in tasks_data if t['task_id'] == task_id), None)

            if current_task:
                task_status['status'] = current_task.get('status', 'unknown')
                task_status['token_count'] = current_task.get('token_count', 0)
                task_status['query'] = current_task.get('query')
                task_status['cuda_device'] = current_task.get('cuda_device')
                task_status['total_prompt_tokens'] = current_task.get('total_prompt_tokens', 0)
                task_status['total_completion_tokens'] = current_task.get('total_completion_tokens', 0)
                task_status['total_llm_tokens'] = current_task.get('total_llm_tokens', 0)
                task_status['abstract'] = current_task.get('abstract')
                update_dashboard_url(current_task.get('dashboard_url'))

                if current_task.get('started_at'):
                    try:
                        task_status['start_time'] = datetime.fromisoformat(current_task['started_at'].replace('Z', '+00:00'))
                    except:
                        task_status['start_time'] = datetime.now()
            else:
                console.print(f"[red]✗ Task {task_id} not found[/red]")
                sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ Failed to load task details: {e}[/red]")
        sys.exit(1)

    # Load historical events
    console.print("[cyan]Loading historical events...[/cyan]")
    try:
        response = requests.get(
            f"{server}/api/tasks/{task_id}/events",
            headers={'Authorization': f'Bearer {token}'},
            timeout=10
        )
        if response.status_code == 200:
            events_data = response.json().get('events', [])
            for event in events_data:
                task_status['events'].append({
                    'timestamp': event.get('created_at', datetime.now().isoformat()),
                    'type': event.get('type', 'activity'),
                    'title': event.get('title', ''),
                    'cycle_number': event.get('cycle_number'),
                    'island_id': event.get('island_id'),
                    'idea_id': event.get('idea_id'),
                })
            console.print(f"[green]✓ Loaded {len(events_data)} events[/green]")
            console.print(f"[dim]   Events in task_status: {len(task_status['events'])}[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not load events: {e}[/yellow]")

    # Load implementation logs from local config
    console.print("[cyan]Loading implementation logs...[/cyan]")
    try:
        load_implementations_for_task(task_id)
        entries = fetch_implementation_entries(task_id)
        if entries:
            console.print(f"[green]✓ Loaded {len(entries)} implementation logs[/green]")
        else:
            console.print(f"[dim]   No implementation logs found[/dim]")
    except Exception as e:
        console.print(f"[yellow]⚠ Could not load implementation logs: {e}[/yellow]")

    console.print(f"[green]✓ Data loading complete[/green]")

    # Check if task is in a terminal state
    terminal_states = ['TERMINATED', 'COMPLETED', 'FAILED']
    current_status = task_status['status'].upper()

    if current_status in terminal_states:
        console.print(f"\n[yellow]⚠ Task is in terminal state: {current_status}[/yellow]")
        console.print("[dim]This task has finished and cannot be monitored in real-time.[/dim]\n")

        # Task Information Panel
        task_info_text = f"""[bold]Task ID:[/bold] {task_id}
[bold]Status:[/bold] [{('cyan' if current_status == 'COMPLETED' else 'red' if current_status == 'FAILED' else 'yellow')}]{current_status}[/]
[bold]Token Count:[/bold] [cyan]{task_status['token_count']:,}[/cyan]
[bold]Research Focus:[/bold] {task_status.get('query', 'N/A')}
[bold]Total Events:[/bold] {len(task_status['events'])}"""

        task_panel = Panel(
            task_info_text,
            title="[bold white]📋 Task Information[/bold white]",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(task_panel)
        console.print()

        # Events Table
        if task_status.get('events'):
            # Merge consecutive identical events (same logic as real-time monitoring)
            merged_events = []
            for event in task_status['events']:
                event_type = event.get('type', 'activity')
                title = event.get('title', 'No title')
                agent_name = event.get('agent_name', '')

                # Extract base title (remove +N suffix if exists)
                base_title = re.sub(r'\s*\+\d+$', '', title)

                should_merge = False
                if merged_events and event_type == 'llm_call':
                    last_event = merged_events[-1]
                    last_title = last_event.get('title', '')
                    last_base_title = re.sub(r'\s*\+\d+$', '', last_title)
                    last_agent = last_event.get('agent_name', '')

                    if (last_event.get('type') == 'llm_call' and
                        last_base_title == base_title and
                        last_agent == agent_name):
                        should_merge = True

                        # Extract current count
                        match = re.search(r'\+(\d+)$', last_title)
                        if match:
                            count = int(match.group(1)) + 1
                        else:
                            count = 2

                        # Update last event
                        last_event['title'] = f"{base_title} +{count}"
                        last_event['timestamp'] = event.get('timestamp')

                if not should_merge:
                    merged_events.append(event.copy())

            # Reverse to show newest first
            if len(merged_events) > 50:
                display_events = builtins.list(reversed(merged_events[-50:]))
            else:
                display_events = builtins.list(reversed(merged_events))

            events_table = Table(
                show_header=True,
                header_style="bold white on blue",
                box=box.ROUNDED,
                title="[bold cyan]📝 Task Events History (Latest First)[/bold cyan]",
                title_style="bold cyan",
                expand=True,
                border_style="cyan"
            )

            events_table.add_column("№", style="dim", width=4, justify="right")
            events_table.add_column("Timestamp", style="yellow", width=19)
            events_table.add_column("Type", style="magenta", width=14, justify="center")
            events_table.add_column("Event", style="white")
            events_table.add_column("Cycle", style="cyan", width=8, justify="center")

            for idx, event in enumerate(display_events, 1):
                timestamp = event.get('timestamp', '')[:19] if event.get('timestamp') else 'N/A'
                event_type = event.get('type', 'activity')
                title = event.get('title', 'No title')
                cycle = event.get('cycle_number')
                cycle_display = f"#{cycle}" if cycle else '—'

                # Truncate long titles
                if len(title) > 80:
                    title = title[:77] + '...'

                # Type emoji with proper display
                type_display = {
                    'activity': '🎯 Activity',
                    'file_update': '📝 File',
                    'terminal_output': '💻 Output',
                    'finding': '⭐ Finding',
                    'llm_call': '🤖 LLM Call'
                }.get(event_type, event_type)

                events_table.add_row(
                    f"{idx}",
                    timestamp,
                    type_display,
                    title,
                    cycle_display
                )

            console.print(events_table)
            console.print()

        # Show tips
        tip_panel = Panel(
            f"[cyan]💡 Use 'findings {task_id}' to view research results[/cyan]\n"
            f"[dim]   Or visit the dashboard for detailed analysis[/dim]",
            border_style="dim",
            padding=(0, 2)
        )
        console.print(tip_panel)
        return

    console.print(f"[dim]→ Proceeding to WebSocket connection...[/dim]")
    console.print(f"\n[bold cyan]📤 Reconnecting to Task: {task_id}[/bold cyan]")
    console.print("─" * 60)
    console.print(f"[cyan]Status: {current_status}[/cyan]")
    console.print(f"[cyan]Token count: {task_status['token_count']:,}[/cyan]\n")

    # Create WebSocket client
    sio = create_websocket_client(server, token)

    try:
        sio.connect(server, transports=['websocket', 'polling'])
        time.sleep(1)

        if not task_status['connected']:
            console.print("[red]✗ Failed to connect to server[/red]")
            sys.exit(1)

        # Join task room
        sio.emit('join_task', {
            'task_id': task_id,
            'token': token
        })

        # Start heartbeat
        heartbeat_t = threading.Thread(target=heartbeat_thread, args=(sio, token), daemon=True)
        heartbeat_t.start()
        # Set initial heartbeat_next using configured interval (in seconds)
        conn_config = get_connection_config()
        task_status['heartbeat_next'] = datetime.now() + timedelta(seconds=conn_config['heartbeat_interval'])

        # Start monitoring
        console.print("[bold green]✓ Now monitoring task in real-time[/bold green]")
        console.print("[dim]Type '/q' or '/quit' + Enter to terminate | Ctrl+C to exit monitoring[/dim]\n")

        # UI monitoring with command input support
        termination_requested = threading.Event()
        termination_confirmed = threading.Event()
        input_queue = queue.Queue()
        keyboard_interrupt_flag = threading.Event()

        def input_listener():
            """Listen for user commands in a separate thread"""
            while not control_flags['stop_requested'] and not termination_requested.is_set() and not control_flags.get('should_exit', False):
                try:
                    # Use a non-blocking approach with select on Unix-like systems
                    if hasattr(sys.stdin, 'fileno'):
                        import select
                        # Check if input is available without blocking
                        ready, _, _ = select.select([sys.stdin], [], [], 0.5)
                        if ready:
                            user_input = sys.stdin.readline().strip()
                            if user_input:
                                input_queue.put(user_input)
                    else:
                        # Fallback for Windows or non-terminal
                        time.sleep(0.5)
                except (EOFError, KeyboardInterrupt):
                    # Signal that Ctrl+C was pressed
                    keyboard_interrupt_flag.set()
                    break
                except Exception as e:
                    # Silently ignore other errors to avoid spam
                    break

        # Start input listener thread
        input_thread = threading.Thread(target=input_listener, daemon=True)
        input_thread.start()

        try:
            last_update = time.time()

            while not control_flags['stop_requested'] and not control_flags['should_exit']:
                current_time = time.time()

                # Check if server terminated the task
                if control_flags['should_exit'] or task_status.get('terminated_by_server'):
                    console.print("\n\n[red]🛑 Task has been terminated by server[/red]")
                    if task_status.get('termination_message'):
                        console.print(f"[yellow]{task_status['termination_message']}[/yellow]")
                    console.print("\n[dim]Exiting to main menu...[/dim]")
                    time.sleep(2)
                    break

                # Check if Ctrl+C was pressed in input listener
                if keyboard_interrupt_flag.is_set():
                    raise KeyboardInterrupt()

                # Check for user input commands
                try:
                    user_input = input_queue.get_nowait()
                    if user_input in ['/q', '/quit']:
                        termination_requested.set()
                    elif user_input:
                        # Don't clear screen when showing command response
                        console.print(f"\n[yellow]Unknown command: {user_input}[/yellow]")
                        console.print(f"[dim]Available commands: /q or /quit to terminate[/dim]\n")
                except queue.Empty:
                    pass

                # Check for termination request
                if termination_requested.is_set() and not termination_confirmed.is_set():
                    console.print("\n\n[yellow]⏳ Termination requested - Stopping task...[/yellow]")
                    console.print("[cyan]Sending stop signal to all agents...[/cyan]\n")

                    # IMMEDIATELY set stop_requested flag to stop all agents (including implementer)
                    # This behaves like Ctrl+C, stopping all running processes immediately
                    control_flags['stop_requested'] = True

                    # Send stop request to backend
                    try:
                        sio.emit('stop_task', {
                            'task_id': task_id,
                            'token': token,
                            'graceful': True
                        })

                        console.print("[cyan]Waiting for backend confirmation...[/cyan]\n")

                        # Wait for confirmation (max 30 seconds)
                        wait_start = time.time()
                        while time.time() - wait_start < 30:
                            if task_status.get('status') == 'terminated':
                                termination_confirmed.set()
                                console.print("[green]✓ Task terminated successfully[/green]")
                                break
                            time.sleep(0.5)

                        if not termination_confirmed.is_set():
                            console.print("[yellow]⚠ Termination request sent, but confirmation timeout[/yellow]")
                            console.print("[yellow]Task may still be terminating in background[/yellow]")
                    except Exception as e:
                        console.print(f"[red]✗ Error sending termination request: {e}[/red]")

                    break

                # Update UI every 5 seconds or when WebSocket events arrive
                if current_time - last_update >= 5.0 or control_flags['force_ui_update']:
                    # Clear screen and render UI
                    console.clear()
                    console.print(generate_enhanced_ui())

                    # Show input prompt after UI update
                    console.print("\n[dim]Commands: /q or /quit to terminate | Ctrl+C to exit monitoring[/dim]")
                    console.print("[bold cyan]>[/bold cyan] ", end='')

                    last_update = current_time
                    control_flags['force_ui_update'] = False

                time.sleep(0.5)

        except KeyboardInterrupt:
            # Restore terminal to normal state
            console.print("\n\n[yellow]⚠️  Monitoring stopped (task continues in background)[/yellow]")
            console.print(f"Task ID: {task_status['task_id']}")
            console.print("Use 'deepscientist-cli list' to reconnect\n")
        else:
            if control_flags.get('final_summary_ready') and task_status.get('completion_summary'):
                console.print("\n[bold cyan]🎉 Task completed. Final summary:[/bold cyan]\n")
                try:
                    summary_panel = build_completion_summary_panel(
                        task_status['completion_summary'],
                        task_status.get('status') or 'completed'
                    )
                    console.print(summary_panel)
                except Exception as summary_exc:
                    console.print(f"[yellow]⚠ Unable to render summary: {summary_exc}[/yellow]")
                    console.print(json.dumps(task_status['completion_summary'], indent=2, ensure_ascii=False))
                finally:
                    control_flags['final_summary_ready'] = False
        finally:
            # ALWAYS restore terminal state when exiting
            import os
            import sys
            # Reset stdin to blocking mode
            if hasattr(sys.stdin, 'fileno'):
                try:
                    import fcntl
                    flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
                    fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
                except:
                    pass
            # Reset terminal settings
            if hasattr(os, 'system'):
                os.system('stty sane 2>/dev/null || true')

    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if sio.connected:
            sio.disconnect()


@cli.command(name="list")
@click.option('--token', help='API authentication token')
@click.option('--server', default=None, help='Backend server URL')
def list_tasks(token, server):
    """List all your tasks"""
    print_banner()

    server = resolve_server(server)
    ensure_server_available(server)
    token, token_source = resolve_token(token, server)

    if not token:
        console.print("[red]✗ Error:[/red] No authentication token")
        sys.exit(1)

    # Show conda configuration if available
    if CONDA_BASE_PATH:
        console.print(f"[dim]🐍 Conda: {CONDA_BASE_PATH} (env: {CONDA_ENV_NAME})[/dim]")

    try:
        tasks = fetch_tasks(server, token)
    except RuntimeError as err:
        console.print(f"[red]✗ Error:[/red] {err}")
        sys.exit(1)

    if not tasks:
        console.print("\n[yellow]No tasks found[/yellow]\n")
        return

    # Display full-screen task list with rich panels
    console.print("\n")

    # Create main task table with full width
    table = Table(
        show_header=True,
        header_style="bold white on blue",
        box=box.DOUBLE_EDGE,
        title="[bold cyan]📋 Your Research Tasks[/bold cyan]",
        title_style="bold cyan",
        expand=True,
        border_style="cyan"
    )

    table.add_column("№", style="dim", width=4, justify="right")
    table.add_column("Task ID", style="cyan bold", width=38)
    table.add_column("Status", style="white", width=12, justify="center")
    table.add_column("Created", style="yellow", width=19)
    table.add_column("Codebase", justify="right", width=10)
    table.add_column("LLM Usage", justify="right", width=12)
    table.add_column("GPU", style="magenta", width=6, justify="center")
    table.add_column("Research Focus", style="white")

    for idx, task in enumerate(tasks, 1):
        tid = task.get('task_id', 'N/A')
        status = task.get('status', 'unknown').lower()
        created = task.get('created_at', 'N/A')
        tokens = task.get('token_count', 0)
        total_prompt = task.get('total_prompt_tokens') or 0
        total_completion = task.get('total_completion_tokens') or 0
        total_llm = task.get('total_llm_tokens')
        if total_llm is None:
            total_llm = (total_prompt or 0) + (total_completion or 0)

        llm_display = f"{total_llm:,}" if total_llm else '—'

        cuda_device = task.get('cuda_device')
        cuda_display = f"cuda:{cuda_device}" if cuda_device not in (None, '') else '—'

        query = task.get('query') or ''
        if len(query) > 50:
            focus_display = query[:47] + '…'
        else:
            focus_display = query or '—'

        # Status emoji and styling
        status_display = {
            'queued': '[yellow]⏳ QUEUED[/yellow]',
            'started': '[green]▶️  STARTED[/green]',
            'running': '[green bold]🟢 RUNNING[/green bold]',
            'paused': '[yellow]⏸️  PAUSED[/yellow]',
            'completed': '[cyan]✅ DONE[/cyan]',
            'failed': '[red]❌ FAILED[/red]',
            'terminated': '[dim]⭕ STOPPED[/dim]'
        }.get(status, f'[dim]{status.upper()}[/dim]')

        table.add_row(
            f"{idx}",
            tid,
            status_display,
            created[:19] if created != 'N/A' else 'N/A',
            f"[cyan]{tokens:,}[/cyan]",
            f"[green]{llm_display}[/green]",
            cuda_display,
            focus_display
        )

    console.print(table)
    console.print()

    # Summary panel
    from rich.panel import Panel
    from rich.columns import Columns

    prompt_sum = sum((task.get('total_prompt_tokens') or 0) for task in tasks)
    completion_sum = sum((task.get('total_completion_tokens') or 0) for task in tasks)
    total_tasks = len(tasks)

    status_counts = {}
    for task in tasks:
        status = task.get('status', 'unknown').lower()
        status_counts[status] = status_counts.get(status, 0) + 1

    # Create summary statistics
    summary_text = f"""[bold]Total Tasks:[/bold] {total_tasks}
[bold]Running:[/bold] [green]{status_counts.get('running', 0)}[/green]  [bold]Queued:[/bold] [yellow]{status_counts.get('queued', 0)}[/yellow]  [bold]Completed:[/bold] [cyan]{status_counts.get('completed', 0)}[/cyan]  [bold]Failed:[/bold] [red]{status_counts.get('failed', 0)}[/red]
[bold]Total LLM Tokens:[/bold] [green]Prompt: {prompt_sum:,}[/green]  [blue]Completion: {completion_sum:,}[/blue]"""

    summary_panel = Panel(
        summary_text,
        title="[bold white]📊 Summary Statistics[/bold white]",
        border_style="blue",
        padding=(1, 2)
    )

    console.print(summary_panel)
    console.print()




@cli.command()
@click.argument('task_id')
@click.option('--token', help='API authentication token')
@click.option('--server', default=None, help='Backend server URL')
def pause(task_id, token, server):
    """Pause a running task"""
    server = resolve_server(server)
    ensure_server_available(server)
    token, _ = resolve_token(token, server)

    if not token:
        console.print("[red]✗ Error:[/red] No authentication token")
        sys.exit(1)

    console.print(f"[yellow]⏸️  Pausing task {task_id}...[/yellow]")

    sio = socketio.Client()
    try:
        sio.connect(server)
        sio.emit('pause_task', {'task_id': task_id, 'token': token})
        time.sleep(1)
        console.print("[green]✓ Pause request sent[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
    finally:
        sio.disconnect()


@cli.command()
@click.argument('task_id')
@click.option('--token', help='API authentication token')
@click.option('--server', default=None, help='Backend server URL')
def resume(task_id, token, server):
    """Resume a paused task"""
    server = resolve_server(server)
    ensure_server_available(server)
    token, _ = resolve_token(token, server)

    if not token:
        console.print("[red]✗ Error:[/red] No authentication token")
        sys.exit(1)

    console.print(f"[green]▶️  Resuming task {task_id}...[/green]")

    sio = socketio.Client()
    try:
        sio.connect(server)
        sio.emit('resume_task', {'task_id': task_id, 'token': token})
        time.sleep(1)
        console.print("[green]✓ Resume request sent[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
    finally:
        sio.disconnect()


@cli.command()
@click.argument('task_id')
@click.option('--token', help='API authentication token')
@click.option('--server', default=None, help='Backend server URL')
@click.confirmation_option(prompt='Are you sure you want to delete this task?')
def delete(task_id, token, server):
    """Delete a task (irreversible)"""
    server = resolve_server(server)
    ensure_server_available(server)
    token, _ = resolve_token(token, server)

    if not token:
        console.print("[red]✗ Error:[/red] No authentication token")
        sys.exit(1)

    console.print(f"[red]🗑️  Deleting task {task_id}...[/red]")

    sio = socketio.Client()
    try:
        sio.connect(server)
        sio.emit('delete_task', {'task_id': task_id, 'token': token})
        time.sleep(1)
        console.print("[green]✓ Task deleted[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
    finally:
        sio.disconnect()


@cli.command()
@click.option('--task', help='Stop a specific task by task_id')
@click.option('--all', 'stop_all', is_flag=True, help='Stop all active tasks')
@click.option('--token', help='API authentication token')
@click.option('--server', default=None, help='Backend server URL')
def stop(task, stop_all, token, server):
    """Stop running task(s)"""
    print_banner()

    server = resolve_server(server)
    ensure_server_available(server)
    token, _ = resolve_token(token, server)

    if not token:
        console.print("[red]✗ Error:[/red] No authentication token")
        sys.exit(1)

    # Validate options
    if task and stop_all:
        console.print("[red]✗ Error:[/red] Cannot use both --task and --all options together")
        sys.exit(1)

    # Interactive prompt if no arguments provided
    if not task and not stop_all:
        console.print("\n[yellow]⚠ No task specified.[/yellow]\n")
        console.print("[bold]Do you want to stop ALL active tasks?[/bold]")
        console.print("[dim]This will terminate all queued, started, running, and paused tasks.[/dim]\n")

        response = console.input("[bold cyan]Continue? (Y/N):[/bold cyan] ").strip().upper()

        if response == 'Y' or response == 'YES':
            stop_all = True
            console.print("[green]✓ Proceeding to stop all active tasks...[/green]\n")
        elif response == 'N' or response == 'NO':
            console.print("\n[yellow]Operation cancelled.[/yellow]")
            console.print("\n[bold]To stop a specific task, use:[/bold]")
            console.print("  [cyan]ds-cli stop --task <task_id>[/cyan]\n")
            console.print("[dim]Tip: Use 'ds-cli list' to view all your tasks and their IDs[/dim]\n")
            sys.exit(0)
        else:
            console.print("\n[red]✗ Invalid input. Please enter Y or N.[/red]")
            console.print("\n[yellow]Examples:[/yellow]")
            console.print("  ds-cli stop --task <task_id>")
            console.print("  ds-cli stop --all\n")
            sys.exit(1)

    sio = socketio.Client()
    stop_results = []

    try:
        sio.connect(server)

        if stop_all:
            # Stop all active tasks
            console.print("[yellow]🛑 Stopping all active tasks...[/yellow]\n")

            # Fetch all tasks
            try:
                tasks = fetch_tasks(server, token)
            except RuntimeError as err:
                console.print(f"[red]✗ Error:[/red] {err}")
                sys.exit(1)

            if not tasks:
                console.print("[yellow]No tasks found to stop[/yellow]")
                return

            # Filter active tasks
            active_statuses = ['queued', 'started', 'running', 'paused']
            active_tasks = [t for t in tasks if t.get('status', '').lower() in active_statuses]

            if not active_tasks:
                console.print("[yellow]No active tasks to stop[/yellow]")
                return

            console.print(f"[cyan]Found {len(active_tasks)} active task(s)[/cyan]\n")

            # Stop each active task
            for task_item in active_tasks:
                task_id = task_item.get('task_id', 'N/A')
                status = task_item.get('status', 'unknown')

                # Set up event handlers for this task
                stop_confirmed = threading.Event()
                error_message = [None]

                @sio.on('stop_confirmed')
                def on_stop_confirmed(data):
                    if data.get('task_id') == task_id:
                        stop_confirmed.set()

                @sio.on('error')
                def on_error(data):
                    error_message[0] = data.get('message', 'Unknown error')
                    stop_confirmed.set()

                # Emit stop request
                console.print(f"  Stopping task [cyan]{task_id}[/cyan] (status: {status})...")
                sio.emit('stop_task', {'task_id': task_id, 'token': token})

                # Wait for confirmation
                if stop_confirmed.wait(timeout=5):
                    if error_message[0]:
                        console.print(f"    [red]✗ Failed:[/red] {error_message[0]}")
                        stop_results.append({'task_id': task_id, 'success': False, 'error': error_message[0]})
                    else:
                        console.print(f"    [green]✓ Stopped successfully[/green]")
                        stop_results.append({'task_id': task_id, 'success': True})
                else:
                    console.print(f"    [yellow]⚠ Timeout waiting for confirmation[/yellow]")
                    stop_results.append({'task_id': task_id, 'success': False, 'error': 'Timeout'})

            # Summary
            console.print("\n[bold]Summary:[/bold]")
            successful = sum(1 for r in stop_results if r['success'])
            failed = len(stop_results) - successful
            console.print(f"  [green]✓ Successfully stopped:[/green] {successful}")
            if failed > 0:
                console.print(f"  [red]✗ Failed:[/red] {failed}")

        else:
            # Stop specific task
            task_id = task
            console.print(f"[yellow]🛑 Stopping task {task_id}...[/yellow]")

            # Set up event handlers
            stop_confirmed = threading.Event()
            error_message = [None]

            @sio.on('stop_confirmed')
            def on_stop_confirmed(data):
                if data.get('task_id') == task_id:
                    stop_confirmed.set()

            @sio.on('error')
            def on_error(data):
                error_message[0] = data.get('message', 'Unknown error')
                stop_confirmed.set()

            # Emit stop request
            sio.emit('stop_task', {'task_id': task_id, 'token': token})

            # Wait for confirmation
            if stop_confirmed.wait(timeout=5):
                if error_message[0]:
                    console.print(f"[red]✗ Error:[/red] {error_message[0]}")
                    sys.exit(1)
                else:
                    console.print("[green]✓ Task stopped successfully[/green]")
            else:
                console.print("[yellow]⚠ Timeout waiting for confirmation. Task may still be stopping...[/yellow]")

    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        sys.exit(1)
    finally:
        sio.disconnect()


@cli.command()
@click.argument('task_id')
@click.option('--token', help='API authentication token')
@click.option('--server', default=None, help='Backend server URL')
def findings(task_id, token, server):
    """View research findings for a task"""
    server = resolve_server(server)
    ensure_server_available(server)
    token, _ = resolve_token(token, server)

    if not token:
        console.print("[red]✗ Error:[/red] No authentication token")
        sys.exit(1)

    console.print(f"[bold cyan]🔬 Research Findings[/bold cyan]")
    console.print("─" * 60)
    console.print(f"Task: {task_id}\n")

    try:
        response = requests.get(
            f"{server}/api/tasks/{task_id}/findings",
            headers={'Authorization': f'Bearer {token}'},
            timeout=10
        )

        if response.status_code != 200:
            console.print(f"[red]✗ Error:[/red] HTTP {response.status_code}")
            sys.exit(1)

        data = response.json()
        findings_list = data.get('findings', [])

        if not findings_list:
            console.print("[yellow]No findings yet[/yellow]\n")
            return

        # Try interactive viewer first
        if view_findings_interactively(findings_list):
            console.print(f"\n[cyan]Total: {len(findings_list)} finding(s) loaded[/cyan]\n")
            return

        # Fallback: render static panels
        for finding in findings_list:
            panel = create_finding_panel(finding)
            console.print(panel)
            console.print("")

        console.print(f"[cyan]Total: {len(findings_list)} finding(s)[/cyan]\n")

    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.option('--show', 'action', flag_value='show', default=True, help='Show configuration (default)')
@click.option('--set', 'action', flag_value='set', help='Set a configuration value')
@click.option('--reset', 'action', flag_value='reset', help='Reset configuration to defaults')
@click.option('--key', help='Configuration key (see help for full list)')
@click.option('--value', help='Configuration value to set')
def config(action, key, value):
    """Display and manage CLI configuration settings

    \b
    Configuration Keys:
      install_dir             - Installation/config directory path
      workspace_dir           - Workspace directory path
      default_server          - Default server URL
      version                 - CLI version (read-only)
      validation_frequency    - Research validation frequency (high/medium/low/auto)
      baseline_upload         - Baseline upload preference (Y/H/ask)
      claude_code_max_retries - Max retries for Claude Code (0-10, default: 2)
      test_sh_max_retries     - Max retries for test.sh (0-10, default: 1)
      reconnection_attempts   - WebSocket reconnection attempts (1-1000, default: 400)
      heartbeat_interval      - Heartbeat interval in seconds (60-7200, default: 1800)

    \b
    Research Settings:
      • validation_frequency: Controls how often experiments are validated
        - high:   Validate every cycle (thorough but slower)
        - medium: Validate every ~3 cycles (balanced, default)
        - low:    Validate every ~10 cycles (faster but less thorough)
        - auto:   Let AI decide based on task complexity

      • baseline_upload: Whether to upload baseline results to backend
        - Y:    Always upload baseline (enables baseline comparison)
        - H:    Already uploaded for this task (skip upload)
        - ask:  Prompt each time (default)

    \b
    Examples:
      # Show current configuration
      deepscientist-cli config

      # Set validation frequency to high
      deepscientist-cli config --set --key validation_frequency --value high

      # Always upload baseline results
      deepscientist-cli config --set --key baseline_upload --value Y

      # Change default server
      deepscientist-cli config --set --key default_server --value http://localhost:5000

      # Reset validation frequency to default
      deepscientist-cli config --reset --key validation_frequency

      # Reset all configuration
      deepscientist-cli config --reset
    """
    print_banner()

    if action == 'set':
        if not key:
            console.print("[red]✗ Error:[/red] --key is required for --set")
            console.print("[dim]Example: deepscientist-cli config --set --key install_dir --value /path/to/dir[/dim]")
            sys.exit(1)
        if not value:
            console.print("[red]✗ Error:[/red] --value is required for --set")
            sys.exit(1)

        # Handle setting configuration
        install_config_path = CONFIG_DIR / "config.json"
        install_config = {}
        if install_config_path.exists():
            try:
                with open(install_config_path, 'r', encoding='utf-8') as f:
                    install_config = json.load(f)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load config: {e}[/yellow]")

        # Validate and set the key
        valid_keys = [
            'install_dir', 'default_server', 'workspace_dir', 'version',
            'validation_frequency', 'baseline_upload',
            'claude_code_max_retries', 'test_sh_max_retries',
            'reconnection_attempts', 'heartbeat_interval'
        ]
        if key not in valid_keys:
            console.print(f"[red]✗ Error:[/red] Invalid key '{key}'")
            console.print(f"[dim]Valid keys: {', '.join(valid_keys)}[/dim]")
            sys.exit(1)

        # Handle research settings (stored in cli_config.json)
        if key == 'validation_frequency':
            valid_freqs = ['high', 'medium', 'low', 'auto']
            if value.lower() not in valid_freqs:
                console.print(f"[red]✗ Error:[/red] Invalid validation frequency '{value}'")
                console.print(f"[dim]Valid values: {', '.join(valid_freqs)}[/dim]")
                sys.exit(1)
            save_validation_frequency(value.lower())
            console.print(f"[green]✓[/green] Validation frequency updated: [cyan]{value.lower()}[/cyan]")
            return

        if key == 'baseline_upload':
            valid_opts = ['Y', 'H', 'ask']
            if value.upper() not in valid_opts:
                console.print(f"[red]✗ Error:[/red] Invalid baseline upload option '{value}'")
                console.print(f"[dim]Valid values: Y, H, ask[/dim]")
                sys.exit(1)
            save_baseline_upload_default(value.upper())
            console.print(f"[green]✓[/green] Baseline upload preference updated: [cyan]{value.upper()}[/cyan]")
            return

        # Handle retry configuration
        if key in ['claude_code_max_retries', 'test_sh_max_retries']:
            try:
                retry_value = int(value)
                if retry_value < 0 or retry_value > 10:
                    console.print(f"[red]✗ Error:[/red] Retry value must be between 0-10")
                    console.print(f"[dim]Default: claude_code_max_retries=2, test_sh_max_retries=1[/dim]")
                    sys.exit(1)
                cli_config = load_cli_config()
                cli_config[key] = retry_value
                save_cli_config(cli_config)
                console.print(f"[green]✓[/green] {key} updated: [cyan]{retry_value}[/cyan] (total {retry_value + 1} attempts)")
                return
            except ValueError:
                console.print(f"[red]✗ Error:[/red] Invalid number: {value}")
                sys.exit(1)

        # Handle connection configuration
        if key == 'reconnection_attempts':
            try:
                attempts = int(value)
                if attempts < 1 or attempts > 1000:
                    console.print(f"[red]✗ Error:[/red] Reconnection attempts must be between 1-1000")
                    console.print(f"[dim]Default: 400 (approx 1 hour with 10s max delay)[/dim]")
                    sys.exit(1)
                max_minutes = (attempts * 10) // 60
                cli_config = load_cli_config()
                cli_config['reconnection_attempts'] = attempts
                save_cli_config(cli_config)
                console.print(f"[green]✓[/green] Reconnection attempts updated: [cyan]{attempts}[/cyan] (~{max_minutes} minutes max)")
                return
            except ValueError:
                console.print(f"[red]✗ Error:[/red] Invalid number: {value}")
                sys.exit(1)

        if key == 'heartbeat_interval':
            try:
                interval = int(value)
                if interval < 60 or interval > 7200:
                    console.print(f"[red]✗ Error:[/red] Heartbeat interval must be between 60-7200 seconds")
                    console.print(f"[dim]Default: 1800 (30 minutes)[/dim]")
                    sys.exit(1)
                minutes = interval // 60
                cli_config = load_cli_config()
                cli_config['heartbeat_interval'] = interval
                save_cli_config(cli_config)
                console.print(f"[green]✓[/green] Heartbeat interval updated: [cyan]{interval}s[/cyan] ({minutes} minutes)")
                return
            except ValueError:
                console.print(f"[red]✗ Error:[/red] Invalid number: {value}")
                sys.exit(1)

        # Special handling for install_dir
        if key == 'install_dir':
            value_path = Path(value).expanduser().resolve()
            if not value_path.exists():
                console.print(f"[yellow]⚠ Warning:[/yellow] Directory {value_path} does not exist")
                if click.confirm("Create it?", default=True):
                    value_path.mkdir(parents=True, exist_ok=True)
                    console.print(f"[green]✓[/green] Created directory: {value_path}")
                else:
                    console.print("[yellow]Configuration not updated[/yellow]")
                    sys.exit(0)
            value = str(value_path)

        old_value = install_config.get(key, 'Not set')
        install_config[key] = value
        install_config['updated_at'] = datetime.now().isoformat()

        # Save configuration
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(install_config_path, 'w', encoding='utf-8') as f:
            json.dump(install_config, f, indent=2)

        console.print(f"[green]✓[/green] Configuration updated:")
        console.print(f"  {key}: [dim]{old_value}[/dim] → [cyan]{value}[/cyan]")
        console.print()
        console.print("[yellow]⚠ Note:[/yellow] Restart the CLI for changes to take effect")
        return

    elif action == 'reset':
        if key:
            # Reset specific key
            # Handle research settings separately (stored in cli_config.json)
            if key == 'validation_frequency':
                cli_config = load_cli_config()
                old_value = cli_config.get('validation_frequency', 'medium')
                cli_config.pop('validation_frequency', None)
                save_cli_config(cli_config)
                console.print(f"[green]✓[/green] Reset validation_frequency: {old_value} → medium (default)")
                return

            if key == 'baseline_upload':
                cli_config = load_cli_config()
                old_value = cli_config.get('baseline_upload', 'ask')
                cli_config.pop('baseline_upload', None)
                save_cli_config(cli_config)
                console.print(f"[green]✓[/green] Reset baseline_upload: {old_value} → ask (default)")
                return

            # Handle retry settings reset
            if key in ['claude_code_max_retries', 'test_sh_max_retries']:
                cli_config = load_cli_config()
                defaults = {'claude_code_max_retries': 2, 'test_sh_max_retries': 1}
                old_value = cli_config.get(key, defaults[key])
                cli_config.pop(key, None)
                save_cli_config(cli_config)
                console.print(f"[green]✓[/green] Reset {key}: {old_value} → {defaults[key]} (default)")
                return

            # Handle connection settings reset
            if key == 'reconnection_attempts':
                cli_config = load_cli_config()
                old_value = cli_config.get('reconnection_attempts', 400)
                cli_config.pop('reconnection_attempts', None)
                save_cli_config(cli_config)
                console.print(f"[green]✓[/green] Reset reconnection_attempts: {old_value} → 400 (default, ~1 hour)")
                return

            if key == 'heartbeat_interval':
                cli_config = load_cli_config()
                old_value = cli_config.get('heartbeat_interval', 1800)
                cli_config.pop('heartbeat_interval', None)
                save_cli_config(cli_config)
                console.print(f"[green]✓[/green] Reset heartbeat_interval: {old_value}s → 1800s (default, 30 minutes)")
                return

            # Handle install config settings
            install_config_path = CONFIG_DIR / "config.json"
            if install_config_path.exists():
                try:
                    with open(install_config_path, 'r', encoding='utf-8') as f:
                        install_config = json.load(f)
                    if key in install_config:
                        old_value = install_config.pop(key)
                        with open(install_config_path, 'w', encoding='utf-8') as f:
                            json.dump(install_config, f, indent=2)
                        console.print(f"[green]✓[/green] Reset {key}: {old_value}")
                    else:
                        console.print(f"[yellow]Key '{key}' not found in configuration[/yellow]")
                except Exception as e:
                    console.print(f"[red]✗ Error:[/red] {e}")
        else:
            # Reset all configuration
            if click.confirm("⚠ Reset all configuration to defaults?", default=False):
                install_config_path = CONFIG_DIR / "config.json"
                if install_config_path.exists():
                    install_config_path.unlink()
                cli_config_path = CONFIG_FILE
                if cli_config_path.exists():
                    cli_config_path.unlink()
                console.print("[green]✓[/green] Configuration reset to defaults")
                console.print("[yellow]⚠ Note:[/yellow] Restart the CLI for changes to take effect")
            else:
                console.print("[yellow]Reset cancelled[/yellow]")
        return

    # Default: show configuration
    console.print("[bold cyan]⚙️  CLI Configuration[/bold cyan]")
    console.print("─" * 80)
    console.print()

    # Load configurations
    cli_config = load_cli_config()

    # Read installation config
    install_config_path = CONFIG_DIR / "config.json"
    install_config = {}
    if install_config_path.exists():
        try:
            with open(install_config_path, 'r', encoding='utf-8') as f:
                install_config = json.load(f)
        except:
            pass

    # Create configuration table
    config_table = Table(show_header=True, header_style="bold cyan", box=ROUNDED)
    config_table.add_column("Setting", style="cyan", width=25)
    config_table.add_column("Value", style="white", width=50)

    # Installation info
    install_dir = install_config.get('install_dir', str(CONFIG_DIR))
    installed_at = install_config.get('installed_at', 'Unknown')
    updated_at = install_config.get('updated_at', 'Never')
    version = install_config.get('version', 'Unknown')

    # Check environment variable overrides
    env_install_dir = os.environ.get('DEEPSCIENTIST_INSTALL_DIR')
    env_workspace_dir = os.environ.get('DEEPSCIENTIST_WORKSPACE_DIR')

    install_dir_display = install_dir
    if env_install_dir:
        install_dir_display = f"{install_dir} [yellow](overridden by env: {env_install_dir})[/yellow]"

    config_table.add_row("📁 Installation Directory", install_dir_display)
    config_table.add_row("📅 Installed At", installed_at)
    config_table.add_row("🔄 Last Updated", updated_at)
    config_table.add_row("🏷️  Version", version)
    config_table.add_row("", "")  # Separator

    # Server configuration
    default_server = cli_config.get('default_server', SERVER_URL)
    config_table.add_row("🌐 Default Server", default_server)

    # Token information
    servers = cli_config.get('servers', {})
    if servers:
        config_table.add_row("", "")  # Separator
        config_table.add_row("[bold]🔐 Configured Servers[/bold]", "")
        for server_url, server_info in servers.items():
            token = server_info.get('token', '')
            saved_at = server_info.get('saved_at', 'Unknown')
            masked = mask_token(token)
            config_table.add_row(f"  {server_url}", f"Token: {masked}")
            config_table.add_row("", f"Saved: {saved_at}")
    else:
        config_table.add_row("🔑 Token Status", "[yellow]No token configured[/yellow]")
        config_table.add_row("", "[dim]Run 'deepscientist-cli login' to configure[/dim]")

    # Workspace information
    workspace_dir = install_config.get('workspace_dir', str(CONFIG_DIR / "workspace"))
    workspace_dir_display = workspace_dir
    if env_workspace_dir:
        workspace_dir_display = f"{workspace_dir} [yellow](overridden by env: {env_workspace_dir})[/yellow]"

    config_table.add_row("", "")  # Separator
    config_table.add_row("📂 Workspace Directory", workspace_dir_display)
    config_table.add_row("📂 Config Directory (active)", str(CONFIG_DIR))
    config_table.add_row("📄 Config File", str(CONFIG_FILE))

    # Conda configuration
    conda_config = cli_config.get('conda', {})
    if conda_config:
        conda_env = conda_config.get('env_name', 'air')
        conda_base = conda_config.get('base_path', CONDA_BASE_PATH)
        config_table.add_row("", "")  # Separator
        config_table.add_row("🐍 Conda Environment", conda_env)
        config_table.add_row("🐍 Conda Base Path", conda_base)
    else:
        config_table.add_row("", "")  # Separator
        config_table.add_row("🐍 Conda Status", "[yellow]Not configured[/yellow]")
        config_table.add_row("", "[dim]Run 'deepscientist conda activate <env>' to configure[/dim]")
    config_table.add_row("📄 Install Config", str(install_config_path))

    # Validation frequency
    validation_frequency = cli_config.get('validation_frequency', 'medium')
    freq_descriptions = {
        'high': 'High (validate every cycle)',
        'medium': 'Medium (validate every ~3 cycles)',
        'low': 'Low (validate every ~10 cycles)',
        'auto': 'Auto (AI decides)'
    }
    freq_display = freq_descriptions.get(validation_frequency, f'{validation_frequency} (custom)')

    # Baseline upload setting
    baseline_upload = cli_config.get('baseline_upload')
    baseline_display = {
        'Y': 'Yes (always upload)',
        'N': 'No (never upload)',
        None: 'Ask each time (default)'
    }.get(baseline_upload, str(baseline_upload))

    config_table.add_row("", "")  # Separator
    config_table.add_row("🔍 Validation Frequency", freq_display)
    config_table.add_row("📤 Baseline Upload", baseline_display)

    # Retry settings
    retry_config = get_retry_config()
    claude_code_retries = retry_config.get('claude_code_max_retries', 2)
    test_sh_retries = retry_config.get('test_sh_max_retries', 1)
    config_table.add_row("🔄 Claude Code Max Retries", f"{claude_code_retries} (total {claude_code_retries + 1} attempts)")
    config_table.add_row("🔄 Test.sh Max Retries", f"{test_sh_retries} (total {test_sh_retries + 1} attempts)")

    # Connection settings
    conn_config = get_connection_config()
    reconnection_attempts = conn_config.get('reconnection_attempts', 400)
    heartbeat_interval = conn_config.get('heartbeat_interval', 1800)
    max_reconnect_minutes = (reconnection_attempts * 10) // 60
    heartbeat_minutes = heartbeat_interval // 60
    config_table.add_row("🔌 Reconnection Attempts", f"{reconnection_attempts} (~{max_reconnect_minutes} minutes max)")
    config_table.add_row("💓 Heartbeat Interval", f"{heartbeat_interval}s ({heartbeat_minutes} minutes)")

    # Last login
    last_login = cli_config.get('last_login_at', 'Never')
    config_table.add_row("", "")  # Separator
    config_table.add_row("🕒 Last Login", last_login)

    # Additional info
    console.print("[bold cyan]💡 Configuration Management[/bold cyan]")
    console.print("─" * 80)
    console.print()

    console.print("[bold]General Settings:[/bold]")
    console.print("  • Modify: [cyan]deepscientist-cli config --set --key <KEY> --value <VALUE>[/cyan]")
    console.print("  • Reset:  [cyan]deepscientist-cli config --reset [--key <KEY>][/cyan]")
    console.print("  • Help:   [cyan]deepscientist-cli config --help[/cyan]")
    console.print()
    console.print("  [dim]Valid keys: install_dir, workspace_dir, default_server, validation_frequency, baseline_upload, claude_code_max_retries, test_sh_max_retries, reconnection_attempts, heartbeat_interval[/dim]")
    console.print()

    console.print("[bold]🔍 Research Settings Configuration:[/bold]")
    console.print(f"  Current Validation Frequency: [cyan]{freq_display}[/cyan]")
    console.print(f"  Current Baseline Upload:      [cyan]{baseline_display}[/cyan]")
    console.print()
    console.print("  [bold yellow]To Change Research Settings:[/bold yellow]")
    console.print("    # Set validation frequency")
    console.print("    [cyan]deepscientist-cli config --set --key validation_frequency --value high[/cyan]")
    console.print("    [dim]Options: high, medium, low, auto[/dim]")
    console.print()
    console.print("    # Set baseline upload preference")
    console.print("    [cyan]deepscientist-cli config --set --key baseline_upload --value Y[/cyan]")
    console.print("    [dim]Options: Y (always), H (already uploaded), ask (prompt each time)[/dim]")
    console.print()
    console.print("    # Set retry configuration")
    console.print("    [cyan]deepscientist-cli config --set --key claude_code_max_retries --value 3[/cyan]")
    console.print("    [dim]Options: 0-10 (default: 2, total attempts = retries + 1)[/dim]")
    console.print("    [cyan]deepscientist-cli config --set --key test_sh_max_retries --value 2[/cyan]")
    console.print("    [dim]Options: 0-10 (default: 1, total attempts = retries + 1)[/dim]")
    console.print()
    console.print("  [bold yellow]To Change Connection Settings:[/bold yellow]")
    console.print("    # Set reconnection attempts")
    console.print("    [cyan]deepscientist-cli config --set --key reconnection_attempts --value 600[/cyan]")
    console.print("    [dim]Options: 1-1000 (default: 400, ~67 minutes max with 10s delay)[/dim]")
    console.print("    # Set heartbeat interval")
    console.print("    [cyan]deepscientist-cli config --set --key heartbeat_interval --value 3600[/cyan]")
    console.print("    [dim]Options: 60-7200 seconds (default: 1800, 30 minutes)[/dim]")
    console.print()
    console.print("    # Reset to defaults")
    console.print("    [cyan]deepscientist-cli config --reset --key validation_frequency[/cyan]")
    console.print("    [cyan]deepscientist-cli config --reset --key baseline_upload[/cyan]")
    console.print("    [cyan]deepscientist-cli config --reset --key claude_code_max_retries[/cyan]")
    console.print("    [cyan]deepscientist-cli config --reset --key reconnection_attempts[/cyan]")
    console.print()
    console.print("    # Reset ALL settings to defaults")
    console.print("    [cyan]deepscientist-cli config --reset[/cyan]")
    console.print()

    console.print("[bold]🔐 Authentication:[/bold]")
    console.print("  • Login: [cyan]deepscientist-cli login --token <TOKEN>[/cyan]")
    console.print()

    # Display configuration table at the bottom
    console.print("[bold cyan]📊 Complete Configuration Details[/bold cyan]")
    console.print("─" * 80)
    console.print()
    console.print(config_table)
    console.print()


# ============================================================================
# Conda Management Commands
# ============================================================================

@cli.group()
def conda():
    """Manage conda environments for DeepScientist"""
    pass


@conda.command('activate')
@click.argument('env_name')
def conda_activate(env_name):
    """Activate and save a conda environment"""
    print_banner()
    console.print(f"[bold cyan]🐍 Activating Conda Environment: {env_name}[/bold cyan]")
    console.print("─" * 60)

    # Detect conda base path if not already done
    if CONDA_BASE_PATH is None:
        detect_and_configure_conda()

    # Test if environment exists
    console.print(f"\n[cyan]Testing environment '{env_name}'...[/cyan]")

    test_command = f"eval \"$('{CONDA_BASE_PATH}/bin/conda' 'shell.bash' 'hook')\" && conda activate {env_name} && echo 'SUCCESS'"

    try:
        result = subprocess.run(
            ['bash', '-c', test_command],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and 'SUCCESS' in result.stdout:
            console.print(f"[green]✓ Environment '{env_name}' is valid and accessible[/green]")

            # Save to config
            save_conda_config(env_name, CONDA_BASE_PATH)

            # Update global variable
            global CONDA_ENV_NAME
            CONDA_ENV_NAME = env_name

            console.print(f"\n[bold green]✓ Conda environment configured successfully![/bold green]")
            console.print(f"[dim]Environment '{env_name}' will be used for all implementations[/dim]")

        else:
            error_output = result.stderr.strip() if result.stderr else "Unknown error"
            console.print(f"[red]✗ Failed to activate environment '{env_name}'[/red]")
            console.print(f"[yellow]Error:[/yellow] {error_output}")
            console.print(f"\n[dim]💡 Tip: Run 'deepscientist conda list' to see available environments[/dim]")
            sys.exit(1)

    except subprocess.TimeoutExpired:
        console.print(f"[red]✗ Timeout while testing environment '{env_name}'[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        sys.exit(1)


@conda.command('list')
def conda_list():
    """List all available conda environments"""
    print_banner()
    console.print("[bold cyan]🐍 Available Conda Environments[/bold cyan]")
    console.print("─" * 60)

    # Detect conda base path if not already done
    if CONDA_BASE_PATH is None:
        detect_and_configure_conda()

    try:
        result = subprocess.run(
            ['conda', 'env', 'list'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            # Get current configured environment
            conda_config = get_conda_config()
            current_env = conda_config.get('env_name')

            console.print(f"\n[cyan]Conda Base:[/cyan] {CONDA_BASE_PATH}\n")

            # Parse and display environments
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.split()
                if len(parts) >= 1:
                    env_name = parts[0]
                    is_current = env_name == current_env
                    marker = " [bold green]✓ (configured)[/bold green]" if is_current else ""
                    console.print(f"  • {env_name}{marker}")

            console.print(f"\n[dim]💡 Tip: Use 'deepscientist conda activate <env>' to configure[/dim]")

        else:
            console.print(f"[red]✗ Failed to list conda environments[/red]")
            console.print(f"[yellow]Error:[/yellow] {result.stderr}")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        sys.exit(1)


@conda.command('info')
def conda_info():
    """Show current conda configuration"""
    print_banner()
    console.print("[bold cyan]🐍 Conda Configuration[/bold cyan]")
    console.print("─" * 60)

    # Get conda config
    conda_config = get_conda_config()
    env_name = conda_config.get('env_name', 'Not configured')
    base_path = conda_config.get('base_path', 'Not detected')

    # Detect conda if not done
    if CONDA_BASE_PATH is None:
        detect_and_configure_conda()

    # Create info table
    info_table = Table(show_header=False, box=ROUNDED)
    info_table.add_column("Setting", style="cyan")
    info_table.add_column("Value", style="white")

    info_table.add_row("🐍 Configured Environment", env_name)
    info_table.add_row("📂 Conda Base Path", str(base_path))
    info_table.add_row("🔧 Detected Base Path", str(CONDA_BASE_PATH))

    # Test if current env is valid
    if env_name != 'Not configured':
        test_command = f"eval \"$('{CONDA_BASE_PATH}/bin/conda' 'shell.bash' 'hook')\" && conda activate {env_name} && echo 'OK'"
        try:
            result = subprocess.run(
                ['bash', '-c', test_command],
                capture_output=True,
                text=True,
                timeout=5
            )
            status = "[green]✓ Valid[/green]" if result.returncode == 0 else "[red]✗ Invalid[/red]"
        except:
            status = "[yellow]⚠ Unknown[/yellow]"

        info_table.add_row("✓ Environment Status", status)

    console.print()
    console.print(info_table)
    console.print()

    # Show activation command
    if env_name != 'Not configured':
        activate_cmd = get_conda_activate_command(env_name)
        console.print("[dim]Activation command:[/dim]")
        console.print(f"[dim]{activate_cmd}[/dim]")
        console.print()

    console.print("[dim]💡 Commands:[/dim]")
    console.print("[dim]  • deepscientist conda list      - List available environments[/dim]")
    console.print("[dim]  • deepscientist conda activate <env>  - Configure environment[/dim]")


@conda.command('deactivate')
def conda_deactivate():
    """Reset conda configuration to default"""
    print_banner()
    console.print("[bold cyan]🐍 Resetting Conda Configuration[/bold cyan]")
    console.print("─" * 60)

    # Reset to default
    save_conda_config('air')  # Reset to default 'air' environment

    global CONDA_ENV_NAME
    CONDA_ENV_NAME = 'air'

    console.print(f"\n[green]✓ Conda configuration reset to default (air)[/green]")
    console.print(f"[dim]Run 'deepscientist conda activate <env>' to configure a different environment[/dim]")


@cli.command()
@click.argument('feedback_text', required=True)
@click.option('--type', 'feedback_type',
              type=click.Choice(['bug', 'feature', 'question', 'info'], case_sensitive=False),
              default='info',
              help='Type of feedback (bug, feature, question, info). Default: info')
@click.option('--server', default=None, help=f'Backend server URL (default: {SERVER_URL})')
@click.option('--title', help='Optional title for the feedback (defaults to first 50 chars)')
def feedback(feedback_text, feedback_type, server, title):
    """Submit feedback to DeepScientist platform

    FEEDBACK_TEXT: The feedback message content

    Examples:
        deepscientist-cli feedback "CLI crashes when submitting tasks" --type bug
        deepscientist-cli feedback "Add dark mode to CLI" --type feature
        deepscientist-cli feedback "How do I check task status?" --type question
        deepscientist-cli feedback "Great tool! Keep up the good work!" --type info
    """
    print_banner()

    # Resolve server and get token
    server = resolve_server(server)
    token = get_saved_token(server)

    if not token:
        console.print("[red]✗ Error:[/red] No authentication token found")
        console.print("[dim]Please run 'deepscientist-cli login' first[/dim]")
        sys.exit(1)

    # Ensure server is available
    try:
        ensure_server_available(server)
    except Exception as e:
        console.print(f"[red]✗ Server unavailable:[/red] {e}")
        sys.exit(1)

    console.print("[bold cyan]📝 Submitting Feedback[/bold cyan]")
    console.print("─" * 60)

    # Prepare feedback data
    if not title:
        # Use first 50 characters of feedback as title
        title = feedback_text[:50] + ('...' if len(feedback_text) > 50 else '')

    # Normalize feedback type - use 'info' for CLI but send 'question' to backend for API compatibility
    backend_type = feedback_type if feedback_type != 'info' else 'question'

    feedback_data = {
        'title': title,
        'description': feedback_text,
        'type': backend_type,
        'priority': 'medium'  # Default priority
    }

    console.print(f"[dim]Server:[/dim] {server}")
    console.print(f"[dim]Type:[/dim] {backend_type}")
    console.print(f"[dim]Title:[/dim] {title}")
    console.print(f"[dim]Description:[/dim] {feedback_text[:100]}{'...' if len(feedback_text) > 100 else ''}")
    console.print()

    # Submit feedback
    with console.status("[bold cyan]Submitting feedback...[/bold cyan]"):
        try:
            response = requests.post(
                f"{server}/api/feedback",
                headers={
                    'Authorization': f'Bearer {token}',
                    'Content-Type': 'application/json'
                },
                json=feedback_data,
                timeout=15
            )
        except requests.RequestException as e:
            console.print(f"[red]✗ Failed to submit feedback:[/red] {e}")
            sys.exit(1)

    if response.status_code == 200:
        result = response.json()
        if result.get('success'):
            feedback_id = result.get('feedback_id')
            console.print(f"[bold green]✅ Feedback submitted successfully![/bold green]")
            console.print(f"[dim]Feedback ID: {feedback_id}[/dim]")
            console.print()
            console.print("[bold]📋 Summary:[/bold]")
            console.print(f"  • Type: {backend_type}")
            console.print(f"  • Title: {title}")
            console.print(f"  • Status: Submitted to administrators")
            console.print()
            console.print("[dim]💬 Your feedback has been sent to the DeepScientist team.[/dim]")
            console.print("[dim]   We'll review it and respond if needed.[/dim]")
        else:
            console.print("[red]✗ Failed to submit feedback[/red]")
            console.print(f"[yellow]Response:[/yellow] {result}")
            sys.exit(1)
    elif response.status_code == 401:
        console.print("[red]✗ Authentication failed[/red]")
        console.print("[dim]Your token may have expired. Please run 'deepscientist-cli login' again.[/dim]")
        sys.exit(1)
    else:
        console.print(f"[red]✗ Server error ({response.status_code}):[/red]")
        try:
            error_data = response.json()
            console.print(f"[yellow]Error:[/yellow] {error_data.get('error', 'Unknown error')}")
        except:
            console.print(f"[yellow]Response:[/yellow] {response.text}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
