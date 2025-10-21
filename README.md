

# DeepScientist CLI Tool

⚠️ **This CLI is intended only for users who already hold a valid DeepScientist API token from http://deepscientist.cc.**

## Overview

The DeepScientist CLI empowers researchers to harness the full potential of autonomous scientific discovery without the prohibitive setup overhead. It democratizes access to the state-of-the-art DeepScientist system by encapsulating its complex operational requirements into intuitive commands. This tool is designed to accelerate the research lifecycle, allowing users to seamlessly submit tasks, monitor progress, and iterate on ideas at a scale previously reserved for highly specialized teams.


## Prerequisites

Before installing DeepScientist CLI, ensure your environment meets the following requirements:

### System Requirements

1. **Docker Environment** (optional but highly recommended)
   - Load and enter a Docker image on your GPU server
   - Mount appropriate file paths for your workspace

2. **User Permissions**
   - Must use a **non-root user** for security and best practices
   - If currently using root account, create a new user: `adduser <username>`

3. **Python Environment**
   - Python 3.11+ required
   - Recommended: Create a Conda environment
   ```bash
   conda create -n air python=3.11
   conda activate air
   ```

4. **Claude Code**
   - Pre-install Claude Code via npm: https://github.com/anthropics/claude-code
   - Free GLM 4.6 API provided by DeepScientist (no payment required)
   - Alternatively, you can use Claude by paying Anthropic directly

### Task Configuration Requirements

Your research project must be properly configured for one-click execution:

1. **Single-Command Execution**
   - Baseline code must be executable with a single command (e.g., `bash test.sh`)
   - Script must run the complete experiment
   - All logs and results must print to standard output (stdout)
   - Output should be redirected and saved to a `.log` file

2. **Required Files**
   - **CLAUDE.md** - Task and code description
   - **test.sh** - One-click execution script
   - **latex.tex** - Research documentation
   - **.ep** - Exclusion patterns for packaging

   For detailed requirements, see [Docs](http://deepscientist.cc)

3. **Code Standardization**
   - Follow proper code formatting guidelines
   - Reference: [Code Formatting Guide](https://drive.google.com/file/d/1CK8SvfSoI8e8zu59YOzvBQIpsJeZL4Nx/view?usp=sharing)

## Installation

### One-Click Installation

Simply run the installation script to set up everything automatically:

```bash
cd /path/to/DeepScientist-CLI
bash install_cli.sh
```

You may also provide a custom installation directory:

```bash
bash install_cli.sh /custom/path/for/.deepscientist
```

The installer performs the following steps (mirroring `install_cli.sh`):

1. Validates Python ≥ 3.11 and the availability of `pip`.
2. Detects whether you are inside a Conda environment and routes dependency installs accordingly.
3. Copies the CLI source into the installation directory and creates lightweight wrapper executables (`deepscientist`, `deepscientist-cli`, `ds-cli`) under both the install path and `~/.local/bin`.
4. Prompts for your API token (required for deep integration). Tokens are verified against the DeepScientist backend before proceeding.  
   - If verification succeeds, the script automatically runs `deepscientist_cli.py login` so your token is saved in `~/.deepscientist/cli_config.json`.  
   - Supported (institutional) users are asked whether to use DeepScientist-provided Claude Code resources or their own Anthropic API key. Normal users default to local configuration.
5. Installs Claude Code tooling via `src/claude_code_deepscientist_env.sh`, configuring the appropriate API endpoint when DeepScientist resources are selected.
6. Writes installation metadata (including version `v0.2.1`) to both the installation directory and `~/.deepscientist/config.json`.
7. Appends the installation paths to your shell profile (`.bashrc` or `.zshrc`) if they are not already present.
8. Reminds you at completion that you can launch the CLI with `deepscientist-cli` or `ds-cli`.

### What Gets Installed

The installation script automatically:
1. Ensures Node.js (via `nvm`) is available so Claude Code can run.
2. Installs the Claude Code CLI (`@anthropic-ai/claude-code`) and optionally binds it to DeepScientist endpoints.
3. Installs Python dependencies found in `requirements.txt` (inside Conda if active, otherwise under the current user).
4. Generates wrapper commands (`deepscientist-cli`, `deepscientist`, `ds-cli`) and symlinks them into both `~/.local/bin` and `~/.deepscientist/bin`, overwriting any stale wrappers from previous installs.
5. Persists configuration files describing the install path, bin directory, and CLI version (`v0.2.1`).

### Verify Installation

After installation completes:

```bash
# Reload shell configuration
source ~/.bashrc  # or source ~/.zshrc

# Activate your Python environment (if using conda)
conda activate air  # replace 'air' with your chosen env

# Verify CLI command
ds-cli --help

# Optional: refresh shell command cache if ds-cli is not immediately found
hash -r

# Verify Claude Code (Option 1 users)
claude --version
```

## Quick Start

### For Option 1 Users (Free Compute Resources)

Use the CLI immediately after installation:

```bash
# Submit research task
conda run -n <ENV_NAME> ds-cli submit /path/to/your/project \
  --query "Focus on <limitations/methods/objectives>" \
  --gpu 0

# Monitor task progress
ds-cli monitor <TASK_ID>

# List all tasks
ds-cli list

# Stop tasks (interactive - will prompt Y/N)
ds-cli stop
```

## Prepare Your Research Project

Your research project should include:

1. **CLAUDE.md** - Describes research objectives and goals
2. **test.sh** - Test script that validates research results
3. **Code files** - Your research codebase (must be < 50,000 tokens)

## Available Commands

### Basic Commands

- `conda run -n <ENV_NAME> ds-cli submit <path> --query "Focus on <limitations/methods/objectives>" --gpu <ID>` - Submit a research task with a clear research directive and selected GPU device
- `ds-cli monitor <task_id>` - Monitor task progress in real-time
- `ds-cli list` - List all your tasks
- `ds-cli findings <task_id>` - View research findings
- `ds-cli login` - Configure server connection (Option 2 users)
- `ds-cli --help` - Show all available commands

### Task Control Commands

#### Stop Running Tasks

The `stop` command allows you to terminate running tasks individually or all at once:

**Interactive mode (recommended):**
```bash
ds-cli stop
```
When you run `stop` without arguments, the CLI will ask if you want to stop all tasks:
- Enter `Y` (Yes) to stop all active tasks
- Enter `N` (No) to cancel and see usage instructions

**Stop a specific task:**
```bash
ds-cli stop --task <task_id>
```

**Stop all active tasks (non-interactive):**
```bash
ds-cli stop --all
```

**Examples:**
```bash
# Interactive mode - CLI will prompt for confirmation
ds-cli stop

# Stop a specific task by its ID
ds-cli stop --task 4036e2ef-be65-4aa5-ae50-f54cdf148713

# Stop all currently running, queued, or paused tasks (skip prompt)
ds-cli stop --all
```

**Features:**
- **Interactive confirmation** when no arguments are provided (safer)
- Real-time confirmation from the backend server
- Detailed status updates for each task when using `--all`
- Summary statistics showing successful and failed stops
- Graceful termination that allows tasks to clean up resources

**Notes:**
- Interactive mode (no arguments) will prompt you before stopping all tasks
- You can specify either `--task <task_id>` OR `--all`, but not both
- Only active tasks (queued, started, running, paused) will be stopped
- The backend will send confirmation via WebSocket events
- Tasks are marked as 'terminated' in the database
- Use `ds-cli list` to view all your tasks and their IDs


## Getting Help

- Official Website: http://deepscientist.cc
- GitHub Issues: Submit bugs and feature requests
- Documentation: Full guides available in `docs/` directory

For installation troubleshooting, inspect `install_cli.sh`—the script is heavily commented and reflects exactly what the installer performs.

## File Structure

```
cli/
├── install_cli.sh                      # One-click installation script
├── requirements.txt                    # Python dependencies
├── README.md                           # This document
└── src/
    ├── deepscientist_cli.py           # Main CLI program
    └── claude_code_deepscientist_env.sh # Claude Code installer
```
