#!/bin/bash
# DeepScientist CLI Installation Script
# Supports Linux and macOS
# Usage: bash install_cli.sh [INSTALL_PATH]
#   INSTALL_PATH: Optional custom installation directory (default: ~/.deepscientist)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# CLI metadata
CLI_VERSION="v0.2.2"

# Configuration
# Support custom installation path from command line argument
if [ -n "$1" ]; then
    INSTALL_DIR="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    echo -e "${BLUE}Using custom installation path: $INSTALL_DIR${NC}"
else
    INSTALL_DIR="${HOME}/.deepscientist"
fi

BIN_DIR="${HOME}/.local/bin"
CLI_SCRIPT="deepscientist_cli.py"
REPO_URL="https://raw.githubusercontent.com/ResearAI/DeepScientist-CLI/main"

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║         DeepScientist CLI Installation Script            ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Reject executions using sudo or root
if [ "$EUID" -eq 0 ] || [ -n "$SUDO_UID" ] || [ -n "$SUDO_USER" ]; then
   echo -e "${RED}✗ Please do not run this script with sudo or as root${NC}"
   echo -e "${YELLOW}  Switch to a standard user and rerun to avoid permission conflicts${NC}"
   exit 1
fi

# Prompt for DeepScientist token configuration
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  DeepScientist Token Configuration${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo -e "${RED}✗ Unsupported operating system: $OSTYPE${NC}"
    echo -e "${YELLOW}  This script supports Linux and macOS only${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Detected OS: $OS"

# Check Python version and environment
echo ""
echo -e "${BLUE}Checking Python installation...${NC}"

# Detect if running in conda environment
IN_CONDA=false
if [ -n "$CONDA_DEFAULT_ENV" ] || [ -n "$CONDA_PREFIX" ]; then
    IN_CONDA=true
    PYTHON_CMD="python"
    PIP_CMD="pip"
    echo -e "${GREEN}✓${NC} Detected conda environment: ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
else
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

if command -v $PYTHON_CMD &> /dev/null; then
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"

    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
        echo -e "${RED}✗ Python 3.11+ required, but found $PYTHON_VERSION${NC}"
        echo ""
        echo "Please install Python 3.11 or higher:"
        if [ "$IN_CONDA" = true ]; then
            echo "  conda install python>=3.11"
        elif [ "$OS" == "linux" ]; then
            echo "  apt install python3.11 (may require administrator privileges)"
        elif [ "$OS" == "macos" ]; then
            echo "  brew install python@3.11"
        fi
        exit 1
    fi
else
    echo -e "${RED}✗ Python not found${NC}"
    echo ""
    echo "Please install Python 3.11 or higher:"
    if [ "$IN_CONDA" = true ]; then
        echo "  conda install python>=3.11"
    elif [ "$OS" == "linux" ]; then
        echo "  apt install python3.11 (may require administrator privileges)"
    elif [ "$OS" == "macos" ]; then
        echo "  brew install python@3.11"
    fi
    exit 1
fi

# Check pip
if ! command -v $PIP_CMD &> /dev/null; then
    echo -e "${RED}✗ $PIP_CMD not found${NC}"
    echo ""
    echo "Attempting to bootstrap pip..."
    if $PYTHON_CMD -m ensurepip --upgrade >/dev/null 2>&1; then
        hash -r
    fi

    if ! command -v $PIP_CMD &> /dev/null; then
        echo -e "${RED}✗ Unable to install pip automatically${NC}"
        if [ "$IN_CONDA" = true ]; then
            echo "  pip should be included with conda. Try: conda install pip"
        elif [ "$OS" == "linux" ]; then
            echo "  Install pip3 via your package manager (e.g. apt install python3-pip) and re-run"
        elif [ "$OS" == "macos" ]; then
            echo "  Install pip3 via Homebrew (e.g. brew install pipx && pipx ensurepath)"
        fi
        exit 1
    fi
fi

echo -e "${GREEN}✓${NC} $PIP_CMD found"

# Create installation directory
echo ""
echo -e "${BLUE}Creating installation directory...${NC}"
mkdir -p "$INSTALL_DIR"
echo -e "${GREEN}✓${NC} Created $INSTALL_DIR"

# Download CLI script
echo ""
echo -e "${BLUE}Downloading CLI script...${NC}"

# For local installation (development)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/src/$CLI_SCRIPT" ]; then
    echo -e "${YELLOW}ℹ${NC} Using local CLI script"
    cp "$SCRIPT_DIR/src/$CLI_SCRIPT" "$INSTALL_DIR/$CLI_SCRIPT"
else
    echo -e "${YELLOW}ℹ${NC} Downloading from repository"
    if command -v curl &> /dev/null; then
        curl -fsSL "$REPO_URL/src/$CLI_SCRIPT" -o "$INSTALL_DIR/$CLI_SCRIPT"
    elif command -v wget &> /dev/null; then
        wget -q "$REPO_URL/src/$CLI_SCRIPT" -O "$INSTALL_DIR/$CLI_SCRIPT"
    else
        echo -e "${RED}✗ Neither curl nor wget found${NC}"
        echo "  Please install curl or wget"
        exit 1
    fi
fi

chmod +x "$INSTALL_DIR/$CLI_SCRIPT"
echo -e "${GREEN}✓${NC} Downloaded CLI script"

# Install Python dependencies
echo ""
echo -e "${BLUE}Installing Python dependencies...${NC}"

REQUIREMENTS_FILE_LOCAL="$SCRIPT_DIR/requirements.txt"
REQUIREMENTS_TEMP=""

install_requirements() {
    local requirements_path="$1"

    # Determine pip install arguments based on environment
    local PIP_ARGS="-r $requirements_path"

    if [ "$IN_CONDA" = true ]; then
        # In conda environment: install directly to conda environment (no --user)
        echo -e "${YELLOW}ℹ${NC} Installing to conda environment: $CONDA_DEFAULT_ENV"
        if ! $PIP_CMD install $PIP_ARGS; then
            echo -e "${RED}✗ Failed to install Python dependencies from $requirements_path${NC}"
            [ -n "$REQUIREMENTS_TEMP" ] && rm -f "$REQUIREMENTS_TEMP"
            exit 1
        fi
    else
        # Not in conda: install to user directory (with --user)
        echo -e "${YELLOW}ℹ${NC} Installing to user directory (~/.local)"
        if ! $PIP_CMD install --user $PIP_ARGS; then
            echo -e "${RED}✗ Failed to install Python dependencies from $requirements_path${NC}"
            [ -n "$REQUIREMENTS_TEMP" ] && rm -f "$REQUIREMENTS_TEMP"
            exit 1
        fi
    fi
}

if [ -f "$REQUIREMENTS_FILE_LOCAL" ]; then
    echo -e "${YELLOW}ℹ${NC} Installing dependencies from local requirements.txt"
    install_requirements "$REQUIREMENTS_FILE_LOCAL"
else
    echo -e "${YELLOW}ℹ${NC} Downloading requirements.txt from repository"
    if command -v curl &> /dev/null; then
        REQUIREMENTS_TEMP="$(mktemp)"
        curl -fsSL "$REPO_URL/requirements.txt" -o "$REQUIREMENTS_TEMP"
    elif command -v wget &> /dev/null; then
        REQUIREMENTS_TEMP="$(mktemp)"
        wget -q "$REPO_URL/requirements.txt" -O "$REQUIREMENTS_TEMP"
    else
        echo -e "${RED}✗ Neither curl nor wget found to download requirements${NC}"
        exit 1
    fi

    if [ ! -s "$REQUIREMENTS_TEMP" ]; then
        echo -e "${RED}✗ Unable to retrieve requirements.txt${NC}"
        rm -f "$REQUIREMENTS_TEMP"
        exit 1
    fi

    install_requirements "$REQUIREMENTS_TEMP"
    rm -f "$REQUIREMENTS_TEMP"
fi

echo -e "${GREEN}✓${NC} Installed Python dependencies"

# Create executable wrappers early so we can invoke CLI immediately
echo ""
echo -e "${BLUE}Creating executable wrappers...${NC}"

# Determine which Python command to use in the wrapper
if [ "$IN_CONDA" = true ]; then
    WRAPPER_PYTHON_CMD="python"
else
    WRAPPER_PYTHON_CMD="python3"
fi

WRAPPER_CONTENT="#!/bin/bash
# DeepScientist CLI wrapper
$WRAPPER_PYTHON_CMD $INSTALL_DIR/$CLI_SCRIPT \"\$@\"
"

ALIASES=("deepscientist-cli" "deepscientist" "ds-cli")

mkdir -p "$BIN_DIR"
DEFAULT_ALIAS_DIR="$HOME/.deepscientist/bin"
mkdir -p "$DEFAULT_ALIAS_DIR"

# Remove stale wrappers before linking new ones
for DIR in "$BIN_DIR" "$DEFAULT_ALIAS_DIR"; do
    for NAME in "${ALIASES[@]}"; do
        rm -f "$DIR/$NAME"
    done
done

for NAME in "${ALIASES[@]}"; do
    echo "$WRAPPER_CONTENT" > "$INSTALL_DIR/$NAME"
    chmod +x "$INSTALL_DIR/$NAME"
    ln -sf "$INSTALL_DIR/$NAME" "$BIN_DIR/$NAME"
    ln -sf "$INSTALL_DIR/$NAME" "$DEFAULT_ALIAS_DIR/$NAME"
done
echo -e "${GREEN}✓${NC} Installed commands to $BIN_DIR: ${ALIASES[*]}"

# Helper to write a fresh CLI configuration file
write_fresh_cli_config() {
    local token="$1"
    local server="$2"
    local login_at
    login_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    local config_dir="$HOME/.deepscientist"
    local config_file="$config_dir/cli_config.json"

    mkdir -p "$config_dir"

    cat > "$config_file" << EOF
{
  "version": "${CLI_VERSION}",
  "token": "$token",
  "default_server": "$server",
  "servers": {
    "$server": {
      "token": "$token",
      "saved_at": "$login_at",
      "last_login_at": "$login_at"
    }
  },
  "last_login_at": "$login_at"
}
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} CLI configuration initialized at $config_file"
    else
        echo -e "${RED}✗ Failed to write CLI configuration to $config_file${NC}"
    fi
}

# Helper to run CLI login so credentials are stored consistently
run_cli_auto_login() {
    local token="$1"
    local server="$2"
    local CLI_LOGIN_OUTPUT
    local CLI_LOGIN_EXIT

    echo ""
    echo -e "${BLUE}Running DeepScientist CLI login to persist credentials...${NC}"

    set +e
    CLI_LOGIN_OUTPUT=$($WRAPPER_PYTHON_CMD "$INSTALL_DIR/$CLI_SCRIPT" login --token "$token" --server "$server" 2>&1)
    CLI_LOGIN_EXIT=$?
    set -e

    if [ $CLI_LOGIN_EXIT -eq 0 ]; then
        echo -e "${GREEN}✓${NC} DeepScientist CLI login completed and token stored"
        write_fresh_cli_config "$token" "$server"
        return 0
    else
        echo -e "${RED}✗${NC} DeepScientist CLI login failed (exit code $CLI_LOGIN_EXIT)"
        printf '%s\n' "$CLI_LOGIN_OUTPUT"
        echo -e "${YELLOW}ℹ${NC} You can retry manually with: deepscientist-cli login --token YOUR_TOKEN --server $server"
        return 1
    fi
}

# Login variables - using built-in server address
DEEPSCIENTIST_HOST="deepscientist.ai-researcher.net"
DEEPSCIENTIST_PORT="8888"
SERVER_URL="http://${DEEPSCIENTIST_HOST}:${DEEPSCIENTIST_PORT}"
DEEPSCIENTIST_TOKEN=""
DETECTED_USER_TYPE="unknown"
VERIFICATION_SUCCESS=false
SUPPORTED_USER=false

# Token verification and automatic login function
verify_and_login_token() {
    local token="$1"
    echo ""
    echo -e "${BLUE}Verifying token with DeepScientist backend...${NC}"
    echo -e "${BLUE}Server: $SERVER_URL${NC}"
    echo ""

    set +e
    USER_INFO_JSON=$($PYTHON_CMD - "$SERVER_URL" "$token" <<'PY'
import json
import sys
import urllib.request
import urllib.error

server = sys.argv[1].rstrip('/')
token = sys.argv[2]

payload = json.dumps({"token": token}).encode('utf-8')
request = urllib.request.Request(
    f"{server}/api/auth/verify",
    data=payload,
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(request, timeout=10) as response:
        body_bytes = response.read()
except urllib.error.HTTPError as exc:
    try:
        error_body = exc.read().decode('utf-8')
    except Exception:
        error_body = str(exc)
    print(json.dumps({
        "ok": False,
        "error": "HTTP {}: {}".format(exc.code, error_body[:200])
    }))
    sys.exit(1)
except Exception as exc:
    print(json.dumps({"ok": False, "error": str(exc)}))
    sys.exit(1)

try:
    body_text = body_bytes.decode("utf-8")
    data = json.loads(body_text)
except (UnicodeDecodeError, json.JSONDecodeError) as exc:
    preview = body_bytes[:200].decode("utf-8", errors="replace") if body_bytes else ""
    print(json.dumps({
        "ok": False,
        "error": "Invalid JSON response: {}".format(exc),
        "raw": preview
    }))
    sys.exit(1)

user = data.get("user", {}) if isinstance(data, dict) else {}
result = {
    "ok": True,
    "user_type": user.get("user_type", "normal"),
    "api_verified": bool(user.get("api_verified")),
    "username": user.get("username"),
    "role": user.get("role")
}
print(json.dumps(result))
PY
)
    VERIFY_EXIT=$?
    set -e

    if [ $VERIFY_EXIT -eq 0 ] && [ -n "$USER_INFO_JSON" ]; then
        USER_PARSE=$(
            $PYTHON_CMD - "$USER_INFO_JSON" <<'PY'
import json
import sys

if len(sys.argv) < 2:
    print("ERROR|Missing verification payload")
    sys.exit(0)

raw = sys.argv[1]

try:
    data = json.loads(raw)
except json.JSONDecodeError as exc:
    message = "Failed to parse verification response: {}; raw: {}".format(
        exc, (raw.strip()[:200] or "Empty response")
    )
    print("ERROR|{}".format(message))
    sys.exit(0)
except Exception as exc:
    message = "Unexpected parser failure: {}; raw: {}".format(
        exc, (raw.strip()[:200] or "Empty response")
    )
    print("ERROR|{}".format(message))
    sys.exit(0)

if not data.get("ok"):
    print("ERROR|{}".format(data.get("error", "Unknown error")))
else:
    status_line = "OK|{}|{}|{}".format(
        data.get("user_type", "normal"),
        "1" if data.get("api_verified") else "0",
        data.get("username") or ""
    )
    print(status_line)
PY
        )
        IFS='|' read -r PARSE_STATUS USER_TYPE_FLAG API_VERIFIED_FLAG USERNAME <<<"$USER_PARSE"

        if [ "$PARSE_STATUS" = "OK" ]; then
            VERIFICATION_SUCCESS=true
            DETECTED_USER_TYPE="$USER_TYPE_FLAG"
            SUPPORTED_USER=false
            if [ "$DETECTED_USER_TYPE" = "supported" ]; then
                SUPPORTED_USER=true
            fi

            if [ "$API_VERIFIED_FLAG" = "1" ]; then
                API_STATUS_TEXT="Verified"
            else
                API_STATUS_TEXT="Not Verified"
            fi

            DISPLAY_NAME=${USERNAME:-"Unknown user"}
            echo -e "${GREEN}✓${NC} Authenticated as ${DISPLAY_NAME} (${DETECTED_USER_TYPE} user, API ${API_STATUS_TEXT})"

            echo ""
            if run_cli_auto_login "$token" "$SERVER_URL"; then
                return 0
            else
                VERIFICATION_SUCCESS=false
                SUPPORTED_USER=false
                DETECTED_USER_TYPE="unknown"
                return 1
            fi
        else
            echo -e "${RED}✗ Token verification failed:${NC} ${USER_TYPE_FLAG}"
            VERIFICATION_SUCCESS=false
            SUPPORTED_USER=false
            DETECTED_USER_TYPE="unknown"
            return 1
        fi
    else
        echo -e "${RED}✗ Token verification failed.${NC}"
        printf '%s\n' "$USER_INFO_JSON"
        VERIFICATION_SUCCESS=false
        SUPPORTED_USER=false
        DETECTED_USER_TYPE="unknown"
        return 1
    fi
}

# Token verification and user configuration flow
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  DeepScientist Token Authentication${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Loop to ask for token until verification succeeds or user chooses to skip
while true; do
    echo -e "${YELLOW}Enter your DeepScientist API token to authenticate (leave blank to skip):${NC}"
    read -rs -p "API Token: " DEEPSCIENTIST_TOKEN
    echo ""

    if [ -z "$DEEPSCIENTIST_TOKEN" ]; then
        echo -e "${YELLOW}ℹ${NC} No token provided. You can authenticate later using:"
        echo -e "   ${YELLOW}deepscientist-cli login --token YOUR_TOKEN${NC}"
        break
    fi

    # Verify token and attempt login
    if verify_and_login_token "$DEEPSCIENTIST_TOKEN"; then
        # Login successful, check if supported user and ask for second choice
        if [ "$SUPPORTED_USER" = true ]; then
            echo ""
            echo -e "${GREEN}🎯 You are a supported user! Please choose Claude Code configuration:${NC}"
            echo ""
            echo "1) Use DeepScientist provided resources (recommended)"
            echo "2) Use my own Anthropic API key"
            echo ""

            while true; do
                read -p "Please choose [1-2]: " USER_CHOICE
                case $USER_CHOICE in
                    1)
                        echo -e "${GREEN}✓${NC} Chosen to use DeepScientist resources"
                        USE_DEEPSCIENTIST_RESOURCE=true
                        break
                        ;;
                    2)
                        echo -e "${YELLOW}✓${NC} Chosen to use own Anthropic API"
                        USE_DEEPSCIENTIST_RESOURCE=false
                        break
                        ;;
                    *)
                        echo -e "${RED}Please enter 1 or 2${NC}"
                        ;;
                esac
            done
        else
            USE_DEEPSCIENTIST_RESOURCE=false
            echo -e "${YELLOW}ℹ${NC} Normal user detected. Claude Code will be installed without API configuration"
        fi
        break
    else
        # Token verification failed
        echo ""
        echo -e "${RED}❌ Token verification failed. Please check if your API Token is correct${NC}"
        echo -e "${YELLOW}💡 Tip: You can get the correct API Token from http://deepscientist.cc${NC}"
        echo ""

        while true; do
            read -p "Retry token input? [y/N]: " RETRY_CHOICE
            case $RETRY_CHOICE in
                [Yy]|[Yy][Ee][Ss])
                    echo ""
                    echo -e "${BLUE}Retrying token input...${NC}"
                    break
                    ;;
                [Nn]|[Nn][Oo]|"")
                    echo -e "${YELLOW}ℹ${NC} Skipping token verification. You can configure later using deepscientist-cli login command"
                    break 2  # Exit outer while loop
                    ;;
                *)
                    echo -e "${YELLOW}Please enter y or n${NC}"
                    ;;
            esac
        done
    fi
done

# Install and configure Claude Code using the dedicated script
echo ""
echo -e "${BLUE}Setting up Claude Code tooling...${NC}"

# Locate the claude_code_deepscientist_env.sh script
CLAUDE_SETUP_SCRIPT="$SCRIPT_DIR/src/claude_code_deepscientist_env.sh"

if [ ! -f "$CLAUDE_SETUP_SCRIPT" ]; then
    echo -e "${RED}✗ Claude Code setup script not found at: $CLAUDE_SETUP_SCRIPT${NC}"
    exit 1
fi

# Decide configuration mode based on user verification and choice
if [ "$VERIFICATION_SUCCESS" = true ] && [ "$USE_DEEPSCIENTIST_RESOURCE" = true ]; then
    CLAUDE_API_URL="${SERVER_URL%/}/anthropic"
    echo -e "${GREEN}✓${NC} Configuring Claude Code to use DeepScientist provided resources..."
    echo -e "${BLUE}API Endpoint: $CLAUDE_API_URL${NC}"
    bash "$CLAUDE_SETUP_SCRIPT" "$DEEPSCIENTIST_TOKEN" "$CLAUDE_API_URL"
elif [ "$VERIFICATION_SUCCESS" = true ]; then
    echo -e "${YELLOW}ℹ${NC} Installing Claude Code without configuring DeepScientist API..."
    echo -e "${YELLOW}ℹ${NC} You can manually configure your own API provider later"
    bash "$CLAUDE_SETUP_SCRIPT" "SKIP_CONFIGURE"
else
    echo -e "${YELLOW}ℹ${NC} Skipping token verification, installing Claude Code without API configuration..."
    echo -e "${YELLOW}ℹ${NC} After obtaining a token, please run 'deepscientist-cli login' and manually configure Claude Code"
    bash "$CLAUDE_SETUP_SCRIPT" "SKIP_CONFIGURE"
fi

# Reload nvm environment to access newly installed tools
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    # shellcheck disable=SC1090
    . "$NVM_DIR/nvm.sh"
fi

# Verify Claude Code installation
if command -v claude &> /dev/null; then
    echo -e "${GREEN}✓${NC} Claude Code setup completed successfully"
else
    echo -e "${RED}✗ Claude Code installation verification failed${NC}"
    exit 1
fi
# Save installation configuration
echo ""
echo -e "${BLUE}Saving installation configuration...${NC}"

# Save config in installation directory
CONFIG_FILE="$INSTALL_DIR/config.json"
cat > "$CONFIG_FILE" << EOF
{
  "install_dir": "$INSTALL_DIR",
  "bin_dir": "$BIN_DIR",
  "installed_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "version": "${CLI_VERSION}"
}
EOF
echo -e "${GREEN}✓${NC} Configuration saved to $CONFIG_FILE"

# Also save config in default location if using custom path
DEFAULT_DIR="$HOME/.deepscientist"
if [ "$INSTALL_DIR" != "$DEFAULT_DIR" ]; then
    mkdir -p "$DEFAULT_DIR"
    DEFAULT_CONFIG="$DEFAULT_DIR/config.json"
cat > "$DEFAULT_CONFIG" << EOF
{
  "install_dir": "$INSTALL_DIR",
  "bin_dir": "$BIN_DIR",
  "installed_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "version": "${CLI_VERSION}"
}
EOF
    echo -e "${GREEN}✓${NC} Configuration also saved to $DEFAULT_CONFIG (for discovery)"
fi

# Add to PATH if not already there
echo ""
echo -e "${BLUE}Configuring PATH...${NC}"

SHELL_RC=""
if [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
fi

PATH_ENTRIES=("$INSTALL_DIR" "$BIN_DIR" "$DEFAULT_ALIAS_DIR")

if [ -n "$SHELL_RC" ]; then
    ADDED=false
    for ENTRY in "${PATH_ENTRIES[@]}"; do
        PATH_LINE="export PATH=\"$ENTRY:\$PATH\""
        if ! grep -Fq "$PATH_LINE" "$SHELL_RC" 2>/dev/null; then
            if [ "$ADDED" = false ]; then
                {
                    echo ""
                    echo "# DeepScientist CLI"
                } >> "$SHELL_RC"
            fi
            echo "$PATH_LINE" >> "$SHELL_RC"
            ADDED=true
        fi
    done

    if [ "$ADDED" = true ]; then
        echo -e "${GREEN}✓${NC} Added PATH updates to $SHELL_RC"
    else
        echo -e "${YELLOW}ℹ${NC} PATH entries already present in $SHELL_RC"
    fi
fi


# Verify installation
echo ""
echo -e "${BLUE}Verifying installation...${NC}"

MISSING=false
for NAME in "${ALIASES[@]}"; do
    if command -v "$NAME" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $NAME command available"
    else
        MISSING=true
        echo -e "${YELLOW}⚠${NC} $NAME command not found in current session"
    fi
done

if [ "$MISSING" = true ] && [ -n "$SHELL_RC" ]; then
    echo -e "${YELLOW}  Please run: source $SHELL_RC${NC}"
fi

# Run alias validation
echo ""
echo -e "${BLUE}Running final CLI self-check...${NC}"
hash -r 2>/dev/null || true
if "$BIN_DIR/ds-cli" --help >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} ds-cli command executed successfully"
else
    echo -e "${RED}✗ ds-cli command failed to run. Check PATH or installation logs above.${NC}"
fi

# Print success message
echo ""
echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║          ✓ Installation completed successfully!          ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"


echo ""
echo "Installation Details:"
echo "  Installation directory: $INSTALL_DIR"
echo "  Commands installed:"
for NAME in "${ALIASES[@]}"; do
    echo "    - $BIN_DIR/$NAME"
done
echo "  Python version: $PYTHON_VERSION"
if [ "$IN_CONDA" = true ]; then
    echo "  Environment: conda ($CONDA_DEFAULT_ENV)"
else
    echo "  Environment: system Python (user install)"
fi

# Optional tools suggestion
if [ "$OS" == "linux" ] && ! command -v xclip &> /dev/null && ! command -v xsel &> /dev/null; then
    echo -e "${YELLOW}ℹ${NC} Tip: Install 'xclip' or 'xsel' to enable clipboard paste support (e.g. apt install xclip)"
fi
echo ""

if [ "$VERIFICATION_SUCCESS" = true ] && [ "$USE_DEEPSCIENTIST_RESOURCE" = true ]; then
    echo -e "${GREEN}✓${NC} Claude Code has been configured to use DeepScientist provided resources"
    echo -e "${GREEN}✓${NC} You can immediately start using DeepScientist for research tasks"
elif [ "$VERIFICATION_SUCCESS" = true ]; then
    echo -e "${YELLOW}ℹ${NC} Claude Code has been installed but API credentials are not configured"
    echo -e "${YELLOW}ℹ${NC} Please configure your own API provider before running implementation tasks"
else
    echo -e "${YELLOW}ℹ${NC} Please run when ready: ${BLUE}deepscientist-cli login --token YOUR_TOKEN${NC}"
fi
echo ""

echo -e "${GREEN}Next Steps:${NC}"
echo -e "  • ${GREEN}source ~/.bashrc${NC}    # apply PATH updates"
if [ "$IN_CONDA" = true ]; then
    echo -e "  • ${YELLOW}conda activate $CONDA_DEFAULT_ENV${NC}"
else
    echo -e "  • ${YELLOW}conda activate <your-env>${NC}"
fi
echo -e "  • ${YELLOW}ds-cli --help${NC}   # verify CLI is available"
echo ""

echo -e "${GREEN}Reminder:${NC} You can launch the CLI anytime with 'deepscientist-cli' or 'ds-cli'."
echo ""
echo -e "${GREEN}Happy researching! 🚀${NC}"
