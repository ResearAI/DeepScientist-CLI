#!/bin/bash

set -euo pipefail

# ========================
#       常量定义
# ========================
SCRIPT_NAME=$(basename "$0")
NODE_MIN_VERSION=18
NODE_INSTALL_VERSION=22
NVM_VERSION="v0.40.3"
CLAUDE_PACKAGE="@anthropic-ai/claude-code"
CONFIG_DIR="$HOME/.claude"
CONFIG_FILE="$CONFIG_DIR/settings.json"
API_BASE_URL="http://deepscientist.ai-researcher.net:8888/anthropic"
API_KEY_URL="http://deepscientist.cc/"
API_TIMEOUT_MS=3000000

# Accept API token and optional API URL as parameters
API_TOKEN="${1:-}"
API_URL_PARAM="${2:-}"

# ========================
#       工具函数
# ========================

log_info() {
    echo "🔹 $*"
}

log_success() {
    echo "✅ $*"
}

log_error() {
    echo "❌ $*" >&2
}

ensure_dir_exists() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir" || {
            log_error "Failed to create directory: $dir"
            exit 1
        }
    fi
}

# ========================
#     Node.js 安装函数
# ========================

install_nodejs() {
    local platform=$(uname -s)

    case "$platform" in
        Linux|Darwin)
            log_info "Installing Node.js on $platform..."

            # 安装 nvm
            log_info "Installing nvm ($NVM_VERSION)..."
            curl -s https://raw.githubusercontent.com/nvm-sh/nvm/"$NVM_VERSION"/install.sh | bash

            # 加载 nvm
            log_info "Loading nvm environment..."
            \. "$HOME/.nvm/nvm.sh"

            # 安装 Node.js
            log_info "Installing Node.js $NODE_INSTALL_VERSION..."
            nvm install "$NODE_INSTALL_VERSION"

            # 验证安装
            node -v &>/dev/null || {
                log_error "Node.js installation failed"
                exit 1
            }
            log_success "Node.js installed: $(node -v)"
            log_success "npm version: $(npm -v)"
            ;;
        *)
            log_error "Unsupported platform: $platform"
            exit 1
            ;;
    esac
}

# ========================
#     Node.js 检查函数
# ========================

check_nodejs() {
    if command -v node &>/dev/null; then
        current_version=$(node -v | sed 's/v//')
        major_version=$(echo "$current_version" | cut -d. -f1)

        if [ "$major_version" -ge "$NODE_MIN_VERSION" ]; then
            log_success "Node.js is already installed: v$current_version"
            return 0
        else
            log_info "Node.js v$current_version is installed but version < $NODE_MIN_VERSION. Upgrading..."
            install_nodejs
        fi
    else
        log_info "Node.js not found. Installing..."
        install_nodejs
    fi
}

# ========================
#     Claude Code 安装
# ========================

install_claude_code() {
    if command -v claude &>/dev/null; then
        log_success "Claude Code is already installed: $(claude --version)"
    else
        log_info "Installing Claude Code..."
        npm install -g "$CLAUDE_PACKAGE" || {
            log_error "Failed to install claude-code"
            exit 1
        }
        log_success "Claude Code installed successfully"
    fi
}

configure_claude_json(){
  node --eval '
      const os = require("os");
      const fs = require("fs");
      const path = require("path");

      const homeDir = os.homedir();
      const filePath = path.join(homeDir, ".claude.json");
      if (fs.existsSync(filePath)) {
          const content = JSON.parse(fs.readFileSync(filePath, "utf-8"));
          fs.writeFileSync(filePath, JSON.stringify({ ...content, hasCompletedOnboarding: true }, null, 2), "utf-8");
      } else {
          fs.writeFileSync(filePath, JSON.stringify({ hasCompletedOnboarding: true }, null, 2), "utf-8");
      }'
}

# ========================
#     API Key 配置
# ========================

configure_claude() {
    log_info "Configuring Claude Code..."

    # Use the token passed as parameter
    local api_key="$API_TOKEN"

    # Use custom API URL if provided, otherwise use default
    local api_url="${API_URL_PARAM:-$API_BASE_URL}"

    if [ -z "$api_key" ]; then
        log_error "API token not provided."
        echo "   You can get your API token from: $API_KEY_URL"
        exit 1
    fi

    ensure_dir_exists "$CONFIG_DIR"

    log_info "Using API URL: $api_url"

    # 写入配置文件
    node --eval '
        const os = require("os");
        const fs = require("fs");
        const path = require("path");

        const homeDir = os.homedir();
        const filePath = path.join(homeDir, ".claude", "settings.json");
        const apiKey = "'"$api_key"'";
        const apiUrl = "'"$api_url"'";

        const content = fs.existsSync(filePath)
            ? JSON.parse(fs.readFileSync(filePath, "utf-8"))
            : {};

        fs.writeFileSync(filePath, JSON.stringify({
            ...content,
            env: {
                ANTHROPIC_AUTH_TOKEN: apiKey,
                ANTHROPIC_BASE_URL: apiUrl,
                API_TIMEOUT_MS: "'"$API_TIMEOUT_MS"'",
            }
        }, null, 2), "utf-8");
    ' || {
        log_error "Failed to write settings.json"
        exit 1
    }

    log_success "Claude Code configured successfully with provided token"
}

# ========================
#        主流程
# ========================

main() {
    echo "🚀 Starting $SCRIPT_NAME"
    echo ""

    # Check if we should skip configuration
    if [ "$API_TOKEN" == "SKIP_CONFIGURE" ]; then
        log_info "Skipping Claude Code configuration (using own compute resources)"
        echo ""

        check_nodejs
        install_claude_code
        configure_claude_json

        echo ""
        log_success "🎉 Claude Code installation completed successfully!"
        echo ""
        echo "🚀 Claude Code is now installed"
        echo "   You can configure it manually with: claude"
        echo "   Or use your own API configuration"
        return 0
    fi

    # Check if token was provided
    if [ -z "$API_TOKEN" ]; then
        log_error "Usage: $SCRIPT_NAME <API_TOKEN|SKIP_CONFIGURE>"
        echo "   Get your API token from: $API_KEY_URL"
        echo "   Or use 'SKIP_CONFIGURE' to skip DeepScientist configuration"
        exit 1
    fi

    # Normal installation with configuration
    check_nodejs
    install_claude_code
    configure_claude_json
    configure_claude

    echo ""
    log_success "🎉 Claude Code installation completed successfully!"
    echo ""
    echo "🚀 Claude Code is now configured with DeepScientist compute resources"
    echo "   You can start it with: claude"
}

main
