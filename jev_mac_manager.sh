#!/usr/bin/env bash

# ==============================================================================
# Jev MCP - Flawless macOS Apple Silicon (M2 Pro) Environment Manager
# ==============================================================================

set -eou pipefail

# --- Color Definitions ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${BLUE}==>${NC} ${1}"; }
log_success() { echo -e "${GREEN}==> SUCCESS:${NC} ${1}"; }
log_warn() { echo -e "${YELLOW}==> WARNING:${NC} ${1}"; }
log_error() { echo -e "${RED}==> ERROR:${NC} ${1}"; }

COMMAND=${1:-help}
TARGET_MODEL_ALIAS=${2:-0.5b}

# Parse arguments for headless execution
HEADLESS=0
for arg in "$@"; do
    if [[ "$arg" == "--headless" || "$arg" == "-h" ]]; then
        HEADLESS=1
    fi
done

# Load environment variables
if [ -f "$PWD/.env" ]; then
    source "$PWD/.env"
fi
JEV_DAEMON_PORT=${JEV_DAEMON_PORT:-8080}

PLIST_NAME="com.jev.mlx_server"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
LOG_DIR="$HOME/.jev"
LOG_FILE="$LOG_DIR/mlx_server.log"

MODEL_05B="mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MODEL_7B="mlx-community/Qwen2.5-7B-Instruct-4bit"

resolve_model() {
    local alias=$1
    if [[ "$alias" == "0.5b" || "$alias" == "0.5B" ]]; then
        echo "$MODEL_05B"
    elif [[ "$alias" == "7b" || "$alias" == "7B" ]]; then
        echo "$MODEL_7B"
    else
        log_error "Unknown model profile: $alias. Must be '0.5b' or '7b'."
        exit 1
    fi
}

check_architecture() {
    log_info "Checking system architecture..."
    local ARCH
    ARCH=$(uname -m)
    if [ "$ARCH" != "arm64" ]; then
        log_error "This script is optimized for Apple Silicon (M1/M2/M3) 'arm64' architectures."
        exit 1
    fi
}

setup_launchd_plist() {
    local model_path="$1"
    local venv_python="$PWD/.venv/bin/python3"
    local tmp_plist="${PLIST_PATH}.tmp"
    
    log_info "Generating native macOS launchd agent for $model_path on port $JEV_DAEMON_PORT..."
    mkdir -p "$LOG_DIR"
    
    cat << PLIST_EOF > "$tmp_plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_NAME</string>
    <key>ProgramArguments</key>
    <array>
        <string>$venv_python</string>
        <string>-m</string>
        <string>mlx_lm.server</string>
        <string>--model</string>
        <string>$model_path</string>
        <string>--port</string>
        <string>$JEV_DAEMON_PORT</string>
        <!-- MLX Apple Silicon Performance Flags -->
        <string>--prompt-cache-size</string>
        <string>20</string>
        <string>--prompt-cache-bytes</string>
        <string>12G</string>
        <string>--prefill-step-size</string>
        <string>2048</string>
        <string>--log-level</string>
        <string>WARNING</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$LOG_FILE</string>
    <key>StandardErrorPath</key>
    <string>$LOG_FILE</string>
    <key>WorkingDirectory</key>
    <string>$PWD</string>
</dict>
</plist>
PLIST_EOF
    mv "$tmp_plist" "$PLIST_PATH"
    log_success "Generated $PLIST_PATH"
}

manage_daemon() {
    local action=$1
    if [[ "$action" == "stop" || "$action" == "restart" ]]; then
        log_info "Unloading launchd daemon..."
        launchctl bootout gui/$(id -u) "$PLIST_PATH" 2>/dev/null || launchctl unload "$PLIST_PATH" 2>/dev/null || true
        # Force kill to handle models in flight
        pkill -9 -f "mlx_lm.server" 2>/dev/null || true
        sleep 1
    fi
    if [[ "$action" == "start" || "$action" == "restart" ]]; then
        if [ -f "$PLIST_PATH" ]; then
            log_info "Loading launchd daemon..."
            if ! launchctl bootstrap gui/$(id -u) "$PLIST_PATH" 2>/dev/null && ! launchctl load "$PLIST_PATH"; then
                log_warn "Failed to bootstrap launchd daemon. Ensure your Terminal has 'Full Disk Access' in System Settings."
                log_warn "If this persists, try running the server manually with: .venv/bin/python3 -m mlx_lm.server --model $SELECTED_MODEL --port $JEV_DAEMON_PORT"
            else
                log_success "Daemon is running in the background!"
                log_info "Logs are streaming to: $LOG_FILE"
            fi
        fi
    fi
}

case "$COMMAND" in
    install)
        echo -e "${BLUE}====================================================${NC}"
        echo -e "${BLUE}  Jev MCP + MLX Installer for Apple Silicon       ${NC}"
        echo -e "${BLUE}====================================================${NC}"
        
        SELECTED_MODEL=$(resolve_model "$TARGET_MODEL_ALIAS")
        log_info "Target Model Configuration: $TARGET_MODEL_ALIAS ($SELECTED_MODEL)"

        check_architecture

        if ! command -v python3 &> /dev/null; then
            log_error "Python 3 not found. Please install Python 3.10+."
            exit 1
        fi

        log_info "Setting up isolated virtual environment in .venv..."
        python3 -m venv .venv
        source "$PWD/.venv/bin/activate"
        pip install --upgrade pip wheel -q &>/dev/null
        
        if [ -f "requirements.txt" ]; then
            log_info "Installing pinned dependencies from requirements.txt..."
            pip install -r requirements.txt
        else
            log_warn "No requirements.txt found! Falling back to unpinned mlx/mlx-lm..."
            pip install mlx mlx-lm
        fi
        
        if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
            pip install -e .
        fi

        manage_daemon stop
        setup_launchd_plist "$SELECTED_MODEL"
        manage_daemon start
        
        echo -e "${GREEN}Installation Complete! Your native macOS daemon is running the $TARGET_MODEL_ALIAS model.${NC}"
        ;;

    switch)
        echo -e "${YELLOW}====================================================${NC}"
        echo -e "${YELLOW}  Switching Local AI Model                          ${NC}"
        echo -e "${YELLOW}====================================================${NC}"
        
        SELECTED_MODEL=$(resolve_model "$TARGET_MODEL_ALIAS")
        log_info "Switching background engine to: $TARGET_MODEL_ALIAS ($SELECTED_MODEL)"
        
        manage_daemon stop
        setup_launchd_plist "$SELECTED_MODEL"
        manage_daemon start
        ;;

    update)
        echo -e "${BLUE}====================================================${NC}"
        echo -e "${BLUE}  Updating Jev MCP Environment                      ${NC}"
        echo -e "${BLUE}====================================================${NC}"
        
        if [ $HEADLESS -eq 0 ]; then
            read -p "Are you sure you want to update the repository and packages? [y/N] " confirm
            if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                log_info "Update aborted by user."
                exit 0
            fi
        fi
        
        if [ -d ".git" ]; then
            git pull origin main || log_warn "Git pull failed."
        fi

        if [ -d ".venv" ]; then
            source "$PWD/.venv/bin/activate"
            if [ -f "requirements.txt" ]; then
                log_info "Installing pinned dependencies from requirements.txt..."
                pip install -r requirements.txt
            else
                pip install --upgrade mlx mlx-lm
            fi
            
            if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
                pip install -e .
            fi
            
            SELECTED_MODEL=$(resolve_model "$TARGET_MODEL_ALIAS")
            manage_daemon stop
            setup_launchd_plist "$SELECTED_MODEL"
            manage_daemon start
            
            log_success "Environment updated and daemon restarted with $TARGET_MODEL_ALIAS."
        else
            log_error "No .venv found. Run 'install' first."
            exit 1
        fi
        ;;

    uninstall)
        echo -e "${YELLOW}====================================================${NC}"
        echo -e "${YELLOW}  Uninstalling Jev MCP Environment Artifacts        ${NC}"
        echo -e "${YELLOW}====================================================${NC}"
        
        if [ $HEADLESS -eq 0 ]; then
            read -p "Are you sure you want to completely remove the Jev daemon and environment? [y/N] " confirm
            if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                log_info "Uninstallation aborted by user."
                exit 0
            fi
        fi
        
        manage_daemon stop
        
        log_info "Removing launchd plist..."
        rm -f "$PLIST_PATH"
        
        log_info "Removing virtual environment..."
        rm -rf .venv
        
        log_info "Cleaning up logs..."
        rm -rf "$LOG_DIR" || true
        
        echo -e "${GREEN}Uninstallation Complete!${NC}"
        ;;

    *)
        echo -e "${RED}Invalid command.${NC}"
        echo "Usage:"
        echo "  ./jev_mac_manager.sh install [0.5b|7b]"
        echo "  ./jev_mac_manager.sh switch [0.5b|7b]"
        echo "  ./jev_mac_manager.sh update [0.5b|7b]"
        echo "  ./jev_mac_manager.sh uninstall"
        exit 1
        ;;
esac
