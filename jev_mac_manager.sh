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
JEV_FAST_PORT=${JEV_FAST_PORT:-8080}
JEV_SMART_PORT=${JEV_SMART_PORT:-8081}

PLIST_NAME="com.jev.mlx_server"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}.plist"
PLIST_FAST_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}_fast.plist"
PLIST_SMART_PATH="$HOME/Library/LaunchAgents/${PLIST_NAME}_smart.plist"
LOG_DIR="$HOME/.jev"
LOG_FILE="$LOG_DIR/mlx_server.log"
LOG_FILE_FAST="$LOG_DIR/mlx_server_fast.log"
LOG_FILE_SMART="$LOG_DIR/mlx_server_smart.log"

MODEL_05B="jaredpalmer/kev-0.8b"
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


link_mcp_configs() {
    log_info "Linking MCP configurations to local absolute path..."
    $PWD/.venv/bin/python3 -c "
import json
import os

def link_mcp(path, root_key):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception:
        data = {}
        
    if root_key not in data:
        data[root_key] = {}
        
    abs_python = os.path.abspath('.venv/bin/python3')
    abs_dir = os.path.abspath('.')
    
    data[root_key]['jev-mcp'] = {
        'command': abs_python,
        'args': ['-m', 'jev_mcp.server'],
        'env': {'PYTHONPATH': abs_dir + '/src'}
    }
    
    if 'opencode' in path.lower():
        data[root_key]['jev-mcp']['type'] = 'local'
        data[root_key]['jev-mcp']['enabled'] = True
        
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Successfully patched {path}')

link_mcp('~/.gemini/config/mcp_config.json', 'mcpServers')
link_mcp('~/.config/opencode/opencode.json', 'mcp')
link_mcp('~/Library/Application Support/Claude/claude_desktop_config.json', 'mcpServers')
link_mcp('~/.claude.json', 'mcpServers')
"
}

setup_launchd_plist() {
    local model_path="$1"
    local port="$2"
    local target_plist="$3"
    local target_log="$4"
    local label=$(basename "$target_plist" .plist)
    local venv_python="$PWD/.venv/bin/python3"
    local tmp_plist="${target_plist}.tmp"
    
    local engine="mlx_lm.server"
    local extra_args="<string>--prompt-cache-size</string><string>20</string><string>--prompt-cache-bytes</string><string>12G</string><string>--prefill-step-size</string><string>2048</string><string>--log-level</string><string>WARNING</string>"
    
    if [[ "$model_path" == *"kev"* ]]; then
        engine="kev.serve"
        # kev.serve takes --run instead of --model, and handles its own cache logic
        extra_args=""
    fi
    
    log_info "Generating native macOS launchd agent for $model_path on port $port using $engine..."
    mkdir -p "$LOG_DIR"
    
    if [[ "$engine" == "kev.serve" ]]; then
        cat << PLIST_EOF > "$tmp_plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>$venv_python</string>
        <string>-m</string>
        <string>kev.serve</string>
        <string>--run</string>
        <string>$model_path</string>
        <string>--port</string>
        <string>$port</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$target_log</string>
    <key>StandardErrorPath</key>
    <string>$target_log</string>
    <key>WorkingDirectory</key>
    <string>$PWD</string>
</dict>
</plist>
PLIST_EOF
    else
        cat << PLIST_EOF > "$tmp_plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>$venv_python</string>
        <string>-m</string>
        <string>mlx_lm.server</string>
        <string>--model</string>
        <string>$model_path</string>
        <string>--port</string>
        <string>$port</string>
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
    <string>$target_log</string>
    <key>StandardErrorPath</key>
    <string>$target_log</string>
    <key>WorkingDirectory</key>
    <string>$PWD</string>
</dict>
</plist>
PLIST_EOF
    fi

    mv "$tmp_plist" "$target_plist"
    log_success "Generated $target_plist"
}

manage_daemon() {
    local action=$1
    local target_plist=${2:-$PLIST_PATH}
    if [[ "$action" == "stop" || "$action" == "restart" ]]; then
        log_info "Unloading launchd daemon..."
        launchctl bootout gui/$(id -u) "$target_plist" 2>/dev/null || launchctl unload "$target_plist" 2>/dev/null || true
        # Force kill to handle models in flight
        pkill -9 -f "mlx_lm.server" 2>/dev/null || true
        sleep 1
    fi
    if [[ "$action" == "start" || "$action" == "restart" ]]; then
        if [ -f "$target_plist" ]; then
            log_info "Loading launchd daemon..."
            if ! launchctl bootstrap gui/$(id -u) "$target_plist" 2>/dev/null && ! launchctl load "$target_plist"; then
                log_warn "Failed to bootstrap launchd daemon. Ensure your Terminal has 'Full Disk Access' in System Settings."
            else
                log_success "Daemon is running in the background!"
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
        setup_launchd_plist "$SELECTED_MODEL" "$JEV_FAST_PORT" "$PLIST_PATH" "$LOG_FILE"
        manage_daemon start "$PLIST_PATH"
        
        # Inject Opencode Configuration
        log_info "Configuring Opencode integration..."
        python3 -c '
import json, os
config_path = os.path.expanduser("~/.config/opencode/opencode.json")
if os.path.exists(os.path.dirname(config_path)):
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f: config = json.load(f)
        else:
            config = {"$schema": "https://opencode.ai/config.json", "mcp": {}}
        if "mcp" not in config: config["mcp"] = {}
        if "jev-mcp" in config["mcp"]: del config["mcp"]["jev-mcp"]
        config["mcp"]["jev-mcp"] = {
            "type": "local",
            "command": [os.path.join(os.getcwd(), ".venv/bin/python3"), "-m", "jev_mcp.server"],
            "environment": {"PYTHONPATH": os.path.join(os.getcwd(), "src")},
            "enabled": True
        }
        with open(config_path, "w") as f: json.dump(config, f, indent=2)
        print("Successfully integrated with Opencode!")
    except Exception as e:
        pass
'
        
        echo -e "${GREEN}Installation Complete! Your native macOS daemon is running the $TARGET_MODEL_ALIAS model.${NC}"
        ;;



    hybrid)
        echo -e "${BLUE}====================================================${NC}"
        echo -e "${BLUE}  Starting Hybrid Complexity Router Mode            ${NC}"
        echo -e "${BLUE}====================================================${NC}"
        
        log_info "Configuring dual-daemon setup (0.5B on $JEV_FAST_PORT, 7B on $JEV_SMART_PORT)..."
        manage_daemon stop "$PLIST_PATH"
        manage_daemon stop "$PLIST_FAST_PATH"
        manage_daemon stop "$PLIST_SMART_PATH"
        setup_launchd_plist "$MODEL_05B" "$JEV_FAST_PORT" "$PLIST_FAST_PATH" "$LOG_FILE_FAST"
        setup_launchd_plist "$MODEL_7B" "$JEV_SMART_PORT" "$PLIST_SMART_PATH" "$LOG_FILE_SMART"
        
        manage_daemon start "$PLIST_FAST_PATH"
        manage_daemon start "$PLIST_SMART_PATH"
        
        echo -e "${GREEN}Hybrid Mode Active! Jev MCP will now intelligently route requests.${NC}"
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
            setup_launchd_plist "$SELECTED_MODEL" "$JEV_FAST_PORT" "$PLIST_PATH" "$LOG_FILE"
            manage_daemon start "$PLIST_PATH"
            
            # Sync Opencode Configuration
            python3 -c '
import json, os
config_path = os.path.expanduser("~/.config/opencode/opencode.json")
if os.path.exists(os.path.dirname(config_path)):
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f: config = json.load(f)
        else:
            config = {"$schema": "https://opencode.ai/config.json", "mcp": {}}
        if "mcp" not in config: config["mcp"] = {}
        if "jev-mcp" in config["mcp"]: del config["mcp"]["jev-mcp"]
        config["mcp"]["jev-mcp"] = {
            "type": "local",
            "command": [os.path.join(os.getcwd(), ".venv/bin/python3"), "-m", "jev_mcp.server"],
            "environment": {"PYTHONPATH": os.path.join(os.getcwd(), "src")},
            "enabled": True
        }
        with open(config_path, "w") as f: json.dump(config, f, indent=2)
    except Exception as e:
        pass
'
            
            log_success "Environment updated and daemon restarted with $TARGET_MODEL_ALIAS."
        else
            log_error "No .venv found. Run 'install' first."
            exit 1
        fi
        ;;

    start)
        if [ ! -d ".venv" ]; then
            log_error "The Python virtual environment is missing."
            log_error "Please install the system first by running: make install MODEL=$TARGET_MODEL_ALIAS"
            exit 1
        fi
        SELECTED_MODEL=$(resolve_model "$TARGET_MODEL_ALIAS")
        log_info "Configuring daemon for model: $TARGET_MODEL_ALIAS ($SELECTED_MODEL)"
        manage_daemon stop
        setup_launchd_plist "$SELECTED_MODEL"
        echo -e "${GREEN}Starting Jev MCP Daemon...${NC}"
        manage_daemon start
        ;;

    stop)
        echo -e "${YELLOW}Stopping Jev MCP Daemon...${NC}"
        manage_daemon stop "$PLIST_PATH"
        manage_daemon stop "$PLIST_FAST_PATH"
        manage_daemon stop "$PLIST_SMART_PATH"
        echo -e "${GREEN}Daemon stopped.${NC}"
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
        rm -f "$PLIST_FAST_PATH"
        rm -f "$PLIST_SMART_PATH"
        

        log_info "Removing virtual environment..."
        rm -rf .venv
        
        # Remove Opencode Configuration
        log_info "Removing Opencode integration..."
        python3 -c '
import json, os
config_path = os.path.expanduser("~/.config/opencode/opencode.json")
try:
    if os.path.exists(config_path):
        with open(config_path, "r") as f: config = json.load(f)
        if "mcp" in config and "jev" in config["mcp"]:
            del config["mcp"]["jev-mcp"]
            with open(config_path, "w") as f: json.dump(config, f, indent=2)
except Exception: pass
' 2>/dev/null || true

        log_info "Cleaning up logs..."
        rm -rf "$LOG_DIR" || true
        
        echo -e "${GREEN}Uninstallation Complete!${NC}"
        ;;

    *)
        echo -e "${RED}Invalid command.${NC}"
        echo "Usage:"
        echo "  ./jev_mac_manager.sh install "
        echo "  ./jev_mac_manager.sh hybrid"
        echo "  ./jev_mac_manager.sh update "
        echo "  ./jev_mac_manager.sh start "
        echo "  ./jev_mac_manager.sh stop"
        echo "  ./jev_mac_manager.sh uninstall"
        exit 1
        ;;
esac
