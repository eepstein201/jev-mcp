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

# Colors are escape sequences (%b); the message is data and printed as-is (%s), so a backslash in a
# resolved value is never interpreted.
log_info() { printf '%b==>%b %s\n' "$BLUE" "$NC" "$1"; }
log_success() { printf '%b==> SUCCESS:%b %s\n' "$GREEN" "$NC" "$1"; }
log_warn() { printf '%b==> WARNING:%b %s\n' "$YELLOW" "$NC" "$1"; }
log_error() { printf '%b==> ERROR:%b %s\n' "$RED" "$NC" "$1"; }

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

MODEL_05B="$PWD/models/Kev-0.8B-GGUF/Kev-0.8B-Q8_0.gguf"
MODEL_7B="$PWD/models/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

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


check_llama_cpp() {
    if ! command -v llama-server &> /dev/null; then
        log_info "llama-server not found. Installing llama.cpp via Homebrew..."
        if ! command -v brew &> /dev/null; then
            log_error "Homebrew is required to install llama.cpp. Please install Homebrew."
            exit 1
        fi
        brew install llama.cpp
    fi
}

download_gguf() {
    local url="$1"
    local target_dir="$2"
    local target_file="$target_dir/$(basename "$url")"
    
    if [ ! -f "$target_file" ]; then
        log_info "Downloading $(basename "$url")..."
        mkdir -p "$target_dir"
        curl -L -o "$target_file" "$url"
        log_success "Downloaded $(basename "$url")."
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
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Successfully patched {path}')

link_mcp('~/.gemini/config/mcp_config.json', 'mcpServers')
link_mcp('~/Library/Application Support/Claude/claude_desktop_config.json', 'mcpServers')
link_mcp('~/.claude.json', 'mcpServers')
"
    # OpenCode has its own shape (mcp.servers); a config that cannot be read is left untouched.
    "$PWD/.venv/bin/python" -m jev_mcp.client_config link-opencode --python "$PWD/.venv/bin/python" \
        || log_warn "OpenCode config was not updated."
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
    
    if [[ "$model_path" == *".gguf"* ]]; then
        engine="llama-server"
        extra_args=""
    elif [[ "$model_path" == *"kev"* ]]; then
        engine="kev.serve"
        # kev.serve takes --run instead of --model, and handles its own cache logic
        extra_args=""
    fi
    
    log_info "Generating native macOS launchd agent for $model_path on port $port using $engine..."
    mkdir -p "$LOG_DIR"
    
    if [[ "$engine" == "llama-server" ]]; then
        local llama_bin
        llama_bin=$(command -v llama-server)
        
        local dynamic_batch_size="${JEV_BATCH_SIZE:-}"
        if [ -z "$dynamic_batch_size" ]; then
            local mem_bytes
            mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 17179869184)
            local mem_gb=$(( mem_bytes / 1024 / 1024 / 1024 ))
            
            # Base parallel slots on RAM (more conservatively for Mac Unified Memory)
            if [ "$mem_gb" -ge 64 ]; then
                dynamic_batch_size=16
            elif [ "$mem_gb" -ge 32 ]; then
                dynamic_batch_size=8
            elif [ "$mem_gb" -ge 16 ]; then
                dynamic_batch_size=4
            else
                dynamic_batch_size=2
            fi
            
            # If running a larger model (e.g. 7B), halve the parallel slots
            if [[ "$model_path" == *"7b"* || "$model_path" == *"7B"* ]]; then
                dynamic_batch_size=$(( dynamic_batch_size / 2 ))
                if [ "$dynamic_batch_size" -lt 1 ]; then
                    dynamic_batch_size=1
                fi
            fi
        fi
        
        local total_ctx=$(( dynamic_batch_size * 16384 ))
        
        cat << PLIST_EOF > "$tmp_plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$label</string>
    <key>ProgramArguments</key>
    <array>
        <string>$llama_bin</string>
        <string>-m</string>
        <string>$model_path</string>
        <string>--port</string>
        <string>$port</string>
        <string>--parallel</string>
        <string>$dynamic_batch_size</string>
        <string>-c</string>
        <string>$total_ctx</string>
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
    elif [[ "$engine" == "kev.serve" ]]; then
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

# Prints the --port value from a launchd plist (nothing if the file or flag is absent).
plist_port() {
    local plist=$1
    [ -f "$plist" ] || return 0
    awk '/<string>--port<\/string>/ { getline; gsub(/.*<string>|<\/string>.*/, ""); print; exit }' "$plist"
}

manage_daemon() {
    local action=$1
    local target_plist=${2:-$PLIST_PATH}
    if [[ "$action" == "stop" || "$action" == "restart" ]]; then
        log_info "Unloading launchd daemon..."
        # Read the port before unloading; only that agent's engine is force-killed.
        local port
        port=$(plist_port "$target_plist")
        launchctl bootout gui/$(id -u) "$target_plist" 2>/dev/null || launchctl unload "$target_plist" 2>/dev/null || true
        if [ -n "$port" ]; then
            # Force kill to handle models in flight
            pkill -9 -f "(llama-server|mlx_lm.server|kev.serve).*--port $port" 2>/dev/null || true
        fi
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

# --- Engine profiles (hybrid): every model fact comes from `engine_profiles resolve` ---

# Runs `python -m jev_mcp.engine_profiles <subcommand> ...` for this checkout.
profiles_cli() {
    local sub=$1
    shift
    local flags=(--config "$PWD/engine_profiles.toml" --models-dir "$PWD/models")
    if [ "${JEV_ALLOW_EVAL_ONLY:-}" = "1" ]; then
        flags+=(--allow-eval-only)
    fi
    # activate polls the daemons' loopback /props: never send that through a proxy.
    NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}" \
        PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
        "${JEV_PROFILES_PYTHON:-$PWD/.venv/bin/python}" -m jev_mcp.engine_profiles \
        "$sub" "$@" "${flags[@]}"
}

# Exits 2 unless $1 is a valid profile name (the same pattern the resolver enforces).
require_profile_name() {
    local LC_ALL=C
    if ! [[ "$1" =~ ^[a-z0-9][a-z0-9._-]{0,31}$ ]]; then
        log_error "Invalid profile name. Use lowercase letters, digits, '.', '_' or '-' (1-32 characters, starting with a letter or digit)."
        exit 2
    fi
}

# Reads `resolve` output into RESOLVED_<KEY> variables without eval or source.
# An unknown key, a line without a value, or a missing key is an error.
parse_resolved() {
    local k v seen=" "
    RESOLVED_PROFILE="" RESOLVED_FAST_MODEL="" RESOLVED_FAST_PATH="" RESOLVED_FAST_URL="" RESOLVED_FAST_SHA256=""
    RESOLVED_SMART_MODEL="" RESOLVED_SMART_PATH="" RESOLVED_SMART_URL="" RESOLVED_SMART_SHA256=""
    while IFS='=' read -r k v; do
        case "$k" in
            profile) RESOLVED_PROFILE=$v ;;
            fast_model) RESOLVED_FAST_MODEL=$v ;;
            fast_path) RESOLVED_FAST_PATH=$v ;;
            fast_url) RESOLVED_FAST_URL=$v ;;
            fast_sha256) RESOLVED_FAST_SHA256=$v ;;
            smart_model) RESOLVED_SMART_MODEL=$v ;;
            smart_path) RESOLVED_SMART_PATH=$v ;;
            smart_url) RESOLVED_SMART_URL=$v ;;
            smart_sha256) RESOLVED_SMART_SHA256=$v ;;
            *)
                log_error "Unexpected line in the profile resolver output."
                return 1
                ;;
        esac
        case "$seen" in
            *" $k "*)
                log_error "The profile resolver gave '$k' more than once."
                return 1
                ;;
        esac
        seen="$seen$k "
        if [ -z "$v" ]; then
            log_error "The profile resolver gave no value for '$k'."
            return 1
        fi
    done <<< "$1"
    if [ -z "$RESOLVED_PROFILE" ] || [ -z "$RESOLVED_FAST_MODEL" ] || [ -z "$RESOLVED_FAST_PATH" ] \
        || [ -z "$RESOLVED_FAST_URL" ] || [ -z "$RESOLVED_FAST_SHA256" ] || [ -z "$RESOLVED_SMART_MODEL" ] \
        || [ -z "$RESOLVED_SMART_PATH" ] || [ -z "$RESOLVED_SMART_URL" ] || [ -z "$RESOLVED_SMART_SHA256" ]; then
        log_error "The profile resolver output is incomplete."
        return 1
    fi
}

# curl hardening for multi-GB downloads: no URL globbing, bounded connect, abort a transfer that
# stays under 10 KB/s for a minute.
DOWNLOAD_CONNECT_TIMEOUT=30
DOWNLOAD_MIN_SPEED=10240
DOWNLOAD_STALL_SECONDS=60

# Downloads <url> to <path>.part, checks its SHA-256, then moves it to <path>.
# Any failure, or an INT/TERM while it runs, removes the .part file; <path> is never left half-written.
download_verified() {
    local rc=0
    DOWNLOAD_PART="$2.part"
    DOWNLOAD_PID=""
    trap 'abort_download 130' INT
    trap 'abort_download 143' TERM
    fetch_and_verify "$@" || rc=$?
    trap - INT TERM
    return "$rc"
}

# Trap body: kill the running curl, wait for it to be gone, delete its .part file, then exit.
abort_download() {
    if [ -n "$DOWNLOAD_PID" ]; then
        kill "$DOWNLOAD_PID" 2>/dev/null || true
        wait "$DOWNLOAD_PID" 2>/dev/null || true
    fi
    rm -f "$DOWNLOAD_PART"
    exit "$1"
}

fetch_and_verify() {
    local url=$1 path=$2 expected=$3
    local part="$path.part" actual
    if [[ "$url" != https://* ]]; then
        log_error "Refusing to download from a non-https URL."
        return 1
    fi
    log_info "Downloading $(basename "$path")..."
    mkdir -p "$(dirname "$path")" || return 1
    # curl runs in the background and the shell waits on it: bash runs a trap only after a foreground
    # command ends, but `wait` is interrupted at once, so a TERM/INT can kill curl straight away.
    curl --fail --location --proto '=https' --proto-redir '=https' \
        --globoff --connect-timeout "$DOWNLOAD_CONNECT_TIMEOUT" \
        --speed-limit "$DOWNLOAD_MIN_SPEED" --speed-time "$DOWNLOAD_STALL_SECONDS" \
        -o "$part" "$url" &
    DOWNLOAD_PID=$!
    local curl_rc=0
    wait "$DOWNLOAD_PID" || curl_rc=$?
    DOWNLOAD_PID=""
    if [ "$curl_rc" -ne 0 ]; then
        rm -f "$part"
        log_error "Download failed: $(basename "$path")."
        return 1
    fi
    if ! actual=$(shasum -a 256 "$part" | awk '{print $1}') || [ "$actual" != "$expected" ]; then
        rm -f "$part"
        log_error "SHA-256 mismatch for $(basename "$path"); the download was discarded."
        return 1
    fi
    if ! mv "$part" "$path"; then
        rm -f "$part"
        return 1
    fi
    log_success "Downloaded $(basename "$path")."
}

# Writes one tier's plist through the profiles CLI at the existing label, log and port.
# The file is written to <target_plist>.new; the label comes from the live name, not the .new one.
write_tier_plist() {
    local tier=$1 port=$2 target_plist=$3 target_log=$4 llama_bin
    llama_bin=$(command -v llama-server)
    profiles_cli plist "$RESOLVED_PROFILE" --tier "$tier" --port "$port" \
        --label "$(basename "$target_plist" .plist)" --log "$target_log" \
        --llama-server "$llama_bin" --workdir "$PWD" --out "$target_plist.new"
}

# Usage: start_profile [name]
start_profile() {
    local out rc=0
    if [ $# -gt 0 ]; then
        require_profile_name "$1"
    fi
    local python="${JEV_PROFILES_PYTHON:-$PWD/.venv/bin/python}"
    if [ ! -x "$python" ]; then
        if [ -n "${JEV_PROFILES_PYTHON:-}" ]; then
            log_error "JEV_PROFILES_PYTHON points to $python, which is not an executable file."
        else
            log_error "Python for the profile resolver not found at $python. Run: make install"
        fi
        exit 1
    fi

    out=$(profiles_cli resolve ${1+"$1"}) || rc=$?
    if [ "$rc" -ne 0 ]; then
        exit "$rc"
    fi
    parse_resolved "$out" || exit 1
    log_info "Profile: $RESOLVED_PROFILE"

    log_info "Checking models for profile $RESOLVED_PROFILE..."
    if [ ! -f "$RESOLVED_FAST_PATH" ]; then
        download_verified "$RESOLVED_FAST_URL" "$RESOLVED_FAST_PATH" "$RESOLVED_FAST_SHA256" || exit 1
    fi
    if [ ! -f "$RESOLVED_SMART_PATH" ]; then
        download_verified "$RESOLVED_SMART_URL" "$RESOLVED_SMART_PATH" "$RESOLVED_SMART_SHA256" || exit 1
    fi
    check_llama_cpp

    log_info "Configuring dual-daemon setup ($RESOLVED_FAST_MODEL on $JEV_FAST_PORT, $RESOLVED_SMART_MODEL on $JEV_SMART_PORT)..."
    # Both new plists go to <live>.new before anything is stopped, so a failed `plist` leaves the live
    # plists (which `stop` reads for the old ports, and launchd reloads at login) and the running
    # agents untouched. They replace the live plists only after the old agents are stopped.
    mkdir -p "$LOG_DIR" "$(dirname "$PLIST_FAST_PATH")"
    rc=0
    write_tier_plist fast "$JEV_FAST_PORT" "$PLIST_FAST_PATH" "$LOG_FILE_FAST" || rc=$?
    if [ "$rc" -eq 0 ]; then
        write_tier_plist smart "$JEV_SMART_PORT" "$PLIST_SMART_PATH" "$LOG_FILE_SMART" || rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        rm -f "$PLIST_FAST_PATH.new" "$PLIST_SMART_PATH.new"
        log_error "Could not generate the launchd plists; nothing was stopped."
        exit "$rc"
    fi

    manage_daemon stop "$PLIST_PATH"
    # Hybrid mode never uses the single-daemon agent; leaving its plist behind
    # makes launchd reload it at login, where it fights the fast engine for its port.
    rm -f "$PLIST_PATH"
    manage_daemon stop "$PLIST_FAST_PATH"
    manage_daemon stop "$PLIST_SMART_PATH"
    mv "$PLIST_FAST_PATH.new" "$PLIST_FAST_PATH"
    mv "$PLIST_SMART_PATH.new" "$PLIST_SMART_PATH"

    manage_daemon start "$PLIST_FAST_PATH"
    manage_daemon start "$PLIST_SMART_PATH"

    profiles_cli activate "$RESOLVED_PROFILE" --fast-port "$JEV_FAST_PORT" --smart-port "$JEV_SMART_PORT" \
        || { rc=$?; log_error "Profile $RESOLVED_PROFILE did not become active."; exit "$rc"; }
}

case "$COMMAND" in
    install)
        echo -e "${BLUE}====================================================${NC}"
        echo -e "${BLUE}  Jev MCP + MLX Installer for Apple Silicon       ${NC}"
        echo -e "${BLUE}====================================================${NC}"
        
        SELECTED_MODEL=$(resolve_model "$TARGET_MODEL_ALIAS")
        log_info "Target Model Configuration: $TARGET_MODEL_ALIAS ($SELECTED_MODEL)"

        log_info "Checking models..."
        if [[ "$SELECTED_MODEL" == *".gguf"* ]]; then
            if [[ "$TARGET_MODEL_ALIAS" == "0.5b" || "$TARGET_MODEL_ALIAS" == "0.5B" ]]; then
                download_gguf "https://huggingface.co/ggml-org/Kev-0.8B-GGUF/resolve/main/Kev-0.8B-Q8_0.gguf" "$(dirname "$SELECTED_MODEL")"
            elif [[ "$TARGET_MODEL_ALIAS" == "7b" || "$TARGET_MODEL_ALIAS" == "7B" ]]; then
                download_gguf "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf" "$(dirname "$SELECTED_MODEL")"
            fi
            check_llama_cpp
        fi


        check_architecture

        if ! command -v python3 &> /dev/null; then
            log_error "Python 3 not found. Please install Python 3.10+."
            exit 1
        fi

        if [ -d ".git" ]; then
            log_info "Initializing git submodules..."
            git submodule update --init --recursive || log_warn "Git submodule update failed."
        fi

        log_info "Setting up isolated virtual environment in .venv..."
        python3.12 -m venv .venv
        source "$PWD/.venv/bin/activate"
        pip install --upgrade pip wheel -q &>/dev/null
        
        log_info "Installing jev-mcp and the pinned engine packages from pyproject.toml..."
        pip install -e ".[engines]"


        manage_daemon stop
        setup_launchd_plist "$SELECTED_MODEL" "$JEV_FAST_PORT" "$PLIST_PATH" "$LOG_FILE"
        manage_daemon start "$PLIST_PATH"
        
        link_mcp_configs
        
        echo -e "${GREEN}Installation Complete! Your native macOS daemon is running the $TARGET_MODEL_ALIAS model.${NC}"
        ;;



    hybrid)
        echo -e "${BLUE}====================================================${NC}"
        echo -e "${BLUE}  Starting Hybrid Complexity Router Mode            ${NC}"
        echo -e "${BLUE}====================================================${NC}"

        start_profile ${2+"$2"}

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
            git submodule update --init --recursive || log_warn "Git submodule update failed."
        fi

        if [ -d ".venv" ]; then
            source "$PWD/.venv/bin/activate"
            log_info "Installing jev-mcp and the pinned engine packages from pyproject.toml..."
            pip install -e ".[engines]"
            

            # Same flow as `make start`: restart the fast and smart agents on the active profile
            # (the default one without a marker). The legacy single daemon would fight the fast agent for its port.
            start_profile

            link_mcp_configs

            log_success "Environment updated and both daemons restarted on profile $RESOLVED_PROFILE."
        else
            log_error "No .venv found. Run 'install' first."
            exit 1
        fi
        ;;

    start)
        log_error "'start' no longer takes a model alias. Start a model pair with: make start PROFILE=<name>"
        exit 2
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
        
        manage_daemon stop "$PLIST_PATH"
        manage_daemon stop "$PLIST_FAST_PATH"
        manage_daemon stop "$PLIST_SMART_PATH"

        log_info "Removing launchd plist..."
        rm -f "$PLIST_PATH"
        rm -f "$PLIST_FAST_PATH"
        rm -f "$PLIST_SMART_PATH"
        
        # Remove Opencode Configuration (needs the virtual environment, so it runs before its removal)
        log_info "Removing Opencode integration..."
        if [ -x .venv/bin/python ]; then
            .venv/bin/python -m jev_mcp.client_config unlink-opencode \
                || log_warn "OpenCode config was not updated."
        fi

        log_info "Removing virtual environment..."
        rm -rf .venv
        
        log_info "Removing local model adapters and training artifacts (~/.jev)..."
        rm -rf ~/.jev
        
        log_info "Cleaning up logs..."
        rm -rf "$LOG_DIR" || true
        
        echo -e "${GREEN}Uninstallation Complete!${NC}"
        ;;

    *)
        echo -e "${RED}Invalid command.${NC}"
        echo "Usage:"
        echo "  ./jev_mac_manager.sh install "
        echo "  ./jev_mac_manager.sh hybrid [profile]"
        echo "  ./jev_mac_manager.sh update "
        echo "  ./jev_mac_manager.sh stop"
        echo "  ./jev_mac_manager.sh uninstall"
        exit 1
        ;;
esac
