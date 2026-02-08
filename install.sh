#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Header
echo ""
echo -e "${BOLD}VoiceFlow Installer${NC}"
echo "Bilingual dictation for macOS"
echo "──────────────────────────────"
echo ""

# Helper functions
info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $1"; exit 1; }

ask() {
    echo -e -n "${BOLD}$1 [y/N]${NC} "
    read -r answer
    case "$answer" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *) return 1 ;;
    esac
}

# 1. Check macOS
if [[ "$(uname)" != "Darwin" ]]; then
    fail "VoiceFlow only runs on macOS. Detected: $(uname)"
fi
ok "macOS detected ($(sw_vers -productVersion), $(uname -m))"

# 2. Check we're in the voiceflow directory
if [[ ! -f "pyproject.toml" ]] || ! grep -q "voiceflow" pyproject.toml 2>/dev/null; then
    fail "Please run this script from the voiceflow project directory.\n       cd /path/to/voiceflow && bash install.sh"
fi
ok "Running from voiceflow project directory"

# 3. Check/install Homebrew
if command -v brew &>/dev/null; then
    ok "Homebrew found ($(brew --version | head -1))"
else
    warn "Homebrew is not installed."
    if ask "Install Homebrew? (recommended)"; then
        info "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        # Add brew to PATH for the rest of this script
        if [[ "$(uname -m)" == "arm64" ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        else
            eval "$(/usr/local/bin/brew shellenv)"
        fi
        ok "Homebrew installed"
    else
        fail "Homebrew is required to install dependencies. Please install it manually:\n       https://brew.sh"
    fi
fi

# 4. Check Python 3.13+
check_python_version() {
    local py="$1"
    if command -v "$py" &>/dev/null; then
        local version
        version=$("$py" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        local major minor
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [[ "$major" -ge 3 ]] && [[ "$minor" -ge 13 ]]; then
            return 0
        fi
    fi
    return 1
}

PYTHON_OK=false
for py in python3.13 python3 python; do
    if check_python_version "$py"; then
        ok "Python 3.13+ found ($($py --version))"
        PYTHON_OK=true
        break
    fi
done

if [[ "$PYTHON_OK" == false ]]; then
    warn "Python 3.13+ is required but not found."
    if ask "Install Python 3.13 via Homebrew?"; then
        info "Installing Python 3.13..."
        brew install python@3.13
        ok "Python 3.13 installed"
    else
        fail "Python 3.13+ is required. Install it manually:\n       brew install python@3.13"
    fi
fi

# 5. Check/install uv
if command -v uv &>/dev/null; then
    ok "uv found ($(uv --version))"
else
    warn "uv package manager is not installed."
    if ask "Install uv? (recommended)"; then
        if command -v brew &>/dev/null; then
            info "Installing uv via Homebrew..."
            brew install uv
        else
            info "Installing uv via curl..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            # shellcheck disable=SC1091
            source "$HOME/.local/bin/env" 2>/dev/null || true
        fi
        ok "uv installed ($(uv --version))"
    else
        fail "uv is required. Install it manually:\n       brew install uv  OR  curl -LsSf https://astral.sh/uv/install.sh | sh"
    fi
fi

# 6. Install dependencies
info "Installing VoiceFlow dependencies..."
uv sync
ok "Dependencies installed"

# 7. Success message
echo ""
echo -e "${GREEN}──────────────────────────────${NC}"
echo -e "${GREEN}${BOLD}VoiceFlow installed successfully!${NC}"
echo -e "${GREEN}──────────────────────────────${NC}"
echo ""
echo "To start VoiceFlow:"
echo -e "  ${BOLD}uv run voiceflow${NC}"
echo ""
echo "To mine jargon from your codebase:"
echo -e "  ${BOLD}uv run voiceflow --mine ~/code${NC}"
echo ""
echo "On first launch, VoiceFlow will guide you through granting"
echo "Accessibility and Microphone permissions."
echo ""
