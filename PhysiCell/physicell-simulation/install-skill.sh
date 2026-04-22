#!/usr/bin/env bash
# Install the physicell-simulation skill as a Claude Code user skill.
#
# Usage:
#   bash physicell-simulation/install-skill.sh
#
# Installs to ~/.claude/skills/physicell-simulation/ — no plugin/marketplace
# machinery required. Start a new Claude Code session and run /skills to verify.

set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST_DIR="$HOME/.claude/skills/physicell-simulation"

echo "Installing physicell-simulation skill to $DEST_DIR ..."

mkdir -p "$(dirname "$DEST_DIR")"
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR"

cp "$SRC_DIR/SKILL.md" "$DEST_DIR/"
cp -r "$SRC_DIR/references" "$DEST_DIR/"
cp -r "$SRC_DIR/scripts" "$DEST_DIR/"

echo "Done. Start a new Claude Code session and run /skills to verify."
