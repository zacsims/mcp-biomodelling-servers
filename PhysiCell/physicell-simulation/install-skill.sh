#!/usr/bin/env bash
# Install the physicell-simulation AgentSkill for Claude Code.
#
# Usage:
#   bash physicell-simulation/install-skill.sh
#
# Prerequisites:
#   - Claude Code installed
#   - life-sciences marketplace present at ~/.claude/plugins/marketplaces/life-sciences/

set -euo pipefail

SKILL_DIR="$(cd "$(dirname "$0")" && pwd)"
MARKETPLACE_DIR="$HOME/.claude/plugins/marketplaces/life-sciences"
CACHE_DIR="$HOME/.claude/plugins/cache/life-sciences/physicell-simulation/1.0.0"
INSTALLED_PLUGINS="$HOME/.claude/plugins/installed_plugins.json"
SETTINGS="$HOME/.claude/settings.json"

# Check prerequisites
if [ ! -d "$MARKETPLACE_DIR" ]; then
  echo "Error: life-sciences marketplace not found at $MARKETPLACE_DIR"
  echo "Install it first or adjust the script for your marketplace location."
  exit 1
fi

echo "Installing physicell-simulation skill..."

# 1. Copy skill to marketplace directory
echo "  Copying skill to marketplace..."
rm -rf "$MARKETPLACE_DIR/physicell-simulation"
cp -r "$SKILL_DIR" "$MARKETPLACE_DIR/physicell-simulation"

# 2. Create cache directory with plugin manifest
echo "  Creating plugin cache..."
rm -rf "$CACHE_DIR"
mkdir -p "$CACHE_DIR/.claude-plugin"
cp -r "$SKILL_DIR"/* "$CACHE_DIR/"
cat > "$CACHE_DIR/.claude-plugin/plugin.json" << 'MANIFEST'
{
  "name": "physicell-simulation",
  "version": "1.0.0",
  "description": "Build, configure, and run PhysiCell multicellular simulations. Prevents common configuration mistakes.",
  "skills": ["."]
}
MANIFEST

# 3. Add to marketplace.json if not already present
MARKETPLACE_JSON="$MARKETPLACE_DIR/.claude-plugin/marketplace.json"
if ! grep -q '"physicell-simulation"' "$MARKETPLACE_JSON" 2>/dev/null; then
  echo "  Registering in marketplace.json..."
  # Insert before the closing ] of the plugins array
  python3 -c "
import json
with open('$MARKETPLACE_JSON', 'r') as f:
    data = json.load(f)
data['plugins'].append({
    'name': 'physicell-simulation',
    'source': './',
    'description': 'Build, configure, and run PhysiCell multicellular simulations.',
    'category': 'life-sciences',
    'tags': ['simulation', 'multicellular', 'physicell', 'biology', 'modeling'],
    'strict': False,
    'skills': ['./physicell-simulation']
})
with open('$MARKETPLACE_JSON', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
else
  echo "  Already registered in marketplace.json."
fi

# 4. Add to installed_plugins.json if not already present
if ! grep -q '"physicell-simulation@life-sciences"' "$INSTALLED_PLUGINS" 2>/dev/null; then
  echo "  Registering in installed_plugins.json..."
  TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")
  python3 -c "
import json
with open('$INSTALLED_PLUGINS', 'r') as f:
    data = json.load(f)
data['plugins']['physicell-simulation@life-sciences'] = [{
    'scope': 'user',
    'installPath': '$CACHE_DIR',
    'version': '1.0.0',
    'installedAt': '$TIMESTAMP',
    'lastUpdated': '$TIMESTAMP'
}]
with open('$INSTALLED_PLUGINS', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
else
  echo "  Already registered in installed_plugins.json."
fi

# 5. Enable in settings.json if not already enabled
if ! grep -q '"physicell-simulation@life-sciences"' "$SETTINGS" 2>/dev/null; then
  echo "  Enabling in settings.json..."
  python3 -c "
import json
with open('$SETTINGS', 'r') as f:
    data = json.load(f)
if 'enabledPlugins' not in data:
    data['enabledPlugins'] = {}
data['enabledPlugins']['physicell-simulation@life-sciences'] = True
with open('$SETTINGS', 'w') as f:
    json.dump(data, f, indent=2)
    f.write('\n')
"
else
  echo "  Already enabled in settings.json."
fi

echo ""
echo "Done! The physicell-simulation skill is installed."
echo "Start a new Claude Code session and run /skills to verify."
