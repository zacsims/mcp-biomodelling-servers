#!/usr/bin/env python3
"""PreToolUse hook enforcing the blind-calibration constraint.

Blocks tool calls (Read/Edit/Write/Grep/Glob/Bash and common MCP filesystem
tools) whose path arguments fall under the target PhysiCell project tree,
with the exception of the target's output folder. This is the mechanical
enforcement tier of the constraint — defense in depth alongside the prompt
reminder and the driver-side audit of iter_record.json.

Reads configuration from the file pointed to by CALIBRATION_MANIFEST env var.
That manifest must contain at minimum:
    {
      "target_output_folder": "/abs/path/to/target/PhysiCell/output",
      "target_project_root":  "/abs/path/to/target/PhysiCell"   // optional
    }
If target_project_root is absent it defaults to the parent of
target_output_folder.

Hook protocol: stdin is a JSON object with keys "tool_name" and "tool_input".
Exit 0 with no output to allow; exit 2 with a message on stderr to deny.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path


ALLOW = 0
DENY = 2


def _load_manifest() -> dict | None:
    path = os.environ.get("CALIBRATION_MANIFEST")
    if not path:
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _normalize(p: str) -> str:
    return os.path.realpath(os.path.expanduser(os.path.expandvars(p)))


def _is_under(candidate: str, root: str) -> bool:
    try:
        Path(candidate).relative_to(root)
        return True
    except ValueError:
        return False


def _collect_paths_from_bash(command: str) -> list[str]:
    """Pull path-like tokens out of a bash command string.

    Not a full shell parser — we use shlex for the easy case and a regex
    fallback for anything path-shaped that survives. We intentionally
    over-collect: false positives just trigger an innocuous deny, while
    false negatives would let a forbidden read through.
    """
    tokens: list[str] = []
    try:
        tokens.extend(shlex.split(command, posix=True))
    except ValueError:
        pass
    # Also regex-scan for absolute paths that shlex may have glued into
    # compound tokens (e.g. inside $(...), backticks, redirects).
    tokens.extend(re.findall(r"(?<![\w./-])(/[\w./-]+)", command))
    # And relative path fragments that look like they reference the
    # target project by name (last path component of target_project_root).
    return tokens


def _decide(tool_name: str, tool_input: dict, manifest: dict) -> tuple[int, str]:
    output_folder = manifest.get("target_output_folder")
    if not output_folder:
        return ALLOW, ""
    output_folder = _normalize(output_folder)
    project_root = manifest.get("target_project_root") or str(Path(output_folder).parent)
    project_root = _normalize(project_root)

    def _check(path_str: str) -> tuple[int, str]:
        if not path_str:
            return ALLOW, ""
        # Anchors only make sense for absolute-ish paths; relative paths
        # are resolved against CWD, which the driver sets outside the target.
        candidate = _normalize(path_str)
        if _is_under(candidate, output_folder):
            return ALLOW, ""
        if _is_under(candidate, project_root):
            return DENY, (
                f"BLIND-CONSTRAINT VIOLATION: '{path_str}' resolves to "
                f"'{candidate}', which is under the target project root "
                f"('{project_root}') but NOT under the permitted output "
                f"folder ('{output_folder}'). The calibration agent may "
                "read ONLY the target's output files."
            )
        return ALLOW, ""

    # File-path-bearing tools.
    path_keys = ("file_path", "path", "notebook_path")
    for key in path_keys:
        if key in tool_input:
            status, msg = _check(str(tool_input[key]))
            if status == DENY:
                return status, msg

    # Grep/Glob carry path plus pattern; path is the interesting one.
    if tool_name in ("Grep", "Glob") and "path" in tool_input:
        status, msg = _check(str(tool_input["path"]))
        if status == DENY:
            return status, msg

    # Bash — scan the command string for forbidden path tokens.
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        for token in _collect_paths_from_bash(cmd):
            status, msg = _check(token)
            if status == DENY:
                return status, msg
        # Also reject commands that reference the target project's
        # basename with a relative path prefix (e.g. `../target/config`).
        proj_name = Path(project_root).name
        if proj_name and re.search(
            rf"(?<![\w./-])\.\.?/[^\s]*{re.escape(proj_name)}[^\s]*",
            cmd,
        ):
            return DENY, (
                f"BLIND-CONSTRAINT VIOLATION: Bash command appears to "
                f"reference the target project '{proj_name}' via a "
                "relative path. All target access must go through the "
                "permitted output folder only."
            )

    return ALLOW, ""


def main() -> int:
    manifest = _load_manifest()
    if manifest is None:
        # No manifest configured → hook is a no-op. The driver ALWAYS sets
        # CALIBRATION_MANIFEST before spawning; an unset var means we're
        # being run outside the calibration context.
        return ALLOW

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return ALLOW

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}
    status, msg = _decide(tool_name, tool_input, manifest)
    if status == DENY:
        sys.stderr.write(msg + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
