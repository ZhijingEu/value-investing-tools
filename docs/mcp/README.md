# MCP Config Templates

These files are repo-local templates for configuring Claude Desktop to run this repository's MCP server over STDIO.

They do not affect Claude Desktop by themselves.

Use them by copying the `mcpServers.value-investing-tools-local` entry into your actual Claude Desktop local MCP server config (or add the same values in Claude Desktop UI).

Files:
- `claude-desktop.sample.json` (tracked): example config with placeholders.
- `claude-desktop.local.json` (gitignored): local working copy you can edit safely on your machine.

Recommended workflow:
1. Copy `claude-desktop.sample.json` to `claude-desktop.local.json`.
2. Replace paths/usernames with your local paths.
3. Copy only the server entry into Claude Desktop.
4. Restart Claude Desktop.

Notes:
- Use a unique server name to avoid collisions with existing entries.
- Prefer `.venv` (or `vit-env`) Python executable for isolated dependencies.
- Keep secrets out of these files unless the filename is gitignored.
