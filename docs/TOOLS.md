# bro-agent tool catalog

Everything the agent can do, in one place. Tools marked 🔒 are sandboxed
to `BRO_WORKDIR` (default `~/bro-work`); the agent can never touch files
outside it.

## Web

| tool | what it does |
|---|---|
| `web_search` | Keyless web search (DuckDuckGo). Titles, URLs, snippets. |
| `web_fetch` | Fetch a URL as text. SSRF-hardened: DNS-resolution screening, redirect-chain re-validation, credential/port/body-cap blocks. Fetched content is labeled untrusted and planted tool-call fences are neutralized. |

## Files 🔒

| tool | what it does |
|---|---|
| `file_read` | Read a text file (200 KB cap; binaries refused). |
| `file_write` | Create/overwrite a file (5 MB cap). |
| `file_edit` | Surgical replace — the anchor must match exactly once. |
| `file_append` | Append text (creates the file if missing). |
| `file_list` | List a directory (sizes, 500 entries max). |
| `file_search` | Regex grep across text files (skips binaries). |
| `file_mkdir` | Create directories. |
| `file_move` | Move/rename (never overwrites). |
| `file_copy` | Copy a file (never overwrites). |
| `file_delete` | Delete one file (never directories). |
| `file_stat` | Size, type, modification time. |

## Shell 🔒

| tool | what it does |
|---|---|
| `run_shell` | Run a bash command (timeout, capped output). Catastrophic patterns (`rm -rf /`, fork bombs, `mkfs`, raw disk writes, power ops) are refused; everything else passes. `BRO_ACCESS_CODE` is stripped from the environment. |
| `run_python` | Execute Python code (real tracebacks, timeout, capped output). |
| `start_bg` | Start a long-running command in the background → `job_id`. |
| `bg_poll` | Check a background job (running/exited, elapsed, output tail). |
| `bg_kill` | Kill a background job. |

## Git 🔒 (repos must live under the workdir)

| tool | what it does |
|---|---|
| `git_status` | Short status + branch. |
| `git_diff` | Unstaged (or staged) diff. |
| `git_log` | Recent commits, one line each. |
| `git_add` | Stage files. |
| `git_commit` | Commit staged changes (never amends). |
| `git_branch` | List / create-and-switch / delete merged branches. |
| `git_push` | Push a branch. `--force` is refused in code, always. |
| `git_pull` | Pull with `--ff-only` (never merges/rebases). |

Destructive ops — force-push, hard reset, `clean -f`, history rewrites,
remote branch deletes, `checkout -- <paths>` — are refused even when
smuggled inside other arguments.

## Images

| tool | what it does |
|---|---|
| `generate_image` | Keyless AI image generation (Pollinations, flux/turbo). Saves a JPEG under `workdir/images/` and returns the path. For ad creative, thumbnails, mockups. |

Example: "generate an image of a red barn at sunset, 1024x1024" — the agent
calls `generate_image` and gets back `images/img-....jpg`.

## Composio (Jay's connected integrations)

| tool | what it does |
|---|---|
| `composio_list_tools` | List tool slugs for an app (`app_name: "gmail"`, optional `search`) — use first when unsure of the exact slug. |
| `composio_execute` | Run a Composio action (`tool_slug` + `arguments`, optional `connected_account_id`). |
| `composio_apps` | List connected apps and their statuses. |

Reports clearly when `COMPOSIO_API_KEY` is missing or invalid instead of failing.

## Notes for Jay

- The agent sees these tools via its system prompt; just ask in plain words.
- Files the agent writes land in `~/bro-work` on the server (Render free tier:
  ephemeral — download anything that matters).
- Uploads from the chat UI go to `workdir/uploads/`; generated images to
  `workdir/images/`.
