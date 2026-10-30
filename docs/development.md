# Development setup

## Where things live

| Path | What | Who edits |
|---|---|---|
| `/srv/sfsu_srv/wf-spread` | Shared copy: always on `main`. Holds the shared venv (`.venv/`), caches, `data/` and `runs/` | nobody edits code here; update with `git pull --ff-only` after merges |
| `/srv/sfsu_srv/dev/<user>/wf-spread` | Your personal clone, where you branch, edit, test and commit | you |

`/srv/sfsu_srv` is the same folder on both Sparks over NFS, so a clone made on one machine works on the other.

## One-time setup (each person)

```bash
git config --global user.name  "Your Name"
git config --global user.email "<your GitHub noreply or account email>"
mkdir -p /srv/sfsu_srv/dev/$USER && cd /srv/sfsu_srv/dev/$USER
git clone https://github.com/Tieahapani/Wildfire-Spread-Prediction-DeepLearning.git wf-spread
cd wf-spread && scripts/setup/dev-setup.sh      # hooks + rebase-by-default
bin/wf pytest -q                                # should pass
```

To push, you need a GitHub login on the Spark. Either run `gh auth login` (gh is installed) or use a personal access token.

## Daily loop

```bash
git switch main && git pull                     # pull.rebase is on
git switch -c feat/<task>
# edit → bin/wf ruff format . → bin/wf ruff check . → bin/wf pytest -q → git commit
git fetch origin && git rebase origin/main && bin/wf pytest -q
git push                                        # pre-push hook runs the tests again
gh pr create --fill                             # or open it on GitHub
```

## Commands

| Task | Command |
|---|---|
| Python, anything | `bin/wf python ...` |
| Tests (default, no `slow`) | `bin/wf pytest -q` |
| Slow regression tests | `bin/wf pytest -q -m slow` |
| Format, lint | `bin/wf ruff format .`, `bin/wf ruff check .` |
| Shell in the container | `bin/wf bash` |
| Add a package | see CONTRIBUTING §9 |

## Hooks (`.githooks/`, enabled by `scripts/setup/dev-setup.sh`)

| Hook | Checks |
|---|---|
| `pre-commit` | staged Python files are formatted and lint-clean |
| `commit-msg` | Conventional Commits subject, 72 characters or fewer, blank second line |
| `pre-push` | not pushing to `main`; branch contains `origin/main`; tests pass |
