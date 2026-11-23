# Task 1: submission tools

These scripts upload files to the leaderboard. They read `WMLOC_API_KEY` and `WMLOC_BASE_URL` from the environment (or
from `~/.wmloc_env`) and never print them; no key is in the repository. Run from the task folder: `python -m tools.<name>`.
`src/night.py` imports `is_clean` from `autosubmit.py`.

| Script | Role |
|---|---|
| [autosubmit.py](autosubmit.py) | Checks a candidate (clean nested CV, sending rule, md5 never resent, file checker) and uploads it |
| [fastsend.py](fastsend.py) | Same checks before a slot opens, then uploads at a given clock time |
| [autosend3.py](autosend3.py) | Loop that freezes and sends a new candidate when its nested CV beats the best one sent |
| [sendqueue.py](sendqueue.py) | Sends the first valid candidate of `work/sendqueue.txt` at each slot opening |
