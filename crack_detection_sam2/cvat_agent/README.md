# Heritage craq+crack CVAT AI agent

Local auto-annotation for `craquelure` + `crack` mask labels on app.cvat.ai. Model runs locally;
only function metadata is uploaded.

## Prereqs
- venv `cvat_agent_env` built (see requirements.txt + spec).
- A Personal Access Token from app.cvat.ai.
- CVAT project/task has labels named exactly `craquelure` and `crack`, both **mask** type.

## Use
```bash
export CVAT_ACCESS_TOKEN=<your PAT>     # do not commit
bash register.sh                         # one-time; note the printed function-id
bash run_agent.sh <function-id>          # keep running while you annotate
```
Then in the CVAT task: Actions / Automatic annotation -> pick "Heritage craq+crack" -> run -> review.

## Params (override via -p name=type:value on the cvat-cli line)
craq_ckpt, crack_ckpt, device, tile(512), stride(256), thresh_craq(0.5), thresh_crack(0.5),
min_blob(64), priority(craq_over_crack).

## Notes
- craquelure quality > crack quality; expect more manual fixing on crack.
- Agent must stay running; stop with Ctrl-C.
