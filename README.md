# mini_webarena
## Install
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
uv pip install ray fastapi uvicorn fire transformers scipy
uv pip uninstall uvloop # a weird problem
```

## Try directly
```bash
python -m mini_webarena.env
```

## Try in verl-tool
```bash
ray stop
ray start --head
python -m verl_tool.servers.serve         --tool_type text_browser         --url=http://localhost:5003/get_observation --port 5003


# under verl-tool environment
python -m verl_tool.servers.tests.test_text_browser browser             --url=http://localhost:5003/get_observation
```
