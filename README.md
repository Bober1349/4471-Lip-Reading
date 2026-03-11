# 4471 Lip Reading

## Set up enviornment

Check up your nvidia driver version if >=570, you need a new driver to support wsl2.

Install uv to sync library: <br>
```curl -LsSf https://astral.sh/uv/install.sh | sh```

Restart Bash

Create venv with: ```uv venv```

Install Libraies (install from requirements.txt should be a lot easier)<br>
```uv pip install torch``` </br>
```uv pip install ultralytics``` </br>
```uv pip install opencv-python``` </br>
```uv pip install jupyter```</br>
or </br>
```uv pip install -r requirements.txt``` </br>
or


Saving your current library set up </br>
```uv pip freeze > requirements.txt```

If you would like to test if GPU activated, paste into bash </br>
```python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda); print('device count:', torch.cuda.device_count()); print('device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"```