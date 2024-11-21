# Image Classification

## Download

### Python code

File data.py
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("xainano/handwrittenmathsymbols")

print("Path to dataset files:", path)
```

On my mac, the path is:
```bash
/Users/markus/.cache/kagglehub/datasets/xainano/handwrittenmathsymbols/versions/2
```

### Execute
```bash
pip install kagglehub
python data.py
```

## Training
