# toml-edit

A format-preserving TOML parser and writer for Python.

## Usage

```python
import toml_edit

# Files must be opened in binary mode.
with open("pyproject.toml", "rb") as f:
    doc = toml_edit.load(f)

doc["project"]["version"] = "0.2.0"
doc["project"]["dependencies"].append("requests>=2")

print(toml_edit.dumps(doc))   # comments and layout are preserved
```

### Comment API

```python
doc = toml_edit.loads("""
[server]
host = "localhost"  # default
port = 8080
""")

server = doc.table("server")
server.comments["port"] = "override with $PORT"
server.comments["host"] = None         # clear

print(toml_edit.dumps(doc))
# [server]
# host = "localhost"
# port = 8080 # override with $PORT
```
