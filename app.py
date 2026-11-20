"""Root entrypoint for Streamlit Community Cloud and local execution."""
import os
import sys
from pathlib import Path

# Ensure 'src' and project root are always in sys.path
_root = Path(__file__).resolve().parent
_src = _root / "src"

if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

os.chdir(_root)

import runpy

app_target = _src / "smartcret" / "app" / "app.py"
runpy.run_path(str(app_target), run_name="__main__")
