import importlib.util
import os

def get_library_path(library_name):
    try:
        spec = importlib.util.find_spec(library_name)
        if spec and spec.origin:
            return os.path.dirname(spec.origin)
        return f"{library_name} not found"
    except ImportError:
        return f"{library_name} not found"

