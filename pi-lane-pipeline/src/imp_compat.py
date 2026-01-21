"""
Compatibility shim for 'imp' module (removed in Python 3.12+).
This allows packages that still use 'imp' to work on Python 3.13.
"""
import sys
import importlib
import importlib.util
import importlib.machinery

if sys.version_info >= (3, 12):
    class ImpModule:
        """Minimal imp module compatibility shim for Python 3.12+."""
        
        PY_SOURCE = 1
        PY_COMPILED = 2
        C_EXTENSION = 3
        PY_RESOURCE = 4
        PKG_DIRECTORY = 5
        C_BUILTIN = 6
        PY_FROZEN = 7
        
        @staticmethod
        def find_module(name, path=None):
            """Find module (deprecated, use importlib instead)."""
            try:
                spec = importlib.util.find_spec(name, path)
                if spec:
                    loader = spec.loader
                    # Return a mock object with file attribute
                    class MockLoader:
                        def __init__(self, spec):
                            self.spec = spec
                            if hasattr(spec, 'origin') and spec.origin:
                                self.file = open(spec.origin, 'rb') if spec.origin else None
                            else:
                                self.file = None
                    return MockLoader(spec) if loader else None
                return None
            except (ImportError, ValueError, AttributeError):
                return None
        
        @staticmethod
        def load_module(name, file=None, pathname=None, description=None):
            """Load module (deprecated)."""
            if pathname:
                spec = importlib.util.spec_from_file_location(name, pathname)
            else:
                spec = importlib.util.find_spec(name)
            
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
                return module
            raise ImportError(f"Could not load module {name}")
        
        @staticmethod
        def new_module(name):
            """Create a new module."""
            return type(sys)(name)
    
    # Inject into sys.modules so 'import imp' works
    sys.modules['imp'] = ImpModule()