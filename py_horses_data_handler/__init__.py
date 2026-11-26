"""py_horses_data_handler: posprocesado de las soluciones del solver CFD HORSES3D.

Punto de entrada del paquete. Reexporta la clase `HorsesDataHandler` (definida en
``core.py``) para que los usuarios la importen con
``from py_horses_data_handler import HorsesDataHandler``.

``__version__`` se lee de los metadatos del paquete instalado, de modo que siempre coincide
con la versión declarada en ``pyproject.toml``.
"""

from importlib.metadata import version

from .core import HorsesDataHandler

__version__: str = version("py-horses-data-handler")
__author__: str = "Pablo Alcañiz"

__all__ = ["HorsesDataHandler"]
