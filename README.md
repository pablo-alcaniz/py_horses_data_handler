# py_horses_data_handler

Librería de Python para manejar los datos de solución que genera el solver CFD HORSES3D: convierte los binarios `.hsol` a ficheros [HDF5](https://www.hdfgroup.org/solutions/hdf5/) (`.h5`/`.hdf`) y ofrece herramientas para inspeccionarlos.

Requiere Python 3.13 o superior y una instalación de HORSES3D cuyo `horses2plt` esté compilado con soporte HDF5 (ver `SOLVER_PATH` en la sección 3.1). La librería incluye anotaciones de tipos (fichero `py.typed`), de modo que los analizadores de tipos como `pyrefly` o `mypy` las reconocen al importarla.

## 1. Instalación

1. Haz una copia local de la librería:

    ```bash
    git clone https://github.com/pablo-alcaniz/py_horses_data_handler.git
    ```

2. Instálala en el proyecto que quieras con tu gestor de entornos (por ejemplo `uv` o `pip`):

    ```bash
    uv add [directorio de instalación de la librería]
    ```

    o bien

    ```bash
    pip install -e [directorio de instalación de la librería]
    ```

## 2. Configuración básica

Importa la clase `HorsesDataHandler` y crea un objeto. Crear el objeto no necesita argumentos y no imprime nada:

```python
from py_horses_data_handler import HorsesDataHandler

data_handler = HorsesDataHandler()
```

Los nombres de los argumentos de los métodos van en MAYÚSCULAS (`SOLVER_PATH`, `HSOL_DIR`...). Es una decisión de diseño del proyecto y se mantiene.

Las rutas se pueden pasar como `str` o como cualquier objeto tipo ruta (`os.PathLike[str]`, por ejemplo un `pathlib.Path`).

## 3. Funciones

### 3.1. `convert_hsol2hdf`

Convierte los binarios `.hsol` de una carpeta en ficheros HDF5. Usa la utilidad `horses2plt`, que viene con HORSES3D, para generar ficheros `.hdf` (con estructura VTKHDF, el formato de VTK sobre HDF5); después los deja en la carpeta de salida con la extensión pedida (`.h5` por defecto, por compatibilidad).

Nota: para visualizar correctamente en Paraview hacen falta ficheros con extensión `.hdf`.

```python
convert_hsol2hdf(
    SOLVER_PATH: str | os.PathLike[str],
    HSOL_DIR: str | os.PathLike[str],
    MESH_DIR: str | os.PathLike[str],
    OUT_FILES_DIR: str | os.PathLike[str] | None = None,
    OUTPUT_PARAMETERS: str | None = None,
    OUT_EXTENSION: Literal[".h5", ".hdf", "both"] = ".h5",
    LOG: bool = True,
    WRITE_SIM_PARAMETER: bool = True,
    EXTENDED_LOG: bool = False,
) -> None
```

**Argumentos**

| Argumento | Tipo | Significado | Por defecto |
|---|---|---|---|
| `SOLVER_PATH` | ruta | Raíz de una instalación de HORSES3D. Debe contener `Solver/bin/horses2plt`, y este debe estar compilado con HDF5 (`WITH_HDF5=YES`). Si no lo está, `horses2plt` falla con «HDF5 support must be enabled to save to VTKHDF format» y la librería lanza `RuntimeError`. | obligatorio |
| `HSOL_DIR` | ruta | Carpeta con los ficheros `.hsol`. Se procesan todos, ordenados por nombre. | obligatorio |
| `MESH_DIR` | ruta | Fichero `.hmesh` o carpeta que contenga exactamente uno. Si la carpeta tiene varios, da `ValueError` (habría que elegir uno al azar). Pasa entonces la ruta del `.hmesh` concreto. | obligatorio |
| `OUT_FILES_DIR` | ruta o `None` | Carpeta donde se guardan los resultados; se crea si no existe. Si es `None`, se usa `HSOL_DIR`. | `None` |
| `OUTPUT_PARAMETERS` | `str` o `None` | Opciones de `horses2plt` en una sola cadena, troceada como lo haría un terminal (se respetan las comillas). Si es `None`, se usa `"--output-mode=FE --output-variables=rho,u,v,w,p,T,Mach --output-type=vtkhdf"`. | `None` |
| `OUT_EXTENSION` | `".h5"`, `".hdf"` o `"both"` | Extensión de los ficheros de salida. Con `"both"` se guarda una copia `.hdf` en `OUT_FILES_DIR/hdf` y otra `.h5` en `OUT_FILES_DIR/h5`. | `".h5"` |
| `LOG` | `bool` | Si es `True`, la librería imprime mensajes de progreso que empiezan por `>>>`. | `True` |
| `WRITE_SIM_PARAMETER` | `bool` | Si es `True`, escribe el tiempo físico y la iteración de cada solución dentro del fichero (ver más abajo). Ralentiza la conversión unas 4 veces, porque lanza un proceso de `horses2plt` por fichero en lugar de uno solo para todos. | `True` |
| `EXTENDED_LOG` | `bool` | Si es `True`, imprime la salida completa de `horses2plt`. | `False` |

**Devuelve:** `None`. El resultado son los ficheros escritos en disco.

**Qué se escribe con `WRITE_SIM_PARAMETER`.** `horses2plt` imprime el tiempo y la iteración de cada solución, y la librería los lee de esa salida y los guarda como datasets (matrices de datos de HDF5) en el grupo `/VTKHDF/SimulationInfo` de cada fichero:

- `Time`: tiempo físico, en `float64` (con 4 cifras significativas; ver más abajo).
- `Iteration`: número de iteración, en `int64`.

Se guardan en doble precisión (64 bits) para no perder nada al escribir: antes eran `float32`/`int32`, y `float32` solo conserva unas 7 cifras significativas. Si el fichero ya tenía estos datasets, se sobrescriben.

**Precisión de `Time`.** `horses2plt` imprime el tiempo con solo 4 cifras significativas (por ejemplo `Time:  3.732E-02`) y la librería lo lee de ese texto. Por tanto `Time` tiene 4 cifras significativas aunque se guarde en `float64`: la doble precisión evita perder datos al guardar, pero no añade precisión que el texto no tiene. `Iteration` es un entero exacto.

**Límite de 132 bytes en las rutas.** HORSES3D guarda las rutas de fichero en variables de texto de longitud fija (`LINE_LENGTH = 132`, en `Solver/src/libs/foundation/SMConstants.f90`) y corta en silencio las más largas; con un `.hsol` demasiado largo, `horses2plt` ni siquiera da error: simplemente no convierte la solución. Con caracteres ASCII son 132 caracteres; una tilde o una ñ ocupan 2 bytes. Para alejarse del límite, la librería ejecuta `horses2plt` con `HSOL_DIR` como carpeta de trabajo, le pasa cada `.hsol` solo por su nombre y la malla con la más corta entre su ruta absoluta y la relativa a `HSOL_DIR`. Así, el límite afecta al nombre de cada `.hsol` y a esa ruta de la malla, pero no a `HSOL_DIR`, `SOLVER_PATH` ni `OUT_FILES_DIR`. Si se supera, se lanza `ValueError` antes de crear nada.

**Espacios en las rutas.** Funcionan, tanto en las carpetas como en los nombres de fichero.

**Qué ficheros se mueven.** Solo se mueven o copian los `.hdf` generados en esa ejecución. Otros `.hdf` que ya hubiera en `HSOL_DIR` se dejan donde están. La librería supone que `horses2plt` escribe cada `<nombre>.hdf` junto a su `<nombre>.hsol`.

**Errores**

| Excepción | Cuándo salta |
|---|---|
| `ValueError` | `OUT_EXTENSION` no es una de las tres opciones; `MESH_DIR` es un fichero sin extensión `.hmesh` o una carpeta con varios `.hmesh`; un `.hdf` generado no tiene el grupo `/VTKHDF` (no se creó con `--output-type=vtkhdf`); no se encuentra el tiempo o la iteración en la salida de `horses2plt`; o el nombre de algún `.hsol`, o la ruta de la malla (la más corta entre absoluta y relativa a `HSOL_DIR`), supera los 132 bytes. |
| `FileNotFoundError` | No existe `SOLVER_PATH`, `HSOL_DIR`, `horses2plt` o la malla; no hay ningún `.hsol` en `HSOL_DIR`; o falta algún `.hdf` esperado tras la conversión. |
| `RuntimeError` | `horses2plt` termina con un código de salida distinto de 0, o su salida contiene `Error in file` o `ERROR STOP` aunque el código sea 0 (así falla, p. ej., si la ruta de la malla es demasiado larga: «Mesh file was not specified»). El mensaje incluye el comando, la carpeta de trabajo y las últimas 20 líneas de la salida. |

Todos los argumentos se validan antes de crear nada en disco, de modo que un error de configuración no deja carpetas a medias.

**Ejemplo**

```python
from pathlib import Path

from py_horses_data_handler import HorsesDataHandler

data_handler = HorsesDataHandler()

data_handler.convert_hsol2hdf(
    SOLVER_PATH=Path("path/to/horses3d/installation"),
    HSOL_DIR="path/to/hsol/files",
    MESH_DIR="path/to/mesh/file.hmesh",
    OUT_FILES_DIR="path/to/output/files",
    OUT_EXTENSION="both",
)
```

### 3.2. `get_sizes`

Suma el tamaño de los ficheros de una extensión dentro de una carpeta (sin entrar en subcarpetas). Imprime el total en MB (por debajo de 1 GB) o en GB, con prefijos decimales (1 MB = 10^6 bytes), y además lo devuelve.

```python
get_sizes(PATH: str | os.PathLike[str], EXTENSION: str = ".hdf") -> int
```

**Argumentos**

| Argumento | Tipo | Significado | Por defecto |
|---|---|---|---|
| `PATH` | ruta | Carpeta que se examina. | obligatorio |
| `EXTENSION` | `str` | Terminación de los nombres de fichero que se cuentan. | `".hdf"` |

**Devuelve:** un `int` con la suma de los tamaños, en bytes.

```python
n_bytes = data_handler.get_sizes("path/to/output/files", ".h5")
# imprime, por ejemplo: >>> Data size: 152.3 MB
```

### 3.3. `explore_hdf5_structure`

Imprime los grupos y datasets de un fichero HDF5. Para cada dataset muestra su forma (`shape`) y su tipo de dato; el anidamiento se indica con sangría.

```python
explore_hdf5_structure(FILEPATH: str | os.PathLike[str]) -> None
```

**Argumentos**

| Argumento | Tipo | Significado | Por defecto |
|---|---|---|---|
| `FILEPATH` | ruta | Fichero `.hdf`/`.h5` que se inspecciona. | obligatorio |

**Devuelve:** `None`.

```python
data_handler.explore_hdf5_structure("path/to/output/files/solution.h5")
```

## 4. Desarrollo

Las comprobaciones se lanzan con `uv` desde la raíz del repositorio (`--all-groups` instala también las herramientas de desarrollo):

```bash
uv sync --all-groups                    # prepara el entorno con todas las dependencias
uv run --all-groups pytest -q           # tests
uv run --all-groups pyrefly check       # comprobación de tipos
uv run --all-groups ruff check          # revisión de estilo y errores comunes
uv run --all-groups ruff format --check # comprueba el formato sin modificar nada
```

- `pytest` ejecuta los tests automáticos de la carpeta `tests/`.
- `pyrefly` es un comprobador de tipos: lee las anotaciones y avisa de usos incoherentes sin ejecutar el código.
- `ruff check` es un linter: revisa el código buscando errores comunes y faltas de estilo.
- `ruff format --check` indica si el código cumple el formato estándar; sin `--check`, lo reformatea.
