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
| `WRITE_SIM_PARAMETER` | `bool` | Si es `True`, lee la cabecera binaria de cada `.hsol` y escribe su tiempo físico, su iteración y sus valores de referencia dentro del fichero convertido (ver más abajo). No ralentiza la conversión: `horses2plt` se llama una sola vez en cualquier caso. Si es `False`, no se leen las cabeceras. | `True` |
| `EXTENDED_LOG` | `bool` | Si es `True`, imprime la salida completa de `horses2plt`. | `False` |

**Devuelve:** `None`. El resultado son los ficheros escritos en disco.

**Qué se escribe con `WRITE_SIM_PARAMETER`.** La librería lee la *cabecera* de cada `.hsol` (los primeros 204 bytes del fichero, donde HORSES3D guarda los datos generales de la solución antes de los datos de los elementos) y guarda estos valores como datasets (matrices de datos de HDF5) escalares en el grupo `/VTKHDF/SimulationInfo` del `.hdf` correspondiente:

| Dataset | Tipo | Significado |
|---|---|---|
| `Time` | `float64` | Tiempo físico de la solución. |
| `Iteration` | `int64` | Número de iteración. |
| `Gamma` | `float64` | Cociente de calores específicos (adimensional). |
| `GasConstant` | `float64` | Constante del gas R, en J/(kg·K). |
| `ReferenceVelocity` | `float64` | Velocidad de referencia, en m/s. |
| `ReferenceDensity` | `float64` | Densidad de referencia, en kg/m³. |
| `ReferenceTemperature` | `float64` | Temperatura de referencia, en K. |
| `MachNumber` | `float64` | Número de Mach de referencia (adimensional). |

Las unidades de los valores de referencia están deducidas de su significado físico, no vienen indicadas en el fichero. Si el `.hdf` ya tenía estos datasets, se sobrescriben.

**Precisión.** Los valores se leen en binario, tal como los escribe HORSES3D, y se guardan con toda su precisión de doble precisión (64 bits). Antes se leían del texto que imprime `horses2plt`, que solo da 4 cifras significativas (`Time:  4.033E-02` frente al valor real `0.04032508838441574`).

**Formato de la cabecera de un `.hsol`.** HORSES3D abre el fichero con *acceso stream* (`form="unformatted", access="stream"`, en `Solver/src/libs/io/SolutionFile.f90`, líneas 152-153): escribe los bytes uno detrás de otro, sin los marcadores de registro que Fortran añade en otros modos. Los números se guardan en el orden de bytes de la máquina, *little-endian* (el byte menos significativo primero) en un PC x86-64. La librería lo lee con el módulo `struct` de la biblioteca estándar. Los offsets (posición en bytes desde el inicio del fichero, empezando en 0) son:

| Offset | Bytes | Tipo | Campo | Línea de `SolutionFile.f90` donde se escribe |
|---|---|---|---|---|
| 0 | 128 | texto, relleno con `\0` | nombre del fichero | 157 |
| 128 | 4 | `int32` | tipo de fichero (2 = solución, 3 = solución y gradientes...) | 161 |
| 132 | 4 | `int32` | tipo de nodos (Gauss o Gauss-Lobatto) | 165 |
| 136 | 4 | `int32` | número de elementos | 169 |
| 140 | 4 | `int32` | iteración | 174 |
| 144 | 8 | `float64` | tiempo físico | 175 |
| 152 | 48 | 6 × `float64` | referencias: gamma, R, velocidad, densidad, temperatura, Mach | 179 |
| 200 | 4 | `int32` | terminador, siempre 88 (`BEGINNING_DATA`) | 183 |
| 204 | | | empiezan los datos de los elementos | |

La librería comprueba el terminador (88) y el tipo de fichero: solo acepta los tipos de solución que admite `horses2plt` (2, 3, 4, 7, 8, 9 y 10). Un fichero de malla (tipos 1 y 6), un tipo desconocido, un fichero truncado o uno escrito con otro orden de bytes o con enteros de otro tamaño dan `ValueError`, antes de crear carpetas o llamar a `horses2plt`. Los datos de los elementos (a partir del byte 204) no se leen.

**Límite de 132 bytes en las rutas.** HORSES3D guarda las rutas de fichero en variables de texto de longitud fija (`LINE_LENGTH = 132`, en `Solver/src/libs/foundation/SMConstants.f90`) y corta en silencio las más largas; con un `.hsol` demasiado largo, `horses2plt` ni siquiera da error: simplemente no convierte la solución. Con caracteres ASCII son 132 caracteres; una tilde o una ñ ocupan 2 bytes. Para alejarse del límite, la librería ejecuta `horses2plt` con `HSOL_DIR` como carpeta de trabajo, le pasa cada `.hsol` solo por su nombre y la malla con la más corta entre su ruta absoluta y la relativa a `HSOL_DIR`. Así, el límite afecta al nombre de cada `.hsol` y a esa ruta de la malla, pero no a `HSOL_DIR`, `SOLVER_PATH` ni `OUT_FILES_DIR`. Si se supera, se lanza `ValueError` antes de crear nada.

**Espacios en las rutas.** Funcionan, tanto en las carpetas como en los nombres de fichero.

**Qué ficheros se mueven.** Solo se mueven o copian los `.hdf` generados en esa ejecución. Otros `.hdf` que ya hubiera en `HSOL_DIR` se dejan donde están. La librería supone que `horses2plt` escribe cada `<nombre>.hdf` junto a su `<nombre>.hsol`.

**Errores**

| Excepción | Cuándo salta |
|---|---|
| `ValueError` | `OUT_EXTENSION` no es una de las tres opciones; `MESH_DIR` es un fichero sin extensión `.hmesh` o una carpeta con varios `.hmesh`; un `.hdf` generado no tiene el grupo `/VTKHDF` (no se creó con `--output-type=vtkhdf`); con `WRITE_SIM_PARAMETER=True`, la cabecera de algún `.hsol` no es válida (menos de 204 bytes, terminador distinto de 88 o tipo que no es de solución); o el nombre de algún `.hsol`, o la ruta de la malla (la más corta entre absoluta y relativa a `HSOL_DIR`), supera los 132 bytes. |
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
