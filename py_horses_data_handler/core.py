"""Núcleo de la librería: conversión e inspección de soluciones de HORSES3D.

Este módulo contiene la clase pública `HorsesDataHandler`, que es lo que importan los
usuarios (se reexporta desde ``py_horses_data_handler/__init__.py``), y funciones
auxiliares privadas. Entre ellas, `_read_hsol_header` lee la cabecera binaria de un
``.hsol`` (sus 204 primeros bytes) para obtener el tiempo físico, la iteración y los valores
de referencia con toda su precisión (la salida de texto de ``horses2plt`` solo da 4 cifras).

Flujo de `HorsesDataHandler.convert_hsol2hdf`:

1. Valida todos los argumentos antes de crear nada en disco, incluido el límite de 132
   bytes (con caracteres ASCII, 132 caracteres) que HORSES3D impone a las rutas de fichero
   y, si se van a escribir, las cabeceras de todos los ``.hsol``.
2. Llama una sola vez a la utilidad ``horses2plt`` de HORSES3D (un proceso externo), desde
   la carpeta de los ``.hsol``, para convertir los binarios ``.hsol`` en ficheros HDF5 con
   estructura VTKHDF (``.hdf``).
3. Opcionalmente, escribe en cada ``.hdf`` el tiempo, la iteración y los valores de
   referencia leídos de la cabecera de su ``.hsol``.
4. Mueve o copia los ficheros generados a la carpeta de salida con la extensión pedida.
"""

import os
import shlex
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py

type PathLike = str | os.PathLike[str]
"""Ruta aceptada por la interfaz pública: un ``str`` o un objeto tipo ruta (``pathlib.Path``)."""

type OutExtension = Literal[".h5", ".hdf", "both"]
"""Extensiones de salida admitidas por `HorsesDataHandler.convert_hsol2hdf`."""

# `Literal` solo lo comprueba el analizador de tipos; en ejecución hay que validarlo a mano.
_VALID_OUT_EXTENSIONS: tuple[str, ...] = (".h5", ".hdf", "both")

_DEFAULT_OUTPUT_PARAMETERS: str = (
    "--output-mode=FE --output-variables=rho,u,v,w,p,T,Mach --output-type=vtkhdf"
)

# Ruta de horses2plt relativa a la raíz de la instalación de HORSES3D.
_HORSES2PLT_RELATIVE_PATH: Path = Path("Solver") / "bin" / "horses2plt"

# --- Cabecera binaria de los .hsol ---------------------------------------------------------
# HORSES3D escribe los .hsol con `form="unformatted", access="stream"`
# (`Solver/src/libs/io/SolutionFile.f90`, l. 152-153): bytes seguidos, sin los marcadores de
# registro que Fortran añade a los ficheros secuenciales, en el orden de bytes de la máquina
# (little-endian en x86-64). Los offsets se calculan como en Fortran (l. 88-96), pero
# empezando en 0: el `POS` de Fortran empieza en 1, así que offset = POS - 1.

# Tamaños de los tipos. `SIZEOF_INT` es un `integer` por defecto de gfortran (4 bytes) y
# `SIZEOF_RP` un real `RP = selected_real_kind(14)` (8 bytes, doble precisión)
# (`Solver/src/libs/foundation/SMConstants.f90`, l. 9-20).
_SIZEOF_INT: int = 4
_SIZEOF_RP: int = 8
# Longitud del nombre del fichero guardado al principio (`SOLFILE_STR_LEN`, l. 72), en bytes.
_SOLFILE_STR_LEN: int = 128
# Número de valores de referencia guardados (`NO_OF_SAVED_REFS`, l. 75).
_NO_OF_SAVED_REFS: int = 6

# Offsets en bytes desde el principio del fichero (`POS_*`, l. 88-96).
_HSOL_FILE_TYPE_OFFSET: int = _SOLFILE_STR_LEN  # 128, POS_FILETYPE
_HSOL_NODE_TYPE_OFFSET: int = _HSOL_FILE_TYPE_OFFSET + _SIZEOF_INT  # 132, POS_NODETYPE
_HSOL_NO_OF_ELEMENTS_OFFSET: int = _HSOL_NODE_TYPE_OFFSET + _SIZEOF_INT  # 136
_HSOL_ITERATION_OFFSET: int = _HSOL_NO_OF_ELEMENTS_OFFSET + _SIZEOF_INT  # 140, POS_ITER
_HSOL_TIME_OFFSET: int = _HSOL_ITERATION_OFFSET + _SIZEOF_INT  # 144, POS_TIME
_HSOL_REFERENCES_OFFSET: int = _HSOL_TIME_OFFSET + _SIZEOF_RP  # 152, POS_REFS
_HSOL_TERMINATOR_OFFSET: int = (
    _HSOL_REFERENCES_OFFSET + _NO_OF_SAVED_REFS * _SIZEOF_RP
)  # 200, POS_TERMINATOR
# Tamaño de la cabecera: los datos empiezan justo detrás (`POS_INIT_DATA`).
_HSOL_HEADER_SIZE: int = _HSOL_TERMINATOR_OFFSET + _SIZEOF_INT  # 204

# Formatos de `struct`: `<` fija little-endian y sin relleno; `i` es int32 y `d` float64.
_INT32_FORMAT: str = "<i"
_FLOAT64_FORMAT: str = "<d"
_REFERENCES_FORMAT: str = f"<{_NO_OF_SAVED_REFS}d"

# Valor que HORSES3D escribe al final de la cabecera (`BEGINNING_DATA`, l. 74, escrito en
# l. 183). Si no aparece en su sitio, el fichero no tiene la estructura esperada.
_HSOL_BEGINNING_DATA: int = 88

# Tipos de fichero que horses2plt acepta como solución
# (`Solver/src/addons/horses2tecplot/Storage.f90`, l. 236-279; valores en
# `SolutionFile.f90`, l. 61-70): 2 SOLUTION_FILE, 3 SOLUTION_AND_GRADIENTS_FILE, 4 STATS_FILE,
# 7 ZONE_SOLUTION_FILE, 8 ZONE_SOLUTION_AND_DOT_FILE, 9 SOLUTION_AND_SENSOR_FILE y
# 10 SOLUTION_AND_GRADIENTS_AND_SENSOR_FILE. Quedan fuera las mallas (1 MESH_FILE y
# 6 ZONE_MESH_FILE) y 5 STATS_AND_GRADIENTS_FILE, que horses2plt también rechaza.
_HSOL_SOLUTION_FILE_TYPES: tuple[int, ...] = (2, 3, 4, 7, 8, 9, 10)

# Nombres de los datasets de los valores de referencia, en el orden en que se guardan
# (índices `GAMMA_REF`, `RGAS_REF`, `V_REF`, `RHO_REF`, `T_REF` y `MACH_REF`, l. 77-82):
# gamma, constante del gas R, velocidad, densidad, temperatura y número de Mach.
_REFERENCE_VALUE_NAMES: tuple[str, ...] = (
    "Gamma",
    "GasConstant",
    "ReferenceVelocity",
    "ReferenceDensity",
    "ReferenceTemperature",
    "MachNumber",
)

# Número de líneas finales de la salida de horses2plt que se incluyen en los mensajes de
# error. Basta para ver el motivo del fallo y el `ERROR STOP` con su traza.
_SNIPPET_LINES: int = 20

# Textos con los que horses2plt anuncia un error. Hacen falta porque a veces falla y aun así
# termina con código 0: `Error in file` lo imprime la macro de errores de HORSES3D (con el
# fichero Fortran y la línea), y `ERROR STOP` lo imprime el compilador de Fortran al parar.
_HORSES2PLT_ERROR_MARKERS: tuple[str, ...] = ("Error in file", "ERROR STOP")

# Longitud máxima, en bytes, de una ruta de fichero que HORSES3D lee bien. HORSES3D guarda
# las rutas en variables `character(len=LINE_LENGTH)` con `LINE_LENGTH = 132`
# (`Solver/src/libs/foundation/SMConstants.f90`), así que corta las más largas sin avisar.
# gfortran mide esas variables en bytes, no en caracteres: con caracteres ASCII es lo mismo,
# pero una tilde o una ñ ocupan 2 bytes en UTF-8.
_HORSES_MAX_PATH_LENGTH: int = 132


@dataclass(frozen=True)
class _HsolHeader:
    """Datos de la cabecera de un ``.hsol`` que se guardan en ``/VTKHDF/SimulationInfo``.

    Attributes
    ----------
    file_type : int
        Tipo de fichero de HORSES3D (uno de ``_HSOL_SOLUTION_FILE_TYPES``; p. ej. 2 es una
        solución sin gradientes).
    iteration : int
        Iteración del solver en la que se guardó la solución.
    time : float
        Tiempo físico de la solución, tal y como lo guarda HORSES3D (float64, sin redondear).
    reference_values : dict of str to float
        Valores de referencia y del gas, con las claves de ``_REFERENCE_VALUE_NAMES`` y en
        ese orden: gamma (adimensional), constante del gas R (J/(kg·K)), velocidad (m/s),
        densidad (kg/m³), temperatura (K) y número de Mach (adimensional).
    """

    file_type: int
    iteration: int
    time: float
    reference_values: dict[str, float]


def _read_hsol_header(hsol_file: Path) -> _HsolHeader:
    """Lee el tipo, la iteración, el tiempo y las referencias de la cabecera de un ``.hsol``.

    Solo lee los ``_HSOL_HEADER_SIZE`` (204) primeros bytes; los datos de los elementos, que
    vienen detrás, no se tocan. Los valores se leen como little-endian (el orden de bytes de
    x86-64, donde se ejecuta HORSES3D), con enteros de 4 bytes y reales de 8 bytes.

    Parameters
    ----------
    hsol_file : pathlib.Path
        Fichero ``.hsol`` escrito por HORSES3D.

    Returns
    -------
    _HsolHeader
        Tipo de fichero, iteración, tiempo y los 6 valores de referencia.

    Raises
    ------
    ValueError
        Si el fichero tiene menos de 204 bytes; si en el byte 200 no está el terminador 88
        (fichero corrupto, que no es de HORSES3D o escrito con otro orden de bytes o con
        enteros/reales de otro tamaño); o si el tipo de fichero no es de solución (p. ej. una
        malla).
    OSError
        Si el fichero no se puede abrir (no existe, sin permisos...).
    """
    with hsol_file.open("rb") as stream:
        raw = stream.read(_HSOL_HEADER_SIZE)
    if len(raw) < _HSOL_HEADER_SIZE:
        raise ValueError(
            f"The file {hsol_file} has {len(raw)} bytes, fewer than the {_HSOL_HEADER_SIZE} "
            "bytes of a HORSES3D .hsol header: it is truncated or not a .hsol file."
        )
    # El terminador va primero: si no cuadra, el resto de los campos no significa nada.
    (terminator,) = struct.unpack_from(_INT32_FORMAT, raw, _HSOL_TERMINATOR_OFFSET)
    if terminator != _HSOL_BEGINNING_DATA:
        raise ValueError(
            f"The file {hsol_file} has {terminator} at byte {_HSOL_TERMINATOR_OFFSET} instead "
            f"of the header terminator {_HSOL_BEGINNING_DATA} (BEGINNING_DATA). The file may be "
            "corrupt, not a HORSES3D .hsol file, or written with a different byte order or "
            "integer/real sizes (only little-endian with 4-byte integers and 8-byte reals is "
            "supported)."
        )
    (file_type,) = struct.unpack_from(_INT32_FORMAT, raw, _HSOL_FILE_TYPE_OFFSET)
    if file_type not in _HSOL_SOLUTION_FILE_TYPES:
        accepted = ", ".join(str(value) for value in _HSOL_SOLUTION_FILE_TYPES)
        raise ValueError(
            f"The file {hsol_file} has file type {file_type}, which horses2plt does not accept "
            f"as a solution (accepted types: {accepted}; 1 and 6 are mesh files)."
        )
    (iteration,) = struct.unpack_from(_INT32_FORMAT, raw, _HSOL_ITERATION_OFFSET)
    (time,) = struct.unpack_from(_FLOAT64_FORMAT, raw, _HSOL_TIME_OFFSET)
    references = struct.unpack_from(_REFERENCES_FORMAT, raw, _HSOL_REFERENCES_OFFSET)
    return _HsolHeader(
        file_type=file_type,
        iteration=iteration,
        time=time,
        reference_values=dict(zip(_REFERENCE_VALUE_NAMES, references, strict=True)),
    )


def _resolve_mesh_file(mesh_dir: Path) -> Path:
    """Devuelve el fichero ``.hmesh`` que hay que pasar a ``horses2plt``.

    Parameters
    ----------
    mesh_dir : pathlib.Path
        Ruta a un fichero ``.hmesh`` o a una carpeta que contenga exactamente uno.

    Returns
    -------
    pathlib.Path
        Ruta del fichero de malla.

    Raises
    ------
    FileNotFoundError
        Si la ruta no existe o si la carpeta no contiene ningún ``.hmesh``.
    ValueError
        Si la ruta es un fichero sin extensión ``.hmesh`` o si la carpeta contiene varios
        ``.hmesh`` (elegir uno al azar daría resultados distintos según el sistema).
    """
    if mesh_dir.is_file():
        if mesh_dir.suffix != ".hmesh":
            raise ValueError(f"The mesh file must have the .hmesh extension: {mesh_dir}")
        return mesh_dir
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"The .hmesh path provided does not exist: {mesh_dir}")
    mesh_files = sorted(path for path in mesh_dir.glob("*.hmesh") if path.is_file())
    if not mesh_files:
        raise FileNotFoundError(f"No .hmesh found in {mesh_dir}")
    if len(mesh_files) > 1:
        names = ", ".join(path.name for path in mesh_files)
        raise ValueError(
            f"Several .hmesh files found in {mesh_dir}: {names}. "
            "Pass the path of the .hmesh file to use as MESH_DIR."
        )
    return mesh_files[0]


def _path_length_in_bytes(argument: str) -> int:
    """Mide una ruta como la mide HORSES3D: en bytes.

    Parameters
    ----------
    argument : str
        Ruta tal y como se pasará a ``horses2plt``.

    Returns
    -------
    int
        Número de bytes de la ruta codificada como la codifica el sistema operativo al
        pasarla al proceso (UTF-8 en Linux). Con caracteres ASCII coincide con el número de
        caracteres; una tilde o una ñ cuentan 2.
    """
    # `os.fsencode` es la misma conversión a bytes que hace `subprocess.run` con cada
    # argumento, así que mide exactamente lo que recibe horses2plt.
    return len(os.fsencode(argument))


def _shortest_mesh_argument(mesh_file: Path, hsol_dir: Path) -> str:
    """Elige cómo escribir la ruta de la malla para ``horses2plt``: la más corta posible.

    ``horses2plt`` se ejecuta con ``hsol_dir`` como carpeta de trabajo, así que la malla se
    le puede pasar con su ruta absoluta o con su ruta relativa a ``hsol_dir``. Se devuelve
    la que ocupe menos bytes, para alejarse del límite de ``_HORSES_MAX_PATH_LENGTH`` (132
    bytes; con caracteres ASCII, 132 caracteres) de HORSES3D. Si miden lo mismo, se elige la
    absoluta, que no depende de la carpeta de trabajo.

    Parameters
    ----------
    mesh_file : pathlib.Path
        Fichero ``.hmesh`` (ruta absoluta o relativa a la carpeta actual de Python).
    hsol_dir : pathlib.Path
        Carpeta de los ``.hsol``, que será la carpeta de trabajo de ``horses2plt``.

    Returns
    -------
    str
        Ruta de la malla tal y como se pasará a ``horses2plt``.
    """
    absolute = os.path.abspath(mesh_file)
    relative = os.path.relpath(mesh_file, hsol_dir)
    # `min` devuelve el primero en caso de empate: por eso la absoluta va delante.
    return min(absolute, relative, key=_path_length_in_bytes)


def _check_path_length(argument: str, kind: str) -> None:
    """Comprueba que una ruta que se pasa a ``horses2plt`` cabe en el límite de HORSES3D.

    HORSES3D corta las rutas de más de ``_HORSES_MAX_PATH_LENGTH`` (132) bytes (con
    caracteres ASCII, 132 caracteres; una tilde o una ñ cuentan 2) y no siempre lo avisa:
    con un ``.hsol`` demasiado largo termina con código 0, sin ningún mensaje de error y sin
    escribir la solución. Por eso se rechaza antes de llamarlo.

    Parameters
    ----------
    argument : str
        Ruta tal y como se pasará a ``horses2plt``.
    kind : str
        Tipo de fichero para el mensaje de error (``".hsol"`` o ``".hmesh"``).

    Raises
    ------
    ValueError
        Si ``argument`` ocupa más de ``_HORSES_MAX_PATH_LENGTH`` bytes.
    """
    length_in_bytes = _path_length_in_bytes(argument)
    if length_in_bytes > _HORSES_MAX_PATH_LENGTH:
        raise ValueError(
            f"The {kind} path passed to horses2plt is {length_in_bytes} bytes long "
            f"({len(argument)} characters), above the limit of {_HORSES_MAX_PATH_LENGTH} bytes "
            f"({_HORSES_MAX_PATH_LENGTH} characters if all of them are ASCII; accented letters "
            f"such as 'á' or 'ñ' take 2 bytes each): {argument!r}. This is a limitation of "
            "HORSES3D, which stores file paths in strings of LINE_LENGTH = "
            f"{_HORSES_MAX_PATH_LENGTH} bytes (Solver/src/libs/foundation/SMConstants.f90) and "
            "truncates longer ones. Shorten the file name or move the file closer to HSOL_DIR."
        )


def _run_horses2plt(
    command: list[str], cwd: Path, extended_log: bool
) -> subprocess.CompletedProcess[str]:
    """Ejecuta ``horses2plt`` y comprueba que termina sin errores.

    ``horses2plt`` no siempre avisa de un fallo con el código de salida: en algunos errores
    (p. ej. cuando la ruta de la malla es demasiado larga y no la reconoce) imprime el error
    y termina con código 0. Por eso, además del código, se busca en la salida alguno de los
    textos de ``_HORSES2PLT_ERROR_MARKERS`` (``"Error in file"`` y ``"ERROR STOP"``).

    Parameters
    ----------
    command : list of str
        Programa y argumentos. Se pasan como lista, sin intérprete de comandos (``shell``),
        para que cada argumento llegue intacto a ``horses2plt`` (con espacios incluidos).
    cwd : pathlib.Path
        Carpeta de trabajo del proceso. Las rutas relativas de ``command`` se interpretan
        desde ella, y ``horses2plt`` escribe ahí los ``.hdf`` de los ``.hsol`` que reciba
        solo por su nombre.
    extended_log : bool
        Si es ``True``, imprime la salida completa de ``horses2plt``.

    Returns
    -------
    subprocess.CompletedProcess of str
        Resultado del proceso, con ``stdout`` y ``stderr`` capturados como texto.

    Raises
    ------
    RuntimeError
        Si ``horses2plt`` termina con un código de salida distinto de 0 o si su salida
        (``stdout`` o ``stderr``) contiene ``"Error in file"`` o ``"ERROR STOP"``. El
        mensaje incluye el comando, la carpeta de trabajo y las últimas ``_SNIPPET_LINES``
        líneas de ``stdout`` y ``stderr`` juntas.
    """
    process = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if extended_log:
        print(process.stdout)
        if process.stderr:
            print(process.stderr)
    # Se juntan los dos canales: el motivo del fallo suele salir por stdout, mientras que
    # stderr solo trae el `ERROR STOP` y la traza (las funciones activas al fallar).
    output = "\n".join(text for text in (process.stdout, process.stderr) if text)
    reported_error = any(marker in output for marker in _HORSES2PLT_ERROR_MARKERS)
    if process.returncode != 0 or reported_error:
        snippet = "\n".join(output.splitlines()[-_SNIPPET_LINES:])
        raise RuntimeError(
            f"horses2plt failed (exit code {process.returncode}).\n"
            f"Command: {shlex.join(command)}\n"
            f"Working directory: {cwd}\n"
            f"Output (last {_SNIPPET_LINES} lines of stdout and stderr):\n{snippet}"
        )
    return process


def _write_simulation_info(file_path: Path, header: _HsolHeader) -> None:
    """Escribe el tiempo, la iteración y las referencias en ``/VTKHDF/SimulationInfo``.

    Se guardan como datasets escalares (no como atributos): ``Time`` en float64,
    ``Iteration`` en int64 y un float64 por cada valor de referencia, con los nombres de
    ``_REFERENCE_VALUE_NAMES`` (``Gamma``, ``GasConstant``, ``ReferenceVelocity``,
    ``ReferenceDensity``, ``ReferenceTemperature`` y ``MachNumber``). Si ya existían, se
    sobrescriben.

    Parameters
    ----------
    file_path : pathlib.Path
        Fichero ``.hdf`` generado por ``horses2plt``.
    header : _HsolHeader
        Cabecera del ``.hsol`` del que sale ``file_path``.

    Raises
    ------
    ValueError
        Si el fichero no tiene el grupo ``/VTKHDF`` (no se generó con
        ``--output-type=vtkhdf``).
    """
    with h5py.File(file_path, "r+") as hdf_file:
        vtkhdf_group = hdf_file.get("/VTKHDF")
        if not isinstance(vtkhdf_group, h5py.Group):
            raise ValueError(
                f"The file {file_path} has no /VTKHDF group. "
                "Generate it with the horses2plt option --output-type=vtkhdf."
            )
        info_group = vtkhdf_group.require_group("SimulationInfo")
        datasets: list[tuple[str, float | int, str]] = [
            ("Time", header.time, "float64"),
            ("Iteration", header.iteration, "int64"),
            *((name, value, "float64") for name, value in header.reference_values.items()),
        ]
        for name, value, dtype in datasets:
            # h5py no deja cambiar el tipo de un dataset existente: se borra y se crea de nuevo.
            if name in info_group:
                del info_group[name]
            info_group.create_dataset(name, data=value, dtype=dtype)


def _place_file(source: Path, destination: Path, copy: bool) -> None:
    """Mueve o copia un fichero, salvo que origen y destino sean el mismo.

    Parameters
    ----------
    source : pathlib.Path
        Fichero de origen.
    destination : pathlib.Path
        Ruta completa de destino (incluido el nombre del fichero).
    copy : bool
        ``True`` para copiar (conserva el origen) y ``False`` para mover.
    """
    if source.resolve() == destination.resolve():
        return
    if copy:
        shutil.copy2(source, destination)
    else:
        # `shutil.move` también funciona si origen y destino están en discos distintos.
        shutil.move(source, destination)


class HorsesDataHandler:
    """Herramientas para convertir e inspeccionar las soluciones de HORSES3D.

    La clase no guarda estado: agrupa las operaciones de posprocesado para que se usen
    desde un único objeto, ``HorsesDataHandler()``.
    """

    def __init__(self) -> None:
        """Crea el objeto. No necesita argumentos ni imprime nada."""

    def convert_hsol2hdf(
        self,
        SOLVER_PATH: PathLike,
        HSOL_DIR: PathLike,
        MESH_DIR: PathLike,
        OUT_FILES_DIR: PathLike | None = None,
        OUTPUT_PARAMETERS: str | None = None,
        OUT_EXTENSION: OutExtension = ".h5",
        LOG: bool = True,
        WRITE_SIM_PARAMETER: bool = True,
        EXTENDED_LOG: bool = False,
    ) -> None:
        """Convierte los binarios ``.hsol`` a HDF5 (``.h5``/``.hdf``) con ``horses2plt``.

        Antes de crear nada en disco se validan todos los argumentos y, si
        ``WRITE_SIM_PARAMETER`` es ``True``, las cabeceras de todos los ``.hsol``. Después se
        ejecuta ``horses2plt`` una sola vez con todos los ``.hsol``, se escriben (si se pide)
        el tiempo, la iteración y los valores de referencia en cada fichero y se colocan los
        resultados en ``OUT_FILES_DIR``. Solo se tocan los ``.hdf``
        generados en esta ejecución; otros ``.hdf`` que ya hubiera en ``HSOL_DIR`` se
        dejan donde están.

        Límite de 132 bytes: HORSES3D corta las rutas de fichero de más de 132 bytes
        (``LINE_LENGTH`` en ``Solver/src/libs/foundation/SMConstants.f90``). Con caracteres
        ASCII son 132 caracteres; una tilde o una ñ ocupan 2 bytes cada una.
        Para alejarse de ese límite, ``horses2plt`` se ejecuta con ``HSOL_DIR`` como carpeta
        de trabajo: cada ``.hsol`` se le pasa solo por su nombre y la malla con la más corta
        (en bytes) entre su ruta absoluta y su ruta relativa a ``HSOL_DIR``. Así, el límite
        solo afecta al nombre de cada ``.hsol`` y a esa ruta de la malla; si alguno lo supera,
        se lanza ``ValueError`` antes de crear carpetas o llamar a ``horses2plt``. Las rutas
        pueden contener espacios.

        Parameters
        ----------
        SOLVER_PATH : str or os.PathLike
            Raíz de una instalación de HORSES3D compilada, al menos, con soporte HDF5. Debe
            contener ``Solver/bin/horses2plt``, que se llama con su ruta absoluta (sin
            límite de longitud: HORSES3D no la lee).
        HSOL_DIR : str or os.PathLike
            Carpeta con los ficheros ``.hsol``. Se procesan todos, ordenados por nombre.
            Es la carpeta de trabajo de ``horses2plt``. El nombre de cada ``.hsol``
            (extensión incluida) no puede pasar de 132 bytes (con caracteres ASCII, 132
            caracteres); la ruta de la carpeta no tiene límite.
        MESH_DIR : str or os.PathLike
            Fichero ``.hmesh`` o carpeta que contenga exactamente uno. La más corta (en bytes)
            entre la ruta absoluta del ``.hmesh`` y su ruta relativa a ``HSOL_DIR`` no puede pasar
            de 132 bytes (con caracteres ASCII, 132 caracteres).
        OUT_FILES_DIR : str or os.PathLike or None, optional
            Carpeta donde se guardan los resultados; se crea si no existe. Si es ``None``
            se usa ``HSOL_DIR``. Por defecto ``None``.
        OUTPUT_PARAMETERS : str or None, optional
            Opciones de ``horses2plt`` en una sola cadena, troceada como lo haría un
            terminal. Si es ``None`` se usa
            ``"--output-mode=FE --output-variables=rho,u,v,w,p,T,Mach --output-type=vtkhdf"``.
            Por defecto ``None``.
        OUT_EXTENSION : {".h5", ".hdf", "both"}, optional
            Extensión de salida. Con ``"both"`` se guarda una copia ``.hdf`` en
            ``OUT_FILES_DIR/hdf`` y otra ``.h5`` en ``OUT_FILES_DIR/h5``. Por defecto
            ``".h5"``.
        LOG : bool, optional
            Si es ``True``, la librería imprime mensajes de progreso (``>>> ...``). Por
            defecto ``True``.
        WRITE_SIM_PARAMETER : bool, optional
            Si es ``True``, lee la cabecera binaria de cada ``.hsol`` y escribe en
            ``/VTKHDF/SimulationInfo`` de su fichero, como datasets escalares: ``Time``
            (tiempo físico, float64), ``Iteration`` (int64) y los valores de referencia en
            float64 ``Gamma`` (adimensional), ``GasConstant`` (J/(kg·K)),
            ``ReferenceVelocity`` (m/s), ``ReferenceDensity`` (kg/m³),
            ``ReferenceTemperature`` (K) y ``MachNumber`` (adimensional). Los valores se
            guardan con la precisión completa con que los escribe HORSES3D. Por defecto
            ``True``.
        EXTENDED_LOG : bool, optional
            Si es ``True``, imprime la salida completa de ``horses2plt``. Por defecto
            ``False``.

        Raises
        ------
        ValueError
            Si ``OUT_EXTENSION`` no es una de las tres opciones, si ``MESH_DIR`` contiene
            varios ``.hmesh``, si el nombre de algún ``.hsol`` o la ruta que se pasa de la
            malla supera los 132 bytes (antes de crear carpetas o llamar a
            ``horses2plt``), si con ``WRITE_SIM_PARAMETER=True`` la cabecera de algún
            ``.hsol`` no es válida (fichero de menos de 204 bytes, sin el terminador 88 en el
            byte 200 o de un tipo que no es de solución, como una malla; también antes de
            crear carpetas o llamar a ``horses2plt``) o si un ``.hdf`` generado no tiene el
            grupo ``/VTKHDF``.
        FileNotFoundError
            Si no existe ``SOLVER_PATH``, ``HSOL_DIR``, ``horses2plt`` o la malla, si no hay
            ``.hsol`` o si falta algún ``.hdf`` esperado tras la conversión.
        RuntimeError
            Si ``horses2plt`` termina con un código de salida distinto de 0 o si su salida
            contiene ``"Error in file"`` o ``"ERROR STOP"`` (a veces falla devolviendo
            código 0). El mensaje incluye el comando, la carpeta de trabajo y el final de la
            salida.
        """
        ## 1. Validación completa, antes de crear nada en disco.
        if OUT_EXTENSION not in _VALID_OUT_EXTENSIONS:
            raise ValueError(
                f"The output extension must be either '.h5', '.hdf' or 'both', "
                f"not {OUT_EXTENSION!r}."
            )
        solver_path = Path(SOLVER_PATH)
        hsol_dir = Path(HSOL_DIR)
        if not solver_path.is_dir():
            raise FileNotFoundError(f"The solver path provided does not exist: {solver_path}")
        if not hsol_dir.is_dir():
            raise FileNotFoundError(f"The .hsol path provided does not exist: {hsol_dir}")

        # Ruta absoluta: horses2plt se ejecuta desde HSOL_DIR, y una ruta relativa a la
        # carpeta actual de Python dejaría de ser válida allí.
        horses2plt_path = (solver_path / _HORSES2PLT_RELATIVE_PATH).absolute()
        if not horses2plt_path.is_file():
            raise FileNotFoundError(f"The horses2plt utility is not found: {horses2plt_path}")

        # Orden por nombre para que el resultado no dependa del sistema de ficheros.
        hsol_files = sorted(path for path in hsol_dir.glob("*.hsol") if path.is_file())
        if not hsol_files:
            raise FileNotFoundError(f"No .hsol found in {hsol_dir}")

        mesh_file = _resolve_mesh_file(Path(MESH_DIR))

        # HORSES3D corta las rutas de más de 132 bytes (ver `_HORSES_MAX_PATH_LENGTH`).
        # Para que el límite afecte lo menos posible, horses2plt se ejecuta desde HSOL_DIR:
        # cada .hsol se pasa por su nombre y la malla con su ruta más corta en bytes. Al recibir el
        # .hsol por su nombre, horses2plt escribe el .hdf en la carpeta de trabajo, HSOL_DIR.
        hsol_arguments = [hsol.name for hsol in hsol_files]
        mesh_argument = _shortest_mesh_argument(mesh_file, hsol_dir)
        # Se rechazan aquí, antes de crear carpetas, porque con un .hsol demasiado largo
        # horses2plt termina con código 0 sin avisar de nada.
        for hsol_argument in hsol_arguments:
            _check_path_length(hsol_argument, ".hsol")
        _check_path_length(mesh_argument, ".hmesh")

        # Las cabeceras se leen y validan todas aquí, antes de crear carpetas o llamar a
        # horses2plt: un .hsol corrupto o una malla entre las soluciones se detecta sin dejar
        # una conversión a medias. Si no se van a escribir, no hace falta leerlas.
        headers: list[_HsolHeader] = []
        if WRITE_SIM_PARAMETER:
            headers = [_read_hsol_header(hsol) for hsol in hsol_files]

        out_dir = hsol_dir if OUT_FILES_DIR is None else Path(OUT_FILES_DIR)
        if OUTPUT_PARAMETERS is None:
            OUTPUT_PARAMETERS = _DEFAULT_OUTPUT_PARAMETERS
        # Trocea la cadena como un terminal (respeta comillas) para pasarla como lista.
        output_parameters = shlex.split(OUTPUT_PARAMETERS)

        if LOG:
            print(">>> Nº of .hsol files:", len(hsol_files))
            print(">>> Mesh file:", mesh_file)

        ## 2. Carpetas de salida (solo tras validar todo).
        if OUT_EXTENSION == "both":
            (out_dir / "hdf").mkdir(parents=True, exist_ok=True)
            (out_dir / "h5").mkdir(parents=True, exist_ok=True)
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

        ## 3. Conversión con horses2plt.
        # horses2plt escribe `<stem>.hdf` en la ruta del `.hsol` tal y como la recibe; como
        # recibe solo el nombre, eso es la carpeta de trabajo, HSOL_DIR.
        expected_hdf_files = [hsol.with_suffix(".hdf") for hsol in hsol_files]

        # Una sola llamada con todos los `.hsol`; equivale al antiguo `*.hsol` del shell.
        command = [
            str(horses2plt_path),
            *hsol_arguments,
            mesh_argument,
            *output_parameters,
        ]
        _run_horses2plt(command, hsol_dir, EXTENDED_LOG)
        if LOG:
            print(f">>> Converted {len(hsol_files)} .hsol files")

        missing = [hdf for hdf in expected_hdf_files if not hdf.is_file()]
        if missing:
            names = ", ".join(str(hdf) for hdf in missing)
            raise FileNotFoundError(f"horses2plt did not produce the expected files: {names}")

        if WRITE_SIM_PARAMETER:
            # Las tres listas están en el mismo orden: cada .hdf recibe la cabecera de su .hsol.
            for hsol, hdf, header in zip(hsol_files, expected_hdf_files, headers, strict=True):
                _write_simulation_info(hdf, header)
                if LOG:
                    print(
                        f">>> {hsol.name}: time {header.time}, iteration {header.iteration} "
                        f"written in {hdf.name}"
                    )

        ## 4. Colocación de los resultados.
        for hdf in expected_hdf_files:
            if OUT_EXTENSION == "both":
                _place_file(hdf, out_dir / "hdf" / f"{hdf.stem}.hdf", copy=True)
                _place_file(hdf, out_dir / "h5" / f"{hdf.stem}.h5", copy=False)
            else:
                _place_file(hdf, out_dir / f"{hdf.stem}{OUT_EXTENSION}", copy=False)
        if LOG:
            print(f">>> Output files saved in {out_dir}")

    def get_sizes(self, PATH: PathLike, EXTENSION: str = ".hdf") -> int:
        """Calcula el tamaño total de los ficheros de una extensión en una carpeta.

        Imprime el total en MB (por debajo de 1 GB) o en GB, con prefijos decimales
        (1 MB = 10^6 bytes).

        Parameters
        ----------
        PATH : str or os.PathLike
            Carpeta que se examina (sin entrar en subcarpetas).
        EXTENSION : str, optional
            Terminación de los nombres de fichero que se cuentan. Por defecto ``".hdf"``.

        Returns
        -------
        int
            Suma de los tamaños, en bytes.
        """
        folder = Path(PATH)
        size = sum(
            path.stat().st_size
            for path in folder.iterdir()
            if path.is_file() and path.name.endswith(EXTENSION)
        )
        if size < 1e9:
            print(">>> Data size:", size / 1e6, "MB")
        else:
            print(">>> Data size:", size / 1e9, "GB")
        return size

    def explore_hdf5_structure(self, FILEPATH: PathLike) -> None:
        """Imprime los grupos y datasets de un fichero HDF5.

        Para cada dataset muestra su forma (``shape``) y su tipo de dato; los niveles de
        anidamiento se indican con sangría.

        Parameters
        ----------
        FILEPATH : str or os.PathLike
            Ruta del fichero ``.hdf``/``.h5`` que se inspecciona.
        """

        def print_structure(name: str, obj: object) -> None:
            # `name` es la ruta interna (p. ej. "VTKHDF/PointData"): una sangría por nivel.
            indent = "     " * name.count("/")
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: Dataset {obj.shape} {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}{name}: Group")

        print(f"File structure: {FILEPATH}")
        print("-" * 50)
        with h5py.File(FILEPATH, "r") as hdf_file:
            hdf_file.visititems(print_structure)
        print("-" * 50)
