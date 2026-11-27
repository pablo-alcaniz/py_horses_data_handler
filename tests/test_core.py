"""Tests de ``py_horses_data_handler.core``.

No necesitan HORSES3D: ``subprocess.run`` del módulo ``core`` se sustituye por un falso
(`FakeHorses2plt`) que crea, con h5py, el ``.hdf`` que escribiría ``horses2plt`` junto a
cada ``.hsol`` (interpretando las rutas relativas desde la carpeta de trabajo ``cwd``, como
el proceso real). Los ``.hsol`` de los tests tienen una cabecera binaria válida, construida
con `SyntheticHeader`, de la que salen ``Time``, ``Iteration`` y los valores de referencia.
Todos los ficheros se crean en carpetas temporales (``tmp_path``).

Datos reales, para probar con el formato verdadero y no con uno inventado:

- ``tests/data/Cylinder_Re40_0000000030_header.hsol``: los 204 primeros bytes (la cabecera)
  de un ``.hsol`` escrito por HORSES3D (cilindro a Re = 40, iteración 30).
- `REAL_STDOUT`, `REAL_FAILURE_STDOUT` y `REAL_FAILURE_STDERR`: extractos copiados de
  ejecuciones reales de ``horses2plt``, para probar la detección de errores.
"""

import os
import shutil
import struct
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, Unpack, cast

import h5py
import numpy as np
import pytest

from py_horses_data_handler import HorsesDataHandler, __version__, core
from py_horses_data_handler.core import OutExtension, _HsolHeader, _read_hsol_header

# Cabecera real: los 204 primeros bytes de `RESULTS/Cylinder_Re40_0000000030.hsol`.
REAL_HEADER_FILE = Path(__file__).parent / "data" / "Cylinder_Re40_0000000030_header.hsol"

# Valores de referencia de esa cabecera, leídos con un volcado de bytes del fichero real.
REAL_REFERENCE_VALUES = {
    "Gamma": 1.4,
    "GasConstant": 287.15,
    "ReferenceVelocity": 68.15752995002745,
    "ReferenceDensity": 1.2214535421047699,
    "ReferenceTemperature": 288.8888888888889,
    "MachNumber": 0.2,
}

# Extracto fiel, sin modificar, de la salida de `horses2plt` al convertir una solución
# (código 0) de la iteración 0. El tiempo sale con 4 cifras (`ES10.3`), por eso ya no se lee
# de aquí sino de la cabecera del `.hsol`.
REAL_STDOUT = """\
          Loading Mesh File:
          -----------------
               Mesh file "manual/Cylinder_Re40_0000000000.hmesh":
               --------------------------------------------------
                              ->          Number of elements: 2184
                              ->        Discretization nodes: Gauss
          Loading Solution File:
          ---------------------
                              ->               Solution File: manual/Cylinder_Re40_0000000000.hsol


               Solution file "manual/Cylinder_Re40_0000000000.hsol":
               -----------------------------------------------------
                              ->      Solution file contains gradients: no
                              ->                   Iteration: 0
                              ->                        Time:  0.000E+00
                              ->          Reference velocity:  68.158
                              ->           Reference density:   1.221
                              ->       Reference Temperature: 288.889
                              ->       Reference Mach number:   0.200
"""

# Salida del falso por defecto. Sus `Time` e `Iteration` (0) no coinciden con los de las
# cabeceras sintéticas: así se comprueba que los valores escritos salen de la cabecera.
FAKE_STDOUT = REAL_STDOUT

# Cabecera con la que empieza toda salida de horses2plt (recuadro de 100 columnas).
_BANNER_BORDER = "#" * 100
_BANNER_EMPTY = "#" + " " * 98 + "#"
_BANNER_TITLE = "#" + " " * 30 + "HORSES to TecPlot conversion utility" + " " * 32 + "#"
_BANNER = "\n".join(
    [
        _BANNER_BORDER,
        _BANNER_EMPTY,
        _BANNER_EMPTY,
        _BANNER_TITLE,
        _BANNER_EMPTY,
        _BANNER_EMPTY,
        _BANNER_BORDER,
        "Compiled Aug 31 2026, 21:18:38",
        "",
    ]
)

# Fallo real (código 1) de un horses2plt compilado sin HDF5: el motivo sale por stdout y
# stderr solo trae `ERROR STOP` y la traza. Se escribe con `\n` explícitos porque la
# primera línea de stderr acaba en un espacio que hay que conservar.
REAL_FAILURE_STDOUT = _BANNER + "HDF5 support must be enabled to save to VTKHDF format\n"
REAL_FAILURE_STDERR = (
    "ERROR STOP \n"
    "\n"
    "Error termination. Backtrace:\n"
    "#0  0x7fa45fa27602 in ???\n"
    "#1  0x7fa45fa2848a in ???\n"
    "#2  0x558e459db50b in MAIN__\n"
    "#3  0x558e4597a49e in main\n"
)

# Fallo que horses2plt devuelve con código 0 cuando no reconoce la malla (p. ej. porque su
# ruta pasa de 132 caracteres y HORSES3D la corta). Mensajes tal y como los imprime
# getTask.f90.
MESH_NOT_SPECIFIED_STDOUT = (
    _BANNER + "Mesh file was not specified\nError in file getTask.f90, in line 434.\n"
)


# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------


@dataclass
class FakeHorses2plt:
    """Sustituto de ``subprocess.run`` que imita a ``horses2plt``.

    Guarda cada llamada (argumentos y opciones) y cada ``.hdf`` que crea, para que los
    tests los inspeccionen. Como el ``horses2plt`` real, escribe el ``.hdf`` en la ruta del
    ``.hsol`` tal y como la recibe; si es relativa, desde la carpeta de trabajo ``cwd``.
    """

    returncode: int = 0
    stdout: str = FAKE_STDOUT
    stderr: str = ""
    with_vtkhdf: bool = True
    calls: list[tuple[list[str], dict[str, object]]] = field(default_factory=list)
    created: list[Path] = field(default_factory=list)

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, kwargs))
        cwd = kwargs["cwd"]
        assert isinstance(cwd, Path)
        if self.returncode == 0:
            for arg in args:
                if arg.endswith(".hsol"):
                    # Si `arg` es absoluta, `cwd / arg` es `arg`: el operador `/` lo resuelve.
                    hdf_path = cwd / Path(arg).with_suffix(".hdf")
                    with h5py.File(hdf_path, "w") as hdf_file:
                        if self.with_vtkhdf:
                            hdf_file.create_group("VTKHDF")
                    self.created.append(hdf_path)
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


@dataclass(frozen=True)
class SyntheticHeader:
    """Cabecera de un ``.hsol`` con el formato binario de HORSES3D, para los tests.

    Los valores por defecto son una solución válida (tipo 2) con un tiempo de muchas cifras
    y referencias distintas de las reales, para que un error de offset se note.
    """

    name: str = "RESULTS/synthetic.hsol"
    file_type: int = 2
    node_type: int = 1
    n_elements: int = 8
    iteration: int = 12345
    time: float = 1.2345678901234567e-2
    references: tuple[float, ...] = (1.3, 296.8, 101.25, 0.987654321, 300.5, 0.31)
    terminator: int = 88

    def to_bytes(self, byte_order: str = "<") -> bytes:
        """Empaqueta la cabecera en los 204 bytes que escribe HORSES3D.

        Formato: nombre (128 bytes rellenados con ``\\0``; `struct` lo corta si es más
        largo), 4 int32 (tipo, nodos, elementos, iteración), 1 float64 (tiempo), 6 float64
        (referencias) y el int32 terminador. ``byte_order`` es ``"<"`` (little-endian) o
        ``">"`` (big-endian, para probar que se detecta).
        """
        return struct.pack(
            f"{byte_order}128s4id6di",
            self.name.encode("utf-8"),
            self.file_type,
            self.node_type,
            self.n_elements,
            self.iteration,
            self.time,
            *self.references,
            self.terminator,
        )

    @property
    def reference_values(self) -> dict[str, float]:
        """Referencias con los nombres de los datasets de ``SimulationInfo``."""
        return dict(zip(core._REFERENCE_VALUE_NAMES, self.references, strict=True))


def write_hsol(path: Path, header: SyntheticHeader) -> None:
    """Escribe un ``.hsol`` con ``header`` y unos bytes de datos detrás (que no se leen)."""
    path.write_bytes(header.to_bytes() + b"\xff" * 64)


@dataclass
class Case:
    """Rutas de un caso sintético (instalación, soluciones, malla y salida) y cabeceras.

    ``headers`` guarda la cabecera escrita en cada ``.hsol``, por nombre sin extensión.
    """

    solver: Path
    horses2plt: Path
    hsol_dir: Path
    hsol_files: list[Path]
    mesh: Path
    out: Path
    headers: dict[str, SyntheticHeader]


def make_case(
    root: Path,
    stems: tuple[str, ...] = ("sol_0002", "sol_0001"),
    *,
    solver_name: str = "horses3d",
    hsol_dir_name: str = "results",
    mesh_dir_name: str = "mesh",
    out_dir_name: str = "out",
) -> Case:
    """Crea en ``root`` una instalación falsa de HORSES3D, varios ``.hsol`` y una malla.

    Cada ``.hsol`` tiene una cabecera válida con iteración, tiempo y referencias propios
    (dependen de su posición en ``stems``), para comprobar que cada ``.hdf`` recibe los de
    su ``.hsol``. Los nombres de las carpetas se pueden cambiar (p. ej. para probar rutas
    con espacios).
    """
    solver = root / solver_name
    horses2plt = solver / "Solver" / "bin" / "horses2plt"
    horses2plt.parent.mkdir(parents=True)
    horses2plt.touch()
    hsol_dir = root / hsol_dir_name
    hsol_dir.mkdir(parents=True)
    hsol_files = []
    headers = {}
    for index, stem in enumerate(stems, start=1):
        hsol = hsol_dir / f"{stem}.hsol"
        header = SyntheticHeader(
            name=f"RESULTS/{stem}.hsol",
            iteration=1000 * index + 7,
            time=0.04032508838441574 * index,
            references=(1.4, 287.15, 68.15752995002745 * index, 1.2214535421047699, 288.9, 0.2),
        )
        write_hsol(hsol, header)
        hsol_files.append(hsol)
        headers[stem] = header
    mesh_dir = root / mesh_dir_name
    mesh_dir.mkdir()
    mesh = mesh_dir / "cylinder.hmesh"
    mesh.write_bytes(b"\x00")
    return Case(solver, horses2plt, hsol_dir, hsol_files, mesh_dir, root / out_dir_name, headers)


def assert_simulation_info(hdf_path: Path, header: SyntheticHeader) -> None:
    """Comprueba que ``SimulationInfo`` tiene los valores exactos de ``header`` y su tipo."""
    with h5py.File(hdf_path, "r") as hdf_file:
        info = hdf_file["VTKHDF/SimulationInfo"]
        assert isinstance(info, h5py.Group)
        expected: dict[str, tuple[float | int, type[np.generic]]] = {
            "Time": (header.time, np.float64),
            "Iteration": (header.iteration, np.int64),
            **{name: (value, np.float64) for name, value in header.reference_values.items()},
        }
        assert set(info.keys()) == set(expected)
        for name, (value, dtype) in expected.items():
            dataset = info[name]
            assert isinstance(dataset, h5py.Dataset)
            assert dataset.shape == ()
            assert dataset.dtype == dtype
            # Igualdad exacta: el valor se copia de la cabecera sin pasar por texto.
            assert dataset[()] == value


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> FakeHorses2plt:
    """Sustituye ``core.subprocess.run`` por un `FakeHorses2plt` durante el test."""
    fake = FakeHorses2plt()
    monkeypatch.setattr(core.subprocess, "run", fake)
    return fake


class ConvertOptions(TypedDict, total=False):
    """Opciones de ``convert_hsol2hdf`` que los tests pueden variar."""

    OUTPUT_PARAMETERS: str | None
    OUT_EXTENSION: OutExtension
    LOG: bool
    WRITE_SIM_PARAMETER: bool
    EXTENDED_LOG: bool


def convert(case: Case, **options: Unpack[ConvertOptions]) -> None:
    """Llama a ``convert_hsol2hdf`` con las rutas del caso (como ``str``) y las opciones."""
    HorsesDataHandler().convert_hsol2hdf(
        str(case.solver), str(case.hsol_dir), str(case.mesh), str(case.out), **options
    )


# --------------------------------------------------------------------------------------
# Lectura de la cabecera de los .hsol
# --------------------------------------------------------------------------------------


def test_header_offsets_match_fortran() -> None:
    # Los offsets calculados como en SolutionFile.f90 coinciden con la tabla de la cabecera.
    assert core._HSOL_FILE_TYPE_OFFSET == 128
    assert core._HSOL_ITERATION_OFFSET == 140
    assert core._HSOL_TIME_OFFSET == 144
    assert core._HSOL_REFERENCES_OFFSET == 152
    assert core._HSOL_TERMINATOR_OFFSET == 200
    assert core._HSOL_HEADER_SIZE == 204
    assert len(SyntheticHeader().to_bytes()) == 204


def test_read_real_header() -> None:
    # Cabecera de un .hsol real: horses2plt imprime `Time: 4.033E-02`, pero en el fichero
    # está el valor completo. Se compara con `==`, sin tolerancia.
    header = _read_hsol_header(REAL_HEADER_FILE)
    assert header == _HsolHeader(
        file_type=2,
        iteration=30,
        time=0.04032508838441574,
        reference_values=REAL_REFERENCE_VALUES,
    )
    assert list(header.reference_values) == list(core._REFERENCE_VALUE_NAMES)
    assert type(header.iteration) is int
    assert type(header.time) is float


@pytest.mark.parametrize("file_type", [2, 3, 4, 7, 8, 9, 10])
def test_read_synthetic_header_solution_types(tmp_path: Path, file_type: int) -> None:
    # Todos los tipos de solución que acepta horses2plt, con un tiempo de 17 cifras, la
    # mayor iteración que cabe en un int32 y datos detrás de la cabecera.
    synthetic = SyntheticHeader(
        name="RESULTS/a_long_case_name_0123456789.hsol",
        file_type=file_type,
        iteration=2**31 - 1,
        time=1.2345678901234567e-5,
    )
    hsol = tmp_path / "case.hsol"
    write_hsol(hsol, synthetic)
    header = _read_hsol_header(hsol)
    assert header.file_type == file_type
    assert header.iteration == 2**31 - 1
    assert header.time == 1.2345678901234567e-5
    assert header.reference_values == synthetic.reference_values


def test_read_header_exactly_204_bytes(tmp_path: Path) -> None:
    # Un fichero con solo la cabecera (sin datos) también se lee.
    hsol = tmp_path / "case.hsol"
    hsol.write_bytes(SyntheticHeader(iteration=0, time=0.0).to_bytes())
    header = _read_hsol_header(hsol)
    assert header.iteration == 0
    assert header.time == 0.0


@pytest.mark.parametrize("size", [0, 1, 203])
def test_read_header_truncated(tmp_path: Path, size: int) -> None:
    hsol = tmp_path / "short.hsol"
    hsol.write_bytes(SyntheticHeader().to_bytes()[:size])
    with pytest.raises(ValueError, match=f"has {size} bytes, fewer than the 204") as info:
        _read_hsol_header(hsol)
    assert str(hsol) in str(info.value)


def test_read_header_wrong_terminator(tmp_path: Path) -> None:
    hsol = tmp_path / "bad.hsol"
    write_hsol(hsol, SyntheticHeader(terminator=99))
    with pytest.raises(ValueError, match="has 99 at byte 200 instead") as info:
        _read_hsol_header(hsol)
    message = str(info.value)
    assert str(hsol) in message
    assert "corrupt" in message
    assert "byte order" in message


def test_read_header_big_endian_detected(tmp_path: Path) -> None:
    # Un .hsol escrito en big-endian: el terminador 88 se lee como otro número.
    hsol = tmp_path / "big_endian.hsol"
    hsol.write_bytes(SyntheticHeader().to_bytes(byte_order=">"))
    with pytest.raises(ValueError, match="terminator 88"):
        _read_hsol_header(hsol)


@pytest.mark.parametrize("file_type", [1, 5, 6])
def test_read_header_not_a_solution(tmp_path: Path, file_type: int) -> None:
    # 1 y 6 son mallas; 5 (estadísticas con gradientes) tampoco lo acepta horses2plt.
    hsol = tmp_path / "mesh.hsol"
    write_hsol(hsol, SyntheticHeader(file_type=file_type))
    with pytest.raises(ValueError, match=f"has file type {file_type}") as info:
        _read_hsol_header(hsol)
    assert str(hsol) in str(info.value)


def test_write_simulation_info_overwrites(tmp_path: Path) -> None:
    # Si SimulationInfo ya tiene datasets (de otro tipo), se sustituyen por los nuevos.
    hdf = tmp_path / "sol.hdf"
    with h5py.File(hdf, "w") as hdf_file:
        info = hdf_file.create_group("VTKHDF/SimulationInfo")
        info.create_dataset("Time", data=-1, dtype="int32")
        info.create_dataset("Gamma", data=0.0, dtype="float32")
    header = _read_hsol_header(REAL_HEADER_FILE)
    core._write_simulation_info(hdf, header)
    synthetic = SyntheticHeader(
        iteration=30,
        time=0.04032508838441574,
        references=tuple(REAL_REFERENCE_VALUES.values()),
    )
    assert_simulation_info(hdf, synthetic)


# --------------------------------------------------------------------------------------
# Conversión (criterios 3 a 7)
# --------------------------------------------------------------------------------------


def test_convert_h5(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    convert(case, OUT_EXTENSION=".h5")
    for stem in ("sol_0001", "sol_0002"):
        out_file = case.out / f"{stem}.h5"
        assert out_file.is_file()
        assert not (case.hsol_dir / f"{stem}.hdf").exists()
        # Cada fichero tiene los valores de la cabecera de su .hsol, no los de stdout.
        assert_simulation_info(out_file, case.headers[stem])


def test_simulation_info_from_real_header(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    # Con la cabecera real, el tiempo escrito es el completo (horses2plt imprime 4.033E-02).
    case = make_case(tmp_path, stems=())
    hsol = case.hsol_dir / "Cylinder_Re40_0000000030.hsol"
    shutil.copyfile(REAL_HEADER_FILE, hsol)
    fake_run.stdout = REAL_STDOUT.replace("Iteration: 0", "Iteration: 30").replace(
        "Time:  0.000E+00", "Time:  4.033E-02"
    )
    convert(case, LOG=False)
    with h5py.File(case.out / "Cylinder_Re40_0000000030.h5", "r") as hdf_file:
        assert hdf_file["VTKHDF/SimulationInfo/Time"][()] == 0.04032508838441574
        assert hdf_file["VTKHDF/SimulationInfo/Iteration"][()] == 30
        for name, value in REAL_REFERENCE_VALUES.items():
            assert hdf_file[f"VTKHDF/SimulationInfo/{name}"][()] == value


def test_convert_hdf_in_place(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    # Sin OUT_FILES_DIR y con ".hdf", origen y destino coinciden: el fichero se queda.
    case = make_case(tmp_path, stems=("sol",))
    HorsesDataHandler().convert_hsol2hdf(
        case.solver, case.hsol_dir, case.mesh, OUT_EXTENSION=".hdf", LOG=False
    )
    assert (case.hsol_dir / "sol.hdf").is_file()


def test_convert_both(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    convert(case, OUT_EXTENSION="both")
    for stem in ("sol_0001", "sol_0002"):
        assert (case.out / "hdf" / f"{stem}.hdf").is_file()
        assert (case.out / "h5" / f"{stem}.h5").is_file()
        assert not (case.hsol_dir / f"{stem}.hdf").exists()


@pytest.mark.parametrize("write_sim_parameter", [True, False])
def test_paths_with_spaces_full_conversion(
    tmp_path: Path, fake_run: FakeHorses2plt, write_sim_parameter: bool
) -> None:
    # Criterios A y B: espacios en SOLVER_PATH, HSOL_DIR, MESH_DIR, OUT_FILES_DIR y en el
    # nombre del .hsol. horses2plt se ejecuta desde HSOL_DIR y recibe solo el nombre.
    case = make_case(
        tmp_path / "my case",
        stems=("my sol",),
        solver_name="horses 3d",
        hsol_dir_name="hsol results",
        mesh_dir_name="mesh files",
        out_dir_name="out files",
    )
    convert(
        case,
        OUTPUT_PARAMETERS="--output-mode=FE --output-type=vtkhdf",
        WRITE_SIM_PARAMETER=write_sim_parameter,
        LOG=False,
    )
    assert len(fake_run.calls) == 1
    args, kwargs = fake_run.calls[0]
    assert isinstance(args, list)
    assert not kwargs.get("shell", False)
    assert kwargs["cwd"] == case.hsol_dir
    assert args == [
        str(case.horses2plt),
        "my sol.hsol",
        os.path.join("..", "mesh files", "cylinder.hmesh"),
        "--output-mode=FE",
        "--output-type=vtkhdf",
    ]
    # El .hdf se escribe primero en HSOL_DIR y después se mueve a OUT_FILES_DIR.
    assert fake_run.created == [case.hsol_dir / "my sol.hdf"]
    assert not (case.hsol_dir / "my sol.hdf").exists()
    assert (case.out / "my sol.h5").is_file()


def test_relative_paths(
    tmp_path: Path, fake_run: FakeHorses2plt, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Con rutas relativas, el ejecutable se pasa con ruta absoluta (horses2plt se ejecuta
    # desde HSOL_DIR, donde la relativa no valdría) y la malla relativa a HSOL_DIR.
    make_case(tmp_path, stems=("sol",))
    monkeypatch.chdir(tmp_path)
    HorsesDataHandler().convert_hsol2hdf("horses3d", "results", "mesh", "out", LOG=False)
    args, kwargs = fake_run.calls[0]
    assert args[0] == str(Path.cwd() / "horses3d" / "Solver" / "bin" / "horses2plt")
    assert args[1:3] == ["sol.hsol", os.path.join("..", "mesh", "cylinder.hmesh")]
    assert kwargs["cwd"] == Path("results")
    assert (tmp_path / "out" / "sol.h5").is_file()
    assert not (tmp_path / "results" / "sol.hdf").exists()


def test_default_output_parameters(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path, stems=("sol",))
    convert(case, LOG=False)
    args, _ = fake_run.calls[0]
    assert args[3:] == [
        "--output-mode=FE",
        "--output-variables=rho,u,v,w,p,T,Mach",
        "--output-type=vtkhdf",
    ]


@pytest.mark.parametrize("write_sim_parameter", [True, False])
def test_single_call_with_all_hsol_sorted(
    tmp_path: Path, fake_run: FakeHorses2plt, write_sim_parameter: bool
) -> None:
    # Una sola llamada a horses2plt con todos los .hsol en orden, se escriban o no los datos.
    case = make_case(tmp_path, stems=("b", "c", "a"))
    convert(
        case,
        WRITE_SIM_PARAMETER=write_sim_parameter,
        OUTPUT_PARAMETERS="--output-type=vtkhdf",
        LOG=False,
    )
    assert len(fake_run.calls) == 1
    args, kwargs = fake_run.calls[0]
    assert not kwargs.get("shell", False)
    assert kwargs["cwd"] == case.hsol_dir
    assert args == [
        str(case.horses2plt),
        "a.hsol",
        "b.hsol",
        "c.hsol",
        os.path.join("..", "mesh", "cylinder.hmesh"),
        "--output-type=vtkhdf",
    ]
    for stem in ("a", "b", "c"):
        out_file = case.out / f"{stem}.h5"
        assert out_file.is_file()
        if write_sim_parameter:
            assert_simulation_info(out_file, case.headers[stem])
        else:
            with h5py.File(out_file, "r") as hdf_file:
                assert "SimulationInfo" not in hdf_file["VTKHDF"]


def _truncated(hsol: Path) -> None:
    hsol.write_bytes(SyntheticHeader().to_bytes()[:100])


def _bad_terminator(hsol: Path) -> None:
    write_hsol(hsol, SyntheticHeader(terminator=0))


def _mesh_file_type(hsol: Path) -> None:
    write_hsol(hsol, SyntheticHeader(file_type=1))


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_truncated, "fewer than the 204"),
        (_bad_terminator, "terminator 88"),
        (_mesh_file_type, "file type 1"),
    ],
)
def test_invalid_header_rejected_before_horses2plt(
    tmp_path: Path,
    fake_run: FakeHorses2plt,
    corrupt: Callable[[Path], None],
    message: str,
) -> None:
    # Basta con que un .hsol (el último por orden) tenga mal la cabecera para que no se haga
    # nada: ni carpeta de salida ni llamada a horses2plt.
    case = make_case(tmp_path, stems=("sol_0001", "sol_0002", "sol_0003"))
    corrupt(case.hsol_files[-1])
    with pytest.raises(ValueError, match=message) as info:
        convert(case, LOG=False)
    assert "sol_0003.hsol" in str(info.value)
    assert fake_run.calls == []
    assert not case.out.exists()


def test_write_sim_parameter_false_does_not_read_headers(
    tmp_path: Path, fake_run: FakeHorses2plt, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Con WRITE_SIM_PARAMETER=False no se leen cabeceras: una inválida no impide convertir.
    case = make_case(tmp_path, stems=("sol",))
    case.hsol_files[0].write_bytes(b"\x00")

    def fail_if_called(hsol_file: Path) -> _HsolHeader:
        raise AssertionError(f"header read for {hsol_file}")

    monkeypatch.setattr(core, "_read_hsol_header", fail_if_called)
    convert(case, WRITE_SIM_PARAMETER=False, LOG=False)
    assert len(fake_run.calls) == 1
    assert (case.out / "sol.h5").is_file()


def test_foreign_hdf_untouched(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    foreign = case.hsol_dir / "other.hdf"
    foreign.write_bytes(b"foreign")
    convert(case, OUT_EXTENSION="both")
    assert foreign.read_bytes() == b"foreign"
    assert not (case.out / "h5" / "other.h5").exists()
    assert not (case.out / "hdf" / "other.hdf").exists()


def test_extended_log_prints_output(
    tmp_path: Path, fake_run: FakeHorses2plt, capsys: pytest.CaptureFixture[str]
) -> None:
    case = make_case(tmp_path, stems=("sol",))
    convert(case, LOG=False, EXTENDED_LOG=True)
    out = capsys.readouterr().out
    assert FAKE_STDOUT in out
    assert ">>>" not in out


def test_log_false_silent(
    tmp_path: Path, fake_run: FakeHorses2plt, capsys: pytest.CaptureFixture[str]
) -> None:
    case = make_case(tmp_path, stems=("sol",))
    HorsesDataHandler()
    convert(case, LOG=False)
    assert capsys.readouterr().out == ""


def test_log_true_prints_progress(
    tmp_path: Path, fake_run: FakeHorses2plt, capsys: pytest.CaptureFixture[str]
) -> None:
    # Un mensaje por fichero con el tiempo completo de su cabecera.
    case = make_case(tmp_path, stems=("sol",))
    convert(case)
    out = capsys.readouterr().out
    assert ">>> " in out
    assert ">>> sol.hsol: time 0.04032508838441574, iteration 1007 written in sol.hdf" in out


# --------------------------------------------------------------------------------------
# Validación y errores (criterios 8 a 12)
# --------------------------------------------------------------------------------------


def test_invalid_extension_no_output_dir(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    with pytest.raises(ValueError, match="extension"):
        # `cast` engaña al analizador de tipos a propósito: se prueba la comprobación en
        # tiempo de ejecución.
        convert(case, OUT_EXTENSION=cast(OutExtension, ".vtk"))
    assert not case.out.exists()
    assert fake_run.calls == []


def test_missing_solver(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    case.solver = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        convert(case)
    assert not case.out.exists()


def test_missing_hsol_dir(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    case.hsol_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        convert(case)
    assert not case.out.exists()


def test_missing_horses2plt(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    case.horses2plt.unlink()
    with pytest.raises(FileNotFoundError, match="horses2plt"):
        convert(case)
    assert not case.out.exists()


def test_no_hsol_files(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    for hsol in case.hsol_files:
        hsol.unlink()
    with pytest.raises(FileNotFoundError, match="hsol"):
        convert(case)
    assert not case.out.exists()


def test_missing_mesh(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    case.mesh = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        convert(case)
    assert not case.out.exists()


def test_mesh_dir_without_hmesh(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    (case.mesh / "cylinder.hmesh").unlink()
    with pytest.raises(FileNotFoundError, match="hmesh"):
        convert(case)
    assert not case.out.exists()


@pytest.mark.parametrize("write_sim_parameter", [True, False])
def test_nonzero_exit_code(
    tmp_path: Path, fake_run: FakeHorses2plt, write_sim_parameter: bool
) -> None:
    case = make_case(tmp_path)
    fake_run.returncode = 1
    fake_run.stderr = "ERROR: mesh and solution do not match"
    with pytest.raises(RuntimeError, match="do not match"):
        convert(case, WRITE_SIM_PARAMETER=write_sim_parameter)


@pytest.mark.parametrize("write_sim_parameter", [True, False])
def test_exit_code_zero_with_error_in_output(
    tmp_path: Path, fake_run: FakeHorses2plt, write_sim_parameter: bool
) -> None:
    # horses2plt imprime el error pero termina con código 0.
    case = make_case(tmp_path)
    fake_run.stdout = MESH_NOT_SPECIFIED_STDOUT
    with pytest.raises(RuntimeError, match="Error in file getTask.f90, in line 434") as info:
        convert(case, WRITE_SIM_PARAMETER=write_sim_parameter, LOG=False)
    assert "Mesh file was not specified" in str(info.value)


@pytest.mark.parametrize("write_sim_parameter", [True, False])
def test_exit_code_zero_with_error_stop_in_stderr(
    tmp_path: Path, fake_run: FakeHorses2plt, write_sim_parameter: bool
) -> None:
    case = make_case(tmp_path)
    fake_run.stderr = REAL_FAILURE_STDERR
    with pytest.raises(RuntimeError, match="ERROR STOP"):
        convert(case, WRITE_SIM_PARAMETER=write_sim_parameter, LOG=False)


@pytest.mark.parametrize("write_sim_parameter", [True, False])
def test_real_failure_output_shows_reason(
    tmp_path: Path, fake_run: FakeHorses2plt, write_sim_parameter: bool
) -> None:
    # Fallo real: el motivo está en stdout y stderr solo trae `ERROR STOP` y la traza.
    case = make_case(tmp_path)
    fake_run.returncode = 1
    fake_run.stdout = REAL_FAILURE_STDOUT
    fake_run.stderr = REAL_FAILURE_STDERR
    with pytest.raises(RuntimeError, match="HDF5 support must be enabled") as info:
        convert(case, WRITE_SIM_PARAMETER=write_sim_parameter, LOG=False)
    message = str(info.value)
    assert "exit code 1" in message
    assert "ERROR STOP" in message
    assert str(case.horses2plt) in message
    assert f"Working directory: {case.hsol_dir}" in message


@pytest.mark.parametrize("write_sim_parameter", [True, False])
def test_hsol_name_too_long(
    tmp_path: Path, fake_run: FakeHorses2plt, write_sim_parameter: bool
) -> None:
    # Criterio D: un nombre de .hsol de 133 caracteres se rechaza antes de hacer nada.
    stem = "a" * (133 - len(".hsol"))
    case = make_case(tmp_path, stems=(stem,))
    with pytest.raises(ValueError, match=r"\.hsol path passed to horses2plt is 133") as info:
        convert(case, WRITE_SIM_PARAMETER=write_sim_parameter, LOG=False)
    message = str(info.value)
    assert f"{stem}.hsol" in message
    assert "limit of 132" in message
    assert "HORSES3D" in message
    assert "LINE_LENGTH" in message
    assert "SMConstants.f90" in message
    assert not case.out.exists()
    assert fake_run.calls == []


def test_hsol_name_too_long_in_bytes(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    # El límite se mide en bytes, como gfortran: 131 caracteres, pero cada "ñ" y cada "á"
    # ocupan 2 bytes en UTF-8, así que el nombre pasa de 132 bytes y se rechaza.
    stem = "ñá" + "a" * (131 - len(".hsol") - 2)
    name = f"{stem}.hsol"
    assert len(name) == 131
    assert len(name.encode("utf-8")) == 133
    case = make_case(tmp_path, stems=(stem,))
    with pytest.raises(ValueError, match=r"is 133 bytes long \(131 characters\)") as info:
        convert(case, LOG=False)
    assert "limit of 132 bytes" in str(info.value)
    assert not case.out.exists()
    assert fake_run.calls == []


def test_hsol_name_at_limit_accepted(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    # Criterio D: exactamente 132 caracteres sí se aceptan.
    stem = "a" * (132 - len(".hsol"))
    case = make_case(tmp_path, stems=(stem,))
    convert(case, LOG=False)
    args, _ = fake_run.calls[0]
    assert args[1] == f"{stem}.hsol"
    assert len(args[1]) == 132
    assert (case.out / f"{stem}.h5").is_file()


def test_mesh_argument_relative_wins(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    # Criterio C: malla en una carpeta vecina de HSOL_DIR; la relativa es la más corta.
    case = make_case(tmp_path, stems=("sol",))
    mesh_file = case.mesh / "cylinder.hmesh"
    convert(case, LOG=False)
    args, _ = fake_run.calls[0]
    assert args[2] == os.path.join("..", "mesh", "cylinder.hmesh")
    assert len(args[2]) < len(os.path.abspath(mesh_file))


def test_mesh_argument_absolute_wins(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    # Criterio C: HSOL_DIR muy anidada y malla cerca de la raíz del caso. La ruta relativa
    # necesita un "../" por nivel y acaba siendo más larga que la absoluta. La profundidad
    # se calcula a partir de la longitud de `tmp_path`, que cambia de un equipo a otro.
    depth = len(str(tmp_path)) // 3 + 2
    case = make_case(tmp_path, stems=("sol",), hsol_dir_name=os.path.join(*["d"] * depth))
    mesh_file = case.mesh / "cylinder.hmesh"
    convert(case, LOG=False)
    args, _ = fake_run.calls[0]
    assert args[2] == os.path.abspath(mesh_file)
    assert len(args[2]) < len(os.path.relpath(mesh_file, case.hsol_dir))
    assert (case.out / "sol.h5").is_file()


def _mesh_in_hsol_dir(case: Case, length: int) -> Path:
    """Crea en ``HSOL_DIR`` una malla cuyo nombre mide ``length`` caracteres.

    Su ruta relativa a ``HSOL_DIR`` es solo el nombre, y la absoluta es más larga: el
    argumento que se pasa a horses2plt mide exactamente ``length`` caracteres.
    """
    mesh_file = case.hsol_dir / ("m" * (length - len(".hmesh")) + ".hmesh")
    mesh_file.write_bytes(b"\x00")
    return mesh_file


def test_mesh_path_too_long(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    # Criterio E: la ruta más corta de la malla mide 133 caracteres.
    case = make_case(tmp_path)
    case.mesh = _mesh_in_hsol_dir(case, 133)
    with pytest.raises(ValueError, match=r"\.hmesh path passed to horses2plt is 133") as info:
        convert(case, LOG=False)
    message = str(info.value)
    assert case.mesh.name in message
    assert "limit of 132" in message
    assert "LINE_LENGTH" in message
    assert not case.out.exists()
    assert fake_run.calls == []


def test_mesh_path_at_limit_accepted(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    # Criterio E: con 132 caracteres exactos se acepta.
    case = make_case(tmp_path, stems=("sol",))
    case.mesh = _mesh_in_hsol_dir(case, 132)
    convert(case, LOG=False)
    args, _ = fake_run.calls[0]
    assert args[2] == case.mesh.name
    assert (case.out / "sol.h5").is_file()


@pytest.mark.parametrize("write_sim_parameter", [True, False])
def test_missing_expected_hdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_sim_parameter: bool
) -> None:
    # horses2plt termina bien pero no escribe el .hdf esperado.
    case = make_case(tmp_path)

    def run_without_output(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, FAKE_STDOUT, "")

    monkeypatch.setattr(core.subprocess, "run", run_without_output)
    with pytest.raises(FileNotFoundError):
        convert(case, WRITE_SIM_PARAMETER=write_sim_parameter)


def test_hdf_without_vtkhdf(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    fake_run.with_vtkhdf = False
    with pytest.raises(ValueError, match="vtkhdf"):
        convert(case)


def test_mesh_dir_with_several_hmesh(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    (case.mesh / "other.hmesh").write_bytes(b"\x00")
    with pytest.raises(ValueError, match="other.hmesh"):
        convert(case)
    assert not case.out.exists()


def test_mesh_file_path(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    (case.mesh / "other.hmesh").write_bytes(b"\x00")
    case.mesh = case.mesh / "other.hmesh"
    convert(case, LOG=False)
    args, _ = fake_run.calls[0]
    # Una sola llamada: ejecutable, los dos .hsol y después la malla.
    assert args[1:4] == [
        "sol_0001.hsol",
        "sol_0002.hsol",
        os.path.join("..", "mesh", "other.hmesh"),
    ]
    assert (case.out / "sol_0001.h5").is_file()


# --------------------------------------------------------------------------------------
# Rutas, utilidades e inicialización (criterios 13 a 15)
# --------------------------------------------------------------------------------------


def test_accepts_pathlib_paths(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path)
    HorsesDataHandler().convert_hsol2hdf(case.solver, case.hsol_dir, case.mesh, case.out, LOG=False)
    assert (case.out / "sol_0001.h5").is_file()
    assert (case.out / "sol_0002.h5").is_file()


def test_get_sizes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "a.hdf").write_bytes(b"x" * 100)
    (tmp_path / "b.hdf").write_bytes(b"x" * 250)
    (tmp_path / "c.h5").write_bytes(b"x" * 1000)
    assert HorsesDataHandler().get_sizes(tmp_path) == 350
    assert "MB" in capsys.readouterr().out
    assert HorsesDataHandler().get_sizes(str(tmp_path), EXTENSION=".h5") == 1000


def test_explore_hdf5_structure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    file_path = tmp_path / "data.hdf"
    with h5py.File(file_path, "w") as hdf_file:
        group = hdf_file.create_group("VTKHDF")
        group.create_dataset("Points", data=np.zeros((4, 3)))
    HorsesDataHandler().explore_hdf5_structure(file_path)
    out = capsys.readouterr().out
    assert "VTKHDF: Group" in out
    assert "VTKHDF/Points: Dataset (4, 3) float64" in out


def test_init_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    HorsesDataHandler()
    assert capsys.readouterr().out == ""


def test_version_matches_metadata() -> None:
    from importlib.metadata import version

    assert __version__ == version("py-horses-data-handler")
