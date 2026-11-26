"""Tests de ``py_horses_data_handler.core``.

No necesitan HORSES3D: ``subprocess.run`` del módulo ``core`` se sustituye por un falso
(`FakeHorses2plt`) que crea, con h5py, el ``.hdf`` que escribiría ``horses2plt`` junto a
cada ``.hsol`` y devuelve una salida de texto con ``Time`` e ``Iteration``. Todos los
ficheros se crean en carpetas temporales (``tmp_path``).
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict, Unpack, cast

import h5py
import numpy as np
import pytest

from py_horses_data_handler import HorsesDataHandler, __version__, core
from py_horses_data_handler.core import OutExtension, _get_iteration, _get_time

FAKE_STDOUT = "Time: 1.5E-02\nIteration: 10"


# --------------------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------------------


@dataclass
class FakeHorses2plt:
    """Sustituto de ``subprocess.run`` que imita a ``horses2plt``.

    Guarda cada llamada (argumentos y opciones) para que los tests las inspeccionen.
    """

    returncode: int = 0
    stdout: str = FAKE_STDOUT
    stderr: str = ""
    with_vtkhdf: bool = True
    calls: list[tuple[list[str], dict[str, object]]] = field(default_factory=list)

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.calls.append((args, kwargs))
        if self.returncode == 0:
            for arg in args:
                if arg.endswith(".hsol"):
                    with h5py.File(Path(arg).with_suffix(".hdf"), "w") as hdf_file:
                        if self.with_vtkhdf:
                            hdf_file.create_group("VTKHDF")
        return subprocess.CompletedProcess(args, self.returncode, self.stdout, self.stderr)


@dataclass
class Case:
    """Rutas de un caso sintético: instalación, soluciones, malla y salida."""

    solver: Path
    horses2plt: Path
    hsol_dir: Path
    hsol_files: list[Path]
    mesh: Path
    out: Path


def make_case(root: Path, stems: tuple[str, ...] = ("sol_0002", "sol_0001")) -> Case:
    """Crea en ``root`` una instalación falsa de HORSES3D, varios ``.hsol`` y una malla."""
    solver = root / "horses3d"
    horses2plt = solver / "Solver" / "bin" / "horses2plt"
    horses2plt.parent.mkdir(parents=True)
    horses2plt.touch()
    hsol_dir = root / "results"
    hsol_dir.mkdir()
    hsol_files = []
    for stem in stems:
        hsol = hsol_dir / f"{stem}.hsol"
        hsol.write_bytes(b"\x00")
        hsol_files.append(hsol)
    mesh_dir = root / "mesh"
    mesh_dir.mkdir()
    mesh = mesh_dir / "cylinder.hmesh"
    mesh.write_bytes(b"\x00")
    return Case(solver, horses2plt, hsol_dir, hsol_files, mesh_dir, root / "out")


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
# Lectura de Time e Iteration (criterios 1 y 2)
# --------------------------------------------------------------------------------------


def test_get_time_valid() -> None:
    out = "Some logs\nTime: 123.456\nMore logs"
    assert _get_time(out) == pytest.approx(123.456)


def test_get_time_missing() -> None:
    out = "No time here"
    with pytest.raises(ValueError):
        _get_time(out)


def test_get_time_none() -> None:
    with pytest.raises(ValueError):
        _get_time(None)


def test_get_iteration_valid() -> None:
    out = "Start\nIteration: 42\nEnd"
    assert _get_iteration(out) == 42


def test_get_iteration_missing() -> None:
    out = "No iter"
    with pytest.raises(ValueError):
        _get_iteration(out)


def test_get_iteration_none() -> None:
    with pytest.raises(ValueError):
        _get_iteration(None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Time: 1.0E-03", 1.0e-3),
        ("Time: -2.5", -2.5),
        ("Time: 0.1D+01", 1.0),
        ("Time: 12", 12.0),
        ("Time: .5", 0.5),
    ],
)
def test_get_time_formats(text: str, expected: float) -> None:
    assert _get_time(f"log line\n{text}\nIteration: 3") == pytest.approx(expected)


def test_time_read_from_stderr(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    # Time en stderr aunque stdout no esté vacío: antes se ignoraba.
    case = make_case(tmp_path, stems=("sol",))
    fake_run.stdout = "Iteration: 7"
    fake_run.stderr = "Time: 2.0"
    convert(case, LOG=False)
    with h5py.File(case.out / "sol.h5", "r") as hdf_file:
        assert hdf_file["VTKHDF/SimulationInfo/Time"][()] == pytest.approx(2.0)


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
        with h5py.File(out_file, "r") as hdf_file:
            time = hdf_file["VTKHDF/SimulationInfo/Time"]
            iteration = hdf_file["VTKHDF/SimulationInfo/Iteration"]
            assert isinstance(time, h5py.Dataset)
            assert isinstance(iteration, h5py.Dataset)
            assert time.dtype == np.float64
            assert iteration.dtype == np.int64
            assert time[()] == pytest.approx(1.5e-2)
            assert iteration[()] == 10


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


def test_command_is_list_without_shell_and_spaces(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    root = tmp_path / "folder with spaces"
    root.mkdir()
    case = make_case(root, stems=("my sol",))
    convert(case, OUTPUT_PARAMETERS="--output-mode=FE --output-type=vtkhdf")
    assert len(fake_run.calls) == 1
    args, kwargs = fake_run.calls[0]
    assert isinstance(args, list)
    assert not kwargs.get("shell", False)
    assert args == [
        str(case.horses2plt),
        str(case.hsol_files[0]),
        str(case.mesh / "cylinder.hmesh"),
        "--output-mode=FE",
        "--output-type=vtkhdf",
    ]
    assert (case.out / "my sol.h5").is_file()


def test_default_output_parameters(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path, stems=("sol",))
    convert(case, LOG=False)
    args, _ = fake_run.calls[0]
    assert args[3:] == [
        "--output-mode=FE",
        "--output-variables=rho,u,v,w,p,T,Mach",
        "--output-type=vtkhdf",
    ]


def test_write_sim_parameter_true_one_call_per_file_sorted(
    tmp_path: Path, fake_run: FakeHorses2plt
) -> None:
    case = make_case(tmp_path, stems=("b", "c", "a"))
    convert(case, LOG=False)
    hsol_args = [args[1] for args, _ in fake_run.calls]
    assert hsol_args == [str(case.hsol_dir / f"{s}.hsol") for s in ("a", "b", "c")]


def test_write_sim_parameter_false_single_call(tmp_path: Path, fake_run: FakeHorses2plt) -> None:
    case = make_case(tmp_path, stems=("b", "c", "a"))
    convert(case, WRITE_SIM_PARAMETER=False, OUTPUT_PARAMETERS="--output-type=vtkhdf")
    assert len(fake_run.calls) == 1
    args, kwargs = fake_run.calls[0]
    assert not kwargs.get("shell", False)
    assert args == [
        str(case.horses2plt),
        str(case.hsol_dir / "a.hsol"),
        str(case.hsol_dir / "b.hsol"),
        str(case.hsol_dir / "c.hsol"),
        str(case.mesh / "cylinder.hmesh"),
        "--output-type=vtkhdf",
    ]
    for stem in ("a", "b", "c"):
        out_file = case.out / f"{stem}.h5"
        assert out_file.is_file()
        with h5py.File(out_file, "r") as hdf_file:
            assert "SimulationInfo" not in hdf_file["VTKHDF"]


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
    case = make_case(tmp_path, stems=("sol",))
    convert(case)
    assert ">>> " in capsys.readouterr().out


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


def test_missing_expected_hdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # horses2plt termina bien pero no escribe el .hdf esperado.
    case = make_case(tmp_path)

    def run_without_output(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, FAKE_STDOUT, "")

    monkeypatch.setattr(core.subprocess, "run", run_without_output)
    with pytest.raises(FileNotFoundError):
        convert(case, WRITE_SIM_PARAMETER=False)


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
    assert args[2] == str(case.mesh)
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
