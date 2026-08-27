import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import torch

from abcd.core.GaussianModel import GaussianModel
from abcd.geometry.grid import GridIndex
from abcd.trainers.basic.state import BasicTrainState

CACHE_FORMAT_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: object, destination: Path) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as file:
        torch.save(payload, file)
        file.flush()
        os.fsync(file.fileno())
    size = temporary.stat().st_size
    digest = _sha256(temporary)
    temporary.replace(destination)
    return size, digest


def _atomic_json_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(destination)


def _verify_file(path: Path, metadata: dict) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != metadata["size"]:
        raise ValueError(f"Stored file has unexpected size: {path}")
    digest = _sha256(path)
    if digest != metadata["sha256"]:
        raise ValueError(f"Stored file checksum does not match: {path}")


def _cell_id(index: GridIndex) -> str:
    return f"{index.x}_{index.y}_{index.z}"


def _parse_cell_id(value: str) -> GridIndex:
    x, y, z = value.split("_", maxsplit=2)
    return GridIndex(int(x), int(y), int(z))


def _camera_id(camera_id: object) -> str:
    return quote(str(camera_id), safe="")


@dataclass(frozen=True)
class CachedRender:
    rgb: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor

    def validate(self) -> None:
        if self.rgb.dtype != torch.uint8:
            raise ValueError("Cached RGB must use uint8")
        if self.depth.dtype != torch.float16:
            raise ValueError("Cached depth must use float16")
        if self.alpha.dtype != torch.uint8:
            raise ValueError("Cached alpha must use uint8")
        if self.rgb.ndim != 3 or self.rgb.shape[0] != 3:
            raise ValueError("Cached RGB must have shape (3, H, W)")
        if self.depth.shape != self.alpha.shape:
            raise ValueError("Cached depth and alpha shapes must match")
        if self.depth.ndim != 3 or self.depth.shape[0] != 1:
            raise ValueError("Cached depth and alpha must have shape (1, H, W)")
        if self.rgb.shape[1:] != self.depth.shape[1:]:
            raise ValueError("Cached render image dimensions must match")


class DirectoryRenderCache:
    """Versioned, atomic disk cache for partition-camera renders."""

    def __init__(self, root: str | Path, fingerprint: str):
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.render_root = self.root / "renders"
        manifest_path = self.root / "render-cache.json"
        expected = {
            "format_version": CACHE_FORMAT_VERSION,
            "fingerprint": fingerprint,
        }
        if manifest_path.exists():
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError(
                    f"Render cache at {self.root} is incompatible with this run"
                )
        else:
            _atomic_json_save(expected, manifest_path)

    def _entry_paths(
        self, cell_id: GridIndex, camera_id: object, iteration: int
    ) -> tuple[Path, Path]:
        directory = self.render_root / _cell_id(cell_id) / str(iteration)
        stem = _camera_id(camera_id)
        return directory / f"{stem}.pt", directory / f"{stem}.json"

    def store(
        self,
        cell_id: GridIndex,
        camera_id: object,
        iteration: int,
        render: CachedRender,
    ) -> None:
        render.validate()
        data_path, metadata_path = self._entry_paths(cell_id, camera_id, iteration)
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "cell_id": tuple(cell_id),
            "camera_id": str(camera_id),
            "iteration": iteration,
            "rgb": render.rgb.detach().cpu(),
            "depth": render.depth.detach().cpu(),
            "alpha": render.alpha.detach().cpu(),
        }
        size, digest = _atomic_torch_save(payload, data_path)
        _atomic_json_save(
            {"size": size, "sha256": digest, "fingerprint": self.fingerprint},
            metadata_path,
        )

    def load(
        self, cell_id: GridIndex, camera_id: object, iteration: int
    ) -> CachedRender:
        data_path, metadata_path = self._entry_paths(cell_id, camera_id, iteration)
        if not metadata_path.is_file():
            raise KeyError(
                f"No cached render for cell={cell_id}, camera={camera_id}, "
                f"iteration={iteration}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != self.fingerprint:
            raise ValueError(
                f"Cached render has an incompatible fingerprint: {data_path}"
            )
        _verify_file(data_path, metadata)
        payload = torch.load(data_path, map_location="cpu", weights_only=True)
        if tuple(payload["cell_id"]) != tuple(cell_id):
            raise ValueError(f"Cached render cell identity does not match: {data_path}")
        if payload["camera_id"] != str(camera_id) or payload["iteration"] != iteration:
            raise ValueError(f"Cached render identity does not match: {data_path}")
        render = CachedRender(payload["rgb"], payload["depth"], payload["alpha"])
        render.validate()
        return render

    def iterations(self, cell_id: GridIndex) -> list[int]:
        directory = self.render_root / _cell_id(cell_id)
        if not directory.is_dir():
            return []
        return sorted(
            int(path.name)
            for path in directory.iterdir()
            if path.is_dir() and path.name.isdigit()
        )

    def remove_older_than(self, cell_id: GridIndex, iteration: int) -> None:
        directory = self.render_root / _cell_id(cell_id)
        for old_iteration in self.iterations(cell_id):
            if old_iteration >= iteration:
                continue
            old_directory = directory / str(old_iteration)
            for path in old_directory.iterdir():
                path.unlink()
            old_directory.rmdir()


class MemoryRenderCache:
    """Host-RAM cache for fast partition switching on large-memory machines."""

    def __init__(self):
        self.renders: dict[tuple[GridIndex, str, int], CachedRender] = {}

    def store(self, cell_id, camera_id, iteration, render: CachedRender) -> None:
        render.validate()
        self.renders[(cell_id, str(camera_id), iteration)] = CachedRender(
            render.rgb.detach().cpu().clone(),
            render.depth.detach().cpu().clone(),
            render.alpha.detach().cpu().clone(),
        )

    def load(self, cell_id, camera_id, iteration) -> CachedRender:
        return self.renders[(cell_id, str(camera_id), iteration)]

    def iterations(self, cell_id: GridIndex) -> list[int]:
        return sorted({key[2] for key in self.renders if key[0] == cell_id})

    def remove_older_than(self, cell_id: GridIndex, iteration: int) -> None:
        for key in list(self.renders):
            if key[0] == cell_id and key[2] < iteration:
                del self.renders[key]


@dataclass
class ShardState:
    model: GaussianModel
    training: BasicTrainState


class DirectoryShardStore:
    """Atomic disk storage for inactive partition parameters and training state."""

    def __init__(self, root: str | Path, fingerprint: str):
        self.root = Path(root) / "shards"
        self.fingerprint = fingerprint

    def _paths(self, cell_id: GridIndex) -> tuple[Path, Path]:
        directory = self.root / _cell_id(cell_id)
        return directory / "current.pt", directory / "current.json"

    def descriptions(self) -> dict[GridIndex, dict]:
        if not self.root.is_dir():
            return {}
        descriptions = {}
        for path in self.root.iterdir():
            metadata_path = path / "current.json"
            if not metadata_path.is_file():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") != self.fingerprint:
                raise ValueError(f"Stored shard is incompatible: {metadata_path}")
            if "gaussian_count" not in metadata or "next_iteration" not in metadata:
                # Checkpoints from before lightweight resume metadata still contain
                # the authoritative values in their verified payload.
                data_path = path / "current.pt"
                _verify_file(data_path, metadata)
                payload = torch.load(data_path, map_location="cpu", weights_only=True)
                metadata["gaussian_count"] = len(payload["model_state"]["positions"])
                metadata["next_iteration"] = payload["training"]["next_iteration"]
            descriptions[_parse_cell_id(path.name)] = metadata
        return descriptions

    def store(self, cell_id: GridIndex, state: ShardState) -> None:
        data_path, metadata_path = self._paths(cell_id)
        model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in state.model.state_dict().items()
        }
        payload = {
            "format_version": CACHE_FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "cell_id": tuple(cell_id),
            "model_metadata": {
                "sh_degree": state.model.sh_degree,
                "scales_range": state.model.scales_range,
            },
            "model_state": model_state,
            "training": {
                "next_iteration": state.training.next_iteration,
                "active_sh_degree": state.training.active_sh_degree,
                "optimizer": state.training.optimizer,
                "densification_stopped_for_memory": (
                    state.training.densification_stopped_for_memory
                ),
                "densification_stopped_for_count": (
                    state.training.densification_stopped_for_count
                ),
            },
        }
        size, digest = _atomic_torch_save(payload, data_path)
        _atomic_json_save(
            {
                "size": size,
                "sha256": digest,
                "fingerprint": self.fingerprint,
                "gaussian_count": len(state.model),
                "next_iteration": state.training.next_iteration,
            },
            metadata_path,
        )

    def load(self, cell_id: GridIndex) -> ShardState:
        data_path, metadata_path = self._paths(cell_id)
        if not metadata_path.is_file():
            raise KeyError(f"No stored shard for cell={cell_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != self.fingerprint:
            raise ValueError(
                f"Stored shard has an incompatible fingerprint: {data_path}"
            )
        _verify_file(data_path, metadata)
        payload = torch.load(data_path, map_location="cpu", weights_only=True)
        if tuple(payload["cell_id"]) != tuple(cell_id):
            raise ValueError(f"Stored shard identity does not match: {data_path}")

        model_state = payload["model_state"]
        model_metadata = payload["model_metadata"]
        model = GaussianModel(
            positions=model_state["positions"],
            sh_coefficients=torch.cat(
                [
                    model_state["sh_coefficients_0"],
                    model_state["sh_coefficients_rest"],
                ],
                dim=1,
            ),
            rotations=model_state["rotations"],
            scales=model_state["scales"],
            opacities=model_state["opacities"],
            sh_degree=model_metadata["sh_degree"],
            background_color=model_state["background_color"],
            scales_range=model_metadata["scales_range"],
        )
        model.load_state_dict(model_state)
        training_payload = payload["training"]
        training = BasicTrainState(**training_payload)
        return ShardState(model=model, training=training)


class MemoryShardStore:
    """Host-RAM storage for inactive partition models and optimizer state."""

    def __init__(self):
        self.states: dict[GridIndex, ShardState] = {}

    def store(self, cell_id: GridIndex, state: ShardState) -> None:
        state.model.to("cpu")
        self.states[cell_id] = state

    def load(self, cell_id: GridIndex) -> ShardState:
        return self.states[cell_id]
