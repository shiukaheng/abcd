import json

import pytest
import torch
from conftest import make_gaussian_model

from abcd.geometry.grid import GridIndex
from abcd.trainers.basic.state import BasicTrainState
from abcd.trainers.grid.storage import (
    CachedRender,
    DirectoryRenderCache,
    DirectoryShardStore,
    MemoryRenderCache,
    MemoryShardStore,
    ShardState,
)


def cached_render():
    return CachedRender(
        rgb=torch.arange(18, dtype=torch.uint8).reshape(3, 2, 3),
        depth=torch.arange(6, dtype=torch.float16).reshape(1, 2, 3),
        alpha=torch.full((1, 2, 3), 127, dtype=torch.uint8),
    )


def test_render_cache_round_trip_and_culling(tmp_path):
    cache = DirectoryRenderCache(tmp_path, "fixture")
    cell = GridIndex(1, -2, 3)
    cache.store(cell, "camera/1", 10, cached_render())
    cache.store(cell, "camera/1", 20, cached_render())

    loaded = cache.load(cell, "camera/1", 10)
    torch.testing.assert_close(loaded.rgb, cached_render().rgb)
    torch.testing.assert_close(loaded.depth, cached_render().depth)
    torch.testing.assert_close(loaded.alpha, cached_render().alpha)
    assert cache.iterations(cell) == [10, 20]

    cache.remove_older_than(cell, 20)
    assert cache.iterations(cell) == [20]
    with pytest.raises(KeyError):
        cache.load(cell, "camera/1", 10)


def test_memory_cache_and_shard_store_round_trip():
    render_cache = MemoryRenderCache()
    cell = GridIndex(1, -2, 3)
    render_cache.store(cell, "camera/1", 10, cached_render())
    torch.testing.assert_close(
        render_cache.load(cell, "camera/1", 10).rgb, cached_render().rgb
    )

    state = ShardState(
        make_gaussian_model(torch.tensor([[0.0, 1.0, 2.0]]), sh_degree=1),
        BasicTrainState(next_iteration=17),
    )
    shard_store = MemoryShardStore()
    shard_store.store(cell, state)
    assert shard_store.load(cell) is state


def test_render_cache_rejects_wrong_fingerprint_and_corruption(tmp_path):
    cache = DirectoryRenderCache(tmp_path, "fixture")
    cell = GridIndex(0, 0, 0)
    cache.store(cell, 1, 1, cached_render())

    with pytest.raises(ValueError, match="incompatible"):
        DirectoryRenderCache(tmp_path, "different")

    metadata_path = next((tmp_path / "renders").rglob("*.json"))
    metadata = json.loads(metadata_path.read_text())
    data_path = metadata_path.with_suffix(".pt")
    data_path.write_bytes(data_path.read_bytes()[:-1])
    assert metadata["size"] != data_path.stat().st_size
    with pytest.raises(ValueError, match="size"):
        cache.load(cell, 1, 1)


def test_shard_store_round_trip_preserves_model_and_training_state(tmp_path):
    model = make_gaussian_model(
        torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), sh_degree=1
    )
    model._gradient_accumulator.copy_(torch.tensor([[1.0], [2.0]]))
    training = BasicTrainState(
        next_iteration=17,
        active_sh_degree=1,
        optimizer={
            "positions": {
                "step": torch.tensor(17.0),
                "exp_avg": torch.ones_like(model.positions),
                "exp_avg_sq": torch.full_like(model.positions, 2.0),
            }
        },
    )
    store = DirectoryShardStore(tmp_path, "fixture")
    cell = GridIndex(-1, 2, 4)

    store.store(cell, ShardState(model, training))
    loaded = store.load(cell)

    assert loaded.training.next_iteration == 17
    assert loaded.training.active_sh_degree == 1
    torch.testing.assert_close(loaded.model.positions, model.positions)
    torch.testing.assert_close(
        loaded.model._gradient_accumulator, model._gradient_accumulator
    )
    torch.testing.assert_close(
        loaded.training.optimizer["positions"]["exp_avg"],
        training.optimizer["positions"]["exp_avg"],
    )


def test_shard_descriptions_allow_resume_without_loading_models(tmp_path):
    model = make_gaussian_model(torch.tensor([[0.0, 1.0, 2.0]]), sh_degree=1)
    store = DirectoryShardStore(tmp_path, "fixture")
    cell = GridIndex(-1, 2, 4)

    store.store(cell, ShardState(model, BasicTrainState(next_iteration=17)))

    assert store.descriptions() == {
        cell: {
            "fingerprint": "fixture",
            "gaussian_count": 1,
            "next_iteration": 17,
            "sha256": store.descriptions()[cell]["sha256"],
            "size": store.descriptions()[cell]["size"],
        }
    }


def test_shard_descriptions_support_older_checkpoint_metadata(tmp_path):
    model = make_gaussian_model(torch.tensor([[0.0, 1.0, 2.0]]), sh_degree=1)
    store = DirectoryShardStore(tmp_path, "fixture")
    cell = GridIndex(-1, 2, 4)
    store.store(cell, ShardState(model, BasicTrainState(next_iteration=17)))
    metadata_path = tmp_path / "shards" / "-1_2_4" / "current.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["gaussian_count"]
    del metadata["next_iteration"]
    metadata_path.write_text(json.dumps(metadata))

    description = store.descriptions()[cell]

    assert description["gaussian_count"] == 1
    assert description["next_iteration"] == 17
