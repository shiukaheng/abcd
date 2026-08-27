import json

from abcd.train import _prepare_cache


def test_prepare_cache_requires_explicit_compatible_resume(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "render-cache.json").write_text(json.dumps({"fingerprint": "same"}))
    (cache / "shards").mkdir()

    assert not _prepare_cache(cache, "same", resume=False)
    assert not cache.exists()

    cache.mkdir()
    (cache / "render-cache.json").write_text(json.dumps({"fingerprint": "same"}))
    (cache / "shards").mkdir()
    assert _prepare_cache(cache, "same", resume=True)
    assert cache.exists()
    assert not _prepare_cache(cache, "different", resume=True)
    assert not cache.exists()
