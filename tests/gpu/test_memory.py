import json

import pytest
import torch

from abcd.memory import measure_memory


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_memory_command_writes_a_passing_trace(tmp_path):
    output = measure_memory(tmp_path / "memory.json", partitions=(1, 2))

    payload = json.loads(output.read_text())
    assert payload["pass"]
    assert [record["partitions"] for record in payload["records"]] == [1, 2]
