import pytest
import torch
import torch.distributed as dist
from vortexgrid.engine.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    copy_to_tensor_parallel_region,
    gather_from_tensor_parallel_region,
    reduce_from_tensor_parallel_region,
    scatter_to_tensor_parallel_region,
)


@pytest.fixture(autouse=True)
def cleanup_dist_group():
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.unit
def test_tensor_parallel_primitives_fallback():
    x = torch.randn(2, 4, 16)
    
    # Test autograd primitives in single-process (no dist) fallback mode
    assert torch.allclose(copy_to_tensor_parallel_region(x), x)
    assert torch.allclose(reduce_from_tensor_parallel_region(x), x)
    assert torch.allclose(scatter_to_tensor_parallel_region(x), x)
    assert torch.allclose(gather_from_tensor_parallel_region(x), x)


@pytest.mark.unit
def test_column_parallel_linear_single_node():
    col_layer = ColumnParallelLinear(in_features=16, out_features=32, bias=True, gather_output=True)
    x = torch.randn(2, 4, 16)
    out = col_layer(x)

    assert out.shape == (2, 4, 32)
    out.sum().backward()
    assert col_layer.weight.grad is not None
    assert col_layer.bias.grad is not None


@pytest.mark.unit
def test_row_parallel_linear_single_node():
    row_layer = RowParallelLinear(in_features=16, out_features=8, bias=True, input_is_parallel=False)
    x = torch.randn(2, 4, 16)
    out = row_layer(x)

    assert out.shape == (2, 4, 8)
    out.sum().backward()
    assert row_layer.weight.grad is not None
    assert row_layer.bias.grad is not None


@pytest.mark.unit
def test_column_row_pipeline():
    # Megatron MLP structure: ColumnParallel -> Act -> RowParallel
    col_layer = ColumnParallelLinear(in_features=16, out_features=32, bias=True, gather_output=False)
    row_layer = RowParallelLinear(in_features=32, out_features=16, bias=True, input_is_parallel=True)

    x = torch.randn(2, 4, 16)
    h = col_layer(x)
    assert h.shape == (2, 4, 32)  # Single process world size = 1

    out = row_layer(h)
    assert out.shape == (2, 4, 16)