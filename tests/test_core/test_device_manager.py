"""Tests for the GPU device manager."""

import torch
import pytest
from astraios.core.device_manager import Backend, DeviceManager, get_device_manager


class TestDeviceManager:
    def test_singleton(self):
        dm1 = get_device_manager()
        dm2 = get_device_manager()
        assert dm1 is dm2

    def test_backend_detected(self):
        dm = get_device_manager()
        assert dm.backend in (Backend.CUDA, Backend.MPS, Backend.CPU)

    def test_device_is_torch_device(self):
        dm = get_device_manager()
        assert isinstance(dm.device, torch.device)

    def test_info(self):
        dm = get_device_manager()
        info = dm.info
        assert info.name
        assert info.backend == dm.backend

    def test_tensor_creation(self):
        dm = get_device_manager()
        t = dm.zeros(10, 10)
        assert t.shape == (10, 10)
        assert t.device == dm.device
        assert t.dtype == torch.float32

    def test_ones(self):
        dm = get_device_manager()
        t = dm.ones(5, 5)
        assert torch.all(t == 1)

    def test_to_device_and_back(self):
        dm = get_device_manager()
        cpu_t = torch.randn(4, 4)
        dev_t = dm.to_device(cpu_t)
        back = dm.to_cpu(dev_t)
        assert back.device == torch.device("cpu")
        assert torch.allclose(cpu_t, back)

    def test_from_numpy(self):
        import numpy as np
        dm = get_device_manager()
        arr = np.random.rand(3, 3).astype(np.float32)
        t = dm.from_numpy(arr)
        assert t.shape == (3, 3)
        assert t.device == dm.device

    def test_optimal_batch_size(self):
        dm = get_device_manager()
        bs = dm.optimal_batch_size(1024 * 1024)  # 1MB per frame
        assert bs >= 1

    def test_synchronize_no_error(self):
        dm = get_device_manager()
        dm.synchronize()

    def test_empty_cache_no_error(self):
        dm = get_device_manager()
        dm.empty_cache()
