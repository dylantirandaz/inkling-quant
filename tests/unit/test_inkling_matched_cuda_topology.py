"""CPU-only CUDA Driver topology tests for the exact matched cell."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any
from uuid import UUID

import pytest

from inkling_quant_lab.gguf.inkling_matched_execution import (
    enumerate_matched_cuda_peer_topology,
    order_matched_nvidia_smi_identity_by_cuda_uuid,
    parse_matched_nvidia_smi_identity_csv,
)

pytestmark = pytest.mark.unit

CUDA_DRIVER_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so.1"


def _uuid(index: int) -> str:
    return f"GPU-{index + 1:08x}-0000-0000-0000-{index + 1:012x}"


def _identity_csv(*, reverse: bool = False, replace_last: bool = False) -> str:
    indices = tuple(reversed(range(8))) if reverse else tuple(range(8))
    rows = [f"{_uuid(index)}, NVIDIA B300 SXM6 AC, 275040, 590.44, 10.3" for index in indices]
    if replace_last:
        rows[-1] = (
            "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff, NVIDIA B300 SXM6 AC, 275040, 590.44, 10.3"
        )
    return "\n".join(rows)


class _FakeCudaFunction:
    def __init__(self, callback: Callable[..., int]) -> None:
        self._callback = callback
        self.argtypes: list[object] | None = None
        self.restype: object | None = None

    def __call__(self, *arguments: object) -> int:
        return self._callback(*arguments)


class _FakeMatchedCudaDriver:
    def __init__(
        self,
        *,
        device_count: int = 8,
        uuid_indices: tuple[int, ...] = tuple(range(8)),
    ) -> None:
        self.device_count = device_count
        self.uuid_bytes = tuple(
            UUID(_uuid(index).removeprefix("GPU-")).bytes for index in uuid_indices
        )
        self.peer_access_calls: list[tuple[int, int]] = []
        self.peer_attribute_calls: list[tuple[int, int, int]] = []
        self.peer_access_values: dict[tuple[int, int], int] = {}
        self.peer_attribute_values: dict[tuple[int, int, int], int] = {}
        self.peer_access_errors: dict[tuple[int, int], int] = {}
        self.peer_attribute_errors: dict[tuple[int, int, int], int] = {}

        def set_int(pointer: object, value: int) -> None:
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = value

        def device_ordinal(device: object) -> int:
            if not isinstance(device, ctypes.c_int):
                raise AssertionError("fake CUDA device must be a c_int")
            return int(device.value)

        def driver_get_version(pointer: object) -> int:
            set_int(pointer, 13_100)
            return 0

        def device_get_count(pointer: object) -> int:
            set_int(pointer, self.device_count)
            return 0

        def device_get(pointer: object, ordinal: object) -> int:
            if not isinstance(ordinal, int):
                raise AssertionError("CUDA ordinal must be an int")
            set_int(pointer, ordinal)
            return 0

        def device_get_uuid(pointer: object, device: object) -> int:
            target = ctypes.cast(pointer, ctypes.POINTER(ctypes.c_ubyte * 16)).contents
            for offset, value in enumerate(self.uuid_bytes[device_ordinal(device)]):
                target[offset] = value
            return 0

        def can_access_peer(
            pointer: object,
            source_device: object,
            destination_device: object,
        ) -> int:
            pair = (device_ordinal(source_device), device_ordinal(destination_device))
            self.peer_access_calls.append(pair)
            error = self.peer_access_errors.get(pair, 0)
            if error:
                return error
            set_int(pointer, self.peer_access_values.get(pair, 1))
            return 0

        def peer_attribute(
            pointer: object,
            attribute: object,
            source_device: object,
            destination_device: object,
        ) -> int:
            if not isinstance(attribute, int):
                raise AssertionError("CUDA peer attribute must be an int")
            key = (
                device_ordinal(source_device),
                device_ordinal(destination_device),
                attribute,
            )
            self.peer_attribute_calls.append(key)
            error = self.peer_attribute_errors.get(key, 0)
            if error:
                return error
            default = {1: key[0] * 8 + key[1], 2: 1, 3: 1, 4: 1, 5: 0}[attribute]
            set_int(pointer, self.peer_attribute_values.get(key, default))
            return 0

        self.cuInit = _FakeCudaFunction(lambda _flags: 0)
        self.cuDriverGetVersion = _FakeCudaFunction(driver_get_version)
        self.cuDeviceGetCount = _FakeCudaFunction(device_get_count)
        self.cuDeviceGet = _FakeCudaFunction(device_get)
        self.cuDeviceGetUuid_v2 = _FakeCudaFunction(device_get_uuid)
        self.cuDeviceCanAccessPeer = _FakeCudaFunction(can_access_peer)
        self.cuDeviceGetP2PAttribute = _FakeCudaFunction(peer_attribute)


def test_uuid_join_reorders_nvidia_smi_rows_to_cuda_ordinals() -> None:
    parsed = parse_matched_nvidia_smi_identity_csv(_identity_csv(reverse=True))

    ordered = order_matched_nvidia_smi_identity_by_cuda_uuid(
        parsed,
        cuda_gpu_uuids=tuple(_uuid(index) for index in range(8)),
    )

    assert tuple(gpu.cuda_ordinal for gpu in ordered) == tuple(range(8))
    assert tuple(gpu.uuid for gpu in ordered) == tuple(_uuid(index) for index in range(8))


def test_enumerator_joins_by_uuid_and_queries_all_56_directed_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _FakeMatchedCudaDriver()
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: driver)
    nvidia_gpus = parse_matched_nvidia_smi_identity_csv(_identity_csv(reverse=True))

    evidence = enumerate_matched_cuda_peer_topology(
        CUDA_DRIVER_PATH,
        nvidia_smi_gpus=nvidia_gpus,
    )

    expected_pairs = tuple(
        (source, destination)
        for source in range(8)
        for destination in range(8)
        if source != destination
    )
    assert evidence.cuda_driver_api_version == 13_100
    assert evidence.gpu_uuids == tuple(_uuid(index) for index in range(8))
    assert (
        tuple((edge.source_cuda_ordinal, edge.destination_cuda_ordinal) for edge in evidence.edges)
        == expected_pairs
    )
    assert driver.peer_access_calls == list(expected_pairs)
    assert driver.peer_attribute_calls == [
        (source, destination, attribute)
        for source, destination in expected_pairs
        for attribute in range(1, 6)
    ]
    assert all(edge.can_access_peer and edge.access_supported for edge in evidence.edges)
    assert all(edge.native_atomic_supported for edge in evidence.edges)
    assert all(edge.cuda_array_access_supported for edge in evidence.edges)
    assert all(not edge.only_partial_native_atomic_supported for edge in evidence.edges)
    assert driver.cuDeviceCanAccessPeer.argtypes == [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    assert driver.cuDeviceGetP2PAttribute.argtypes == [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]


@pytest.mark.parametrize("device_count", (0, 7, 9))
def test_enumerator_requires_exactly_eight_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
    device_count: int,
) -> None:
    driver = _FakeMatchedCudaDriver(device_count=device_count)
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: driver)

    with pytest.raises(RuntimeError, match="exactly eight GPUs"):
        enumerate_matched_cuda_peer_topology(
            CUDA_DRIVER_PATH,
            nvidia_smi_gpus=parse_matched_nvidia_smi_identity_csv(_identity_csv()),
        )


def test_enumerator_rejects_uuid_inventory_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _FakeMatchedCudaDriver()
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: driver)

    with pytest.raises(RuntimeError, match="differs from nvidia-smi identity"):
        enumerate_matched_cuda_peer_topology(
            CUDA_DRIVER_PATH,
            nvidia_smi_gpus=parse_matched_nvidia_smi_identity_csv(_identity_csv(replace_last=True)),
        )


def test_enumerator_rejects_duplicate_cuda_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _FakeMatchedCudaDriver(uuid_indices=(0, 1, 2, 3, 4, 5, 6, 6))
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: driver)

    with pytest.raises(ValueError, match="eight unique GPUs"):
        enumerate_matched_cuda_peer_topology(
            CUDA_DRIVER_PATH,
            nvidia_smi_gpus=parse_matched_nvidia_smi_identity_csv(_identity_csv()),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (("access", (0, 1), 2), "access result must be zero or one"),
        (("attribute", (0, 1, 1), -1), "performance rank must be non-negative"),
        (("attribute", (0, 1, 2), 0), "access queries disagree"),
        (
            ("attribute", (0, 1, 5), 1),
            "atomic support cannot be both full and partial",
        ),
    ),
)
def test_enumerator_rejects_inconsistent_peer_results(
    monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, tuple[int, ...], int],
    match: str,
) -> None:
    driver = _FakeMatchedCudaDriver()
    kind, key, value = mutation
    if kind == "access":
        driver.peer_access_values[(key[0], key[1])] = value
    else:
        driver.peer_attribute_values[(key[0], key[1], key[2])] = value
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: driver)

    with pytest.raises(RuntimeError, match=match):
        enumerate_matched_cuda_peer_topology(
            CUDA_DRIVER_PATH,
            nvidia_smi_gpus=parse_matched_nvidia_smi_identity_csv(_identity_csv()),
        )


@pytest.mark.parametrize(
    "symbol",
    (
        "cuInit",
        "cuDriverGetVersion",
        "cuDeviceGetCount",
        "cuDeviceGet",
        "cuDeviceGetUuid_v2",
        "cuDeviceCanAccessPeer",
        "cuDeviceGetP2PAttribute",
    ),
)
def test_enumerator_requires_every_driver_symbol(
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    driver = _FakeMatchedCudaDriver()
    delattr(driver, symbol)
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: driver)

    with pytest.raises(RuntimeError, match=rf"required function {symbol}"):
        enumerate_matched_cuda_peer_topology(
            CUDA_DRIVER_PATH,
            nvidia_smi_gpus=parse_matched_nvidia_smi_identity_csv(_identity_csv()),
        )


def test_enumerator_rejects_cuda_query_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _FakeMatchedCudaDriver()
    driver.peer_attribute_errors[(0, 1, 4)] = 701
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: driver)

    with pytest.raises(RuntimeError, match=r"CUDA-array-access-supported query failed.*701"):
        enumerate_matched_cuda_peer_topology(
            CUDA_DRIVER_PATH,
            nvidia_smi_gpus=parse_matched_nvidia_smi_identity_csv(_identity_csv()),
        )


def test_enumerator_rejects_build_stub_path(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def load_driver(_path: str) -> Any:
        nonlocal called
        called = True
        return _FakeMatchedCudaDriver()

    monkeypatch.setattr(ctypes, "CDLL", load_driver)

    with pytest.raises(ValueError, match="build stub"):
        enumerate_matched_cuda_peer_topology(
            "/usr/local/cuda/lib64/stubs/libcuda.so",
            nvidia_smi_gpus=parse_matched_nvidia_smi_identity_csv(_identity_csv()),
        )
    assert called is False
