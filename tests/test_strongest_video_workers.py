from __future__ import annotations

from types import SimpleNamespace

from scripts import run_strongest_active_learning as runner


class _Adapter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _ImmediateFuture:
    def __init__(self, value: object) -> None:
        self.value = value

    def result(self) -> object:
        return self.value


class _ImmediateExecutor:
    def __init__(
        self,
        *,
        initializer: object,
        initargs: tuple[object, ...],
        **_: object,
    ) -> None:
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self) -> "_ImmediateExecutor":
        self.initializer(*self.initargs)
        return self

    def __exit__(self, *_: object) -> None:
        runner._close_inference_worker_adapter()

    def submit(self, function: object, *args: object, **kwargs: object) -> object:
        return _ImmediateFuture(function(*args, **kwargs))


def test_parallel_video_scheduler_initializes_one_adapter_per_worker_process(
    monkeypatch,
) -> None:
    adapter = _Adapter()
    build_calls: list[str] = []

    def build_adapter(device: str) -> _Adapter:
        build_calls.append(device)
        return adapter

    config = SimpleNamespace(
        device="cuda:0",
        inference=SimpleNamespace(video_workers=2),
        model=SimpleNamespace(build_adapter=build_adapter),
    )
    works = [
        SimpleNamespace(
            work_id=f"work-{index}",
            source=SimpleNamespace(filename=f"video-{index}.mp4"),
        )
        for index in range(2)
    ]

    def run_inference(config_value, work, adapter_value, provenance, **_):
        assert config_value is config
        assert adapter_value is adapter
        assert provenance == {"model_id": "model"}
        return [work.work_id]

    monkeypatch.setattr(runner, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(runner, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(runner, "run_inference", run_inference)

    results, failures = runner._run_parallel_inference(
        config,
        works,
        {"model_id": "model"},
        force=False,
        resume=False,
    )

    assert failures == {}
    assert sorted(results) == ["work-0", "work-1"]
    assert [results[name]["frames"] for name in sorted(results)] == [1, 1]
    assert build_calls == ["cuda:0"]
    assert adapter.closed
