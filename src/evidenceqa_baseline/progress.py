"""Rich 进度显示的轻量封装。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


def iter_with_progress(
    items: Iterable[T],
    *,
    total: int,
    enabled: bool,
    description: str,
    item_label: Callable[[T], str],
) -> Iterator[T]:
    """用 Rich 进度条包装迭代器。

    Args:
        items: 需要迭代的对象。
        total: 总数量。
        enabled: 是否启用进度显示。
        description: 进度条说明文本。
        item_label: 从当前对象提取显示标签的函数。

    Yields:
        原始迭代对象。
    """

    if not enabled:
        yield from items
        return

    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
    except ImportError:
        yield from items
        return

    console = Console(stderr=True)
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[current]}"),
        console=console,
    )
    with progress:
        task_id = progress.add_task(description, total=total, current="")
        for item in items:
            progress.update(task_id, current=item_label(item))
            yield item
            progress.advance(task_id)
        progress.update(task_id, current="完成")


@contextmanager
def rich_status(enabled: bool, message: str) -> Iterator[None]:
    """用 Rich status 显示短阶段状态。

    Args:
        enabled: 是否启用状态显示。
        message: 状态文本。

    Yields:
        无值，仅作为上下文管理器使用。
    """

    if not enabled:
        yield
        return

    try:
        from rich.console import Console
    except ImportError:
        yield
        return

    console = Console(stderr=True)
    with console.status(message):
        yield
