import random
import sys
import time
import typing
from typing import (
    BinaryIO,
    Callable,
    Iterable,
    Iterator,
    List,
    NewType,
    Optional,
    Sequence,
    TextIO,
    TypeVar,
    Union,
)

from rich.console import Console
from rich.style import StyleType
from rich.text import Text
from rich import filesize

from rich.progress import (
    Progress as RichProgress,
    BarColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    TextColumn,
    ProgressColumn,
    Task,
)

ProgressType = TypeVar("ProgressType")


class SpeedColumn(ProgressColumn):
    @classmethod
    def render_speed(cls, speed: Optional[float]) -> Text:
        if speed is None:
            return Text("-- it/s", style="progress.percentage")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "×10³", "×10⁶", "×10⁹", "×10¹²"],
            1000,
        )
        data_speed = speed / unit
        return Text(f"{data_speed:.1f}{suffix} it/s", style="progress.percentage")

    def render(self, task: "Task") -> Text:
        return self.render_speed(task.finished_speed or task.speed)


class ProgressManager:
    """管理共享的进度条实例"""
    _instance: Optional['ProgressManager'] = None
    _progress: Optional[RichProgress] = None
    _active_count: int = 0
    _console: Optional[Console] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_progress(
        self,
        auto_refresh: bool = True,
        console: Optional[Console] = None,
        transient: bool = False,
        get_time: Optional[Callable[[], float]] = None,
        refresh_per_second: float = 10,
        style: StyleType = "bar.back",
        complete_style: StyleType = "bar.complete",
        finished_style: StyleType = "bar.finished",
        pulse_style: StyleType = "bar.pulse",
        disable: bool = False,
    ) -> RichProgress:
        """获取或创建共享的 RichProgress 实例"""
        if self._progress is None:
            columns: List["ProgressColumn"] = [
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(
                    style=style,
                    complete_style=complete_style,
                    finished_style=finished_style,
                    pulse_style=pulse_style,
                ),
                TaskProgressColumn(show_speed=True),
                SpeedColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(elapsed_when_finished=True),
            ]
            
            self._console = console or Console()
            self._progress = RichProgress(
                *columns,
                auto_refresh=auto_refresh,
                console=self._console,
                transient=transient,
                get_time=get_time,
                refresh_per_second=refresh_per_second,
                disable=disable,
            )
        return self._progress
    
    def start(self):
        """启动进度条（如果还没启动）"""
        if self._active_count == 0 and self._progress is not None:
            self._progress.__enter__()
        self._active_count += 1
    
    def stop(self):
        """停止进度条（如果没有其他活动任务）"""
        self._active_count -= 1
        if self._active_count == 0 and self._progress is not None:
            self._progress.__exit__(None, None, None)
            self._progress = None
            self._console = None


# 全局进度管理器
_progress_manager = ProgressManager()


def track(
    sequence: Union[Sequence[ProgressType], Iterable[ProgressType]],
    description: str = "Working...",
    total: Optional[float] = None,
    auto_refresh: bool = True,
    console: Optional[Console] = None,
    transient: bool = False,
    get_time: Optional[Callable[[], float]] = None,
    refresh_per_second: float = 10,
    style: StyleType = "bar.back",
    complete_style: StyleType = "bar.complete",
    finished_style: StyleType = "bar.finished",
    pulse_style: StyleType = "bar.pulse",
    update_period: float = 0.1,
    disable: bool = False,
    show_speed: bool = True,
) -> Iterator[ProgressType]:
    """
    追踪可迭代对象的进度，类似 rich.progress.track，但支持多个进度条同时运行。
    
    Args:
        sequence: 要迭代的序列
        description: 进度条描述
        total: 总步数，如果为 None 则尝试使用 len(sequence)
        其他参数同 rich.progress.Progress
    
    Yields:
        序列中的每个元素
    
    Example:
        >>> for item in track(range(100), description="Processing"):
        ...     process(item)
    """
    # 获取共享的进度条实例
    progress = _progress_manager.get_progress(
        auto_refresh=auto_refresh,
        console=console,
        transient=transient,
        get_time=get_time,
        refresh_per_second=refresh_per_second,
        style=style,
        complete_style=complete_style,
        finished_style=finished_style,
        pulse_style=pulse_style,
        disable=disable,
    )
    
    # 计算总数
    if total is None:
        try:
            total = len(sequence)
        except (TypeError, AttributeError):
            total = None
    
    # 启动进度条
    _progress_manager.start()
    
    try:
        # 添加任务
        task_id = progress.add_task(description, total=total)
        
        # 迭代序列
        for item in sequence:
            yield item
            progress.update(task_id, advance=1)
            
    finally:
        # 停止进度条
        _progress_manager.stop()


class Progress:
    """兼容原有的 Progress 类"""
    def __init__(
        self,
        sequence: Union[Sequence[ProgressType], Iterable[ProgressType]] = [],
        description: str = "Working...",
        total: Optional[float] = None,
        auto_refresh: bool = True,
        console: Optional[Console] = None,
        transient: bool = False,
        get_time: Optional[Callable[[], float]] = None,
        refresh_per_second: float = 10,
        style: StyleType = "bar.back",
        complete_style: StyleType = "bar.complete",
        finished_style: StyleType = "bar.finished",
        pulse_style: StyleType = "bar.pulse",
        update_period: float = 0.1,
        disable: bool = False,
        show_speed: bool = True,
        other_columns: List["ProgressColumn"] = [],
        shared: bool = True,
    ):
        self.description = description
        self.sequence = sequence
        self.total = (
            total or len(self.sequence)
            if not isinstance(self.sequence, Iterator)
            else None
        )
        self.update_period = update_period
        self.shared = shared
        self.task = None
        self._own_progress = False

        if shared:
            # 使用共享的进度条
            self.progress = _progress_manager.get_progress(
                auto_refresh=auto_refresh,
                console=console,
                transient=transient,
                get_time=get_time,
                refresh_per_second=refresh_per_second,
                style=style,
                complete_style=complete_style,
                finished_style=finished_style,
                pulse_style=pulse_style,
                disable=disable,
            )
        else:
            # 创建独立的进度条
            columns: List["ProgressColumn"] = (
                [SpinnerColumn(), TextColumn("[progress.description]{task.description}")]
                if self.description
                else []
            )
            columns.extend(
                (
                    BarColumn(
                        style=style,
                        complete_style=complete_style,
                        finished_style=finished_style,
                        pulse_style=pulse_style,
                    ),
                    TaskProgressColumn(show_speed=True),
                    SpeedColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(elapsed_when_finished=True),
                )
            )
            columns.extend(other_columns)
            
            self.progress = RichProgress(
                *columns,
                auto_refresh=auto_refresh,
                console=console,
                transient=transient,
                get_time=get_time,
                refresh_per_second=refresh_per_second or 10,
                disable=disable,
            )
            self._own_progress = True
            
        self.tasks = self.progress._tasks

    @property
    def desc(self):
        return self.description

    def update(self, task_id, advance=1, **kwargs):
        self.progress.update(task_id, advance=advance, **kwargs)

    @desc.setter
    def desc(self, desc):
        if self.task is not None:
            self.progress.update(self.task, description=desc)
        self.description = desc

    def __iter__(self):
        if self.shared:
            _progress_manager.start()
        elif self._own_progress:
            self.progress.__enter__()
            
        try:
            self.task = self.progress.add_task(self.description, total=self.total)
            for it in self.sequence:
                self.progress.update(self.task, advance=1)
                yield it
        finally:
            if self.shared:
                _progress_manager.stop()
            elif self._own_progress:
                self.progress.__exit__(None, None, None)

    def __enter__(self):
        if self.shared:
            _progress_manager.start()
        elif self._own_progress:
            self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.shared:
            _progress_manager.stop()
        elif self._own_progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)

    def add_task(self, description: str, total: Optional[float] = None, **kwargs):
        return self.progress.add_task(description, total=total, **kwargs)


if __name__ == "__main__":
    import threading
    
    def process(it):
        time.sleep(random.randint(0, 10) / 100)
        return it

    # 示例1：使用 track 函数（最简单的方式）
    print("示例1：使用 track 函数")
    for i in track(range(20), description="Processing items"):
        process(i)
    
    print("\n示例2：嵌套使用 track")
    for i in track(range(5), description="外层循环"):
        for j in track(range(10), description=f"内层循环 {i}"):
            process(f"{i}-{j}")
    
    print("\n示例3：并发使用 track（多线程）")
    def worker(name, count):
        for i in track(range(count), description=f"线程 {name}"):
            process(i)
    
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(f"Thread-{i}", 15))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print("\n示例4：混合使用 track 和 Progress 类")
    with Progress(shared=True) as prog:
        task1 = prog.add_task("主任务", total=10)
        for i in range(10):
            prog.progress.update(task1, description=f"主任务 {i}")
            # 在 Progress 内部使用 track
            for j in track(range(5), description=f"子任务 {i}"):
                process(f"{i}-{j}")
            prog.progress.update(task1, advance=1)
