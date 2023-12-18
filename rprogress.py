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


class Progress:
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
    ) -> Iterable[ProgressType]:
        """Track progress by iterating over a sequence.

        Args:
            sequence (Iterable[ProgressType]): A sequence (must support "len") you wish to iterate over.
            description (str, optional): Description of task show next to progress bar. Defaults to "Working".
            total: (float, optional): Total number of steps. Default is len(sequence).
            auto_refresh (bool, optional): Automatic refresh, disable to force a refresh after each iteration. Default is True.
            transient: (bool, optional): Clear the progress on exit. Defaults to False.
            console (Console, optional): Console to write to. Default creates internal Console instance.
            refresh_per_second (float): Number of times per second to refresh the progress information. Defaults to 10.
            style (StyleType, optional): Style for the bar background. Defaults to "bar.back".
            complete_style (StyleType, optional): Style for the completed bar. Defaults to "bar.complete".
            finished_style (StyleType, optional): Style for a finished bar. Defaults to "bar.finished".
            pulse_style (StyleType, optional): Style for pulsing bars. Defaults to "bar.pulse".
            update_period (float, optional): Minimum time (in seconds) between calls to update(). Defaults to 0.1.
            disable (bool, optional): Disable display of progress.
            show_speed (bool, optional): Show speed if total isn't known. Defaults to True.
        Returns:
            Iterable[ProgressType]: An iterable of the values in the sequence.

        """
        self.description = description
        self.sequence = sequence
        self.total = (
            total or len(self.sequence)
            if not isinstance(self.sequence, Iterator)
            else None
        )
        self.update_period = update_period

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
        self.tasks = self.progress._tasks

    @property
    def desc(self):
        return self.description

    def update(self, task_id, advance=1, **kwargs):
        self.progress.update(task_id, advance=advance, **kwargs)

    @desc.setter
    def desc(self, desc):
        self.progress.update(self.task, description=desc)

    def __iter__(self):
        with self.progress:
            self.task = self.progress.add_task(self.description, total=self.total)
            for it in self.sequence:
                self.progress.update(self.task, advance=1)
                yield it


if __name__ == "__main__":

    def process(it):
        print(it)
        time.sleep(random.randint(0, 10) / 10)

    progress = Progress(range(10))
    for it in progress:
        process(it)
        progress.desc = f"{it} done."

    prog = Progress()
    with prog.progress:
        task1 = prog.progress.add_task("taski", total=10)
        for i in range(10):
            prog.progress._tasks[task1].description = f"taski {i}"
            task2 = prog.progress.add_task(f"task{i}j", total=10)
            for j in range(10):
                prog.progress._tasks[task2].description = f"task{i}j {j}"
                process(f"{i}_{j}")
                prog.progress.update(task2, advance=1)
            prog.progress.update(task1, advance=1)



