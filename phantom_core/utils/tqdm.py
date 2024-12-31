import os
import sys
from typing import Iterable, Any
from tqdm import tqdm as original_tqdm
from tqdm.notebook import tqdm as notebook_tqdm 
import time
from collections.abc import Sized


class MinimalTqdm(original_tqdm):
    """
    A minimal implementation of tqdm progress bar with added timestamp.

    This class extends the original tqdm class to include a timestamp in the progress bar
    and sets a default minimum interval for updates.
    """

    _bar_format = '{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] [{timestamp}]\n'

    def __init__(self, *args, mininterval=5, **kwargs):
        """
        Initialize the MinimalTqdm progress bar.

        Args:
            *args: Variable length argument list.
            mininterval (int): Minimum interval between progress bar updates in seconds. Defaults to 5.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs, mininterval=mininterval, bar_format=self._bar_format)

    def format_meter(self, *args, **kwargs):
        """
        Format the progress meter with a timestamp.

        This method adds a timestamp to the progress meter and calls the parent class's format_meter method.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The formatted progress meter string.
        """
        # Add timestamp to kwargs
        kwargs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        return super().format_meter(*args, **kwargs)


class MinimalTqdmCustom:
    """
    A minimal implementation of a progress bar, similar to tqdm.

    This class provides a simple progress bar that can be used in environments
    where a full-featured tqdm might not be suitable or desired.
    """

    def __init__(
        self, 
        iterable: Iterable | None = None, 
        total: int | None = None, 
        desc: str | None = None, 
        disable: bool = False,
        leave: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MinimalTqdm object.

        Args:
            iterable (Iterable[Any] | None): The iterable to iterate over.
            total (int | None): The total number of iterations.
            desc (str | None): The description of the progress bar.
            disable (bool): Whether to disable the progress bar. defaults to false.
            leave (bool): Whether to leave the progress bar on screen after completion. defaults to true.
            **kwargs (Any): Additional keyword arguments.
        """
        self.iterable = iter(iterable) if iterable else None
        self.total: int | None = total or (len(iterable) if isinstance(iterable, Sized) else None)
        self.desc: str | None = desc
        self.disable: bool = disable
        self.leave: bool = leave
        self.n: int = 0
        self.start_time: float = time.time()
        self.last_print_time: float = self.start_time
        self.print_interval: int = 5  # print update every 5 seconds

    def __iter__(self) -> 'MinimalTqdmCustom':
        """
        Make the MinimalTqdm object iterable.

        Returns:
            MinimalTqdm: The iterable object.
        """
        return self

    def __next__(self) -> Any:
        """
        Get the next item in the iteration.

        Returns:
            Any: The next item in the iterable.

        Raises:
            StopIteration: When the iteration is complete.
        """
        if self.iterable is None:
            raise StopIteration
        
        if self.total is not None and self.n >= self.total:
            raise StopIteration
        
        try:
            item = next(self.iterable)
            self.n += 1
            self._maybe_print_update()
            return item
        except StopIteration:
            self.close()
            raise

    def update(self, n: int = 1) -> None:
        """
        Update the progress bar by incrementing the counter.

        Args:
            n (int): The number of iterations to increment by. defaults to 1.
        """
        self.n += n
        self._maybe_print_update()

    def _maybe_print_update(self) -> None:
        """
        Print an update if the print interval has elapsed.
        """
        current_time: float = time.time()
        if current_time - self.last_print_time >= self.print_interval and not self.disable:
            self._print_update()
            self.last_print_time = current_time

    def _print_update(self) -> None:
        """
        Print the current progress of the iteration.
        """
        if self.total and not self.disable:
            percent: float = self.n / self.total * 100
            print(f"{self.desc}: {self.n}/{self.total} ({percent:.1f}%)")
        else:
            print(f"{self.desc}: {self.n} iterations")

    def close(self) -> None:
        """
        Close the progress bar and print a final update.
        """
        self._print_update()

    def set_description(self, desc: str | None = None, refresh: bool = True) -> None:
        """
        Set or update the description of the progress bar.

        Args:
            desc (str | None): the new description. if none, the description is not changed.
            refresh (bool): whether to refresh the display immediately. defaults to true.
        """
        self.desc = desc
        if refresh and not self.disable:
            self._print_update()


if 'ipykernel' in sys.modules:
    # if running in a jupyter notebook
    tqdm = notebook_tqdm
elif 'MINIMAL_TQDM' in os.environ:
    # if in minimal output mode (for production)
    tqdm = MinimalTqdm
else:
    # if running in a terminal with full output
    tqdm = original_tqdm
