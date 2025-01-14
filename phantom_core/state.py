import logging
from pathlib import Path
from typing import Any, Callable
from threading import Lock
import functools
import threading
from typing_extensions import Self
from pydantic import BaseModel, PrivateAttr
from contextlib import contextmanager

from .logging import get_logger


class StateManager:
    """
    For use in multithreaded applications with important state.

    Most methods are classmethods so they can be used as decorators
    over application instance methods.

    StateManager internalizes the lock so the application instance
    does not need to bother with it.

    The instance should initialize a StateManager under the attribute
    "state_mgr" for this to work properly.

    Since the decorators are classmethods, there is a trick where
    the initialized StateManager is pulled out from the decorator
    wrapper arguments and used.

    The workhouse is the `StateManager.manage` decorator which runs
    through the full process but checks for overrides. Decorators like
    StateManager.with_lock apply an override, call the manage decorator,
    and then restore the original state of the overrides.

    The context manager `StateManager.no_lock` is a convenience method
    for temporarily disabling the lock that behaves similarly.
    """

    def __init__(self, save_state_func: Callable | None = None):
        
        self._lock = threading.Lock()

        if save_state_func is not None:
            self._save_state_func = save_state_func
        else:
            self._save_state_func = lambda: None

        self._no_lock_override: bool = False
        self._no_save_override: bool = False

    @classmethod
    def _get_state_mgr(cls, obj: Any) -> Self:
        state_mgr = getattr(obj, 'state_mgr')
        assert isinstance(state_mgr, cls)
        return state_mgr
    
    @classmethod
    def _acquire_lock(cls, state_mgr: Self) -> None:
        if not state_mgr._no_lock_override:
            state_mgr._lock.acquire()

    @classmethod
    def _release_lock(cls, state_mgr: Self) -> None:
        if not state_mgr._no_lock_override:
            state_mgr._lock.release()

    @classmethod
    def _save_state(cls, state_mgr: Self) -> None:
        if not state_mgr._no_save_override:
            state_mgr._save_state_func()

    @classmethod
    def manage(cls, f):

        @functools.wraps(f)
        def wrapper(obj, *args, **kwargs):

            # get the initialized state manager from the object
            state_mgr = cls._get_state_mgr(obj)

            cls._acquire_lock(state_mgr)

            result = f(obj, *args, **kwargs)

            cls._save_state(state_mgr)

            cls._release_lock(state_mgr)

            return result
        
        return wrapper
    
    @classmethod
    def with_lock(cls, f):

        @functools.wraps(f)
        def wrapper(obj, *args, **kwargs):

            state_mgr = cls._get_state_mgr(obj)

            original_state = state_mgr._no_save_override
            state_mgr._no_save_override = True
            
            result = cls.manage(f)(obj, *args, **kwargs)
            
            state_mgr._no_save_override = original_state
            
            return result
            
        return wrapper
    
    @contextmanager
    def no_lock(self):
        original_state = self._no_lock_override
        self._no_lock_override = True
        try:
            yield
        finally:
            self._no_lock_override = original_state

    @contextmanager
    def lock_release(self):
        locked_before = self._lock.locked()
        if locked_before:
            self._lock.release()
        try:
            yield
        finally:
            if locked_before:
                self._lock.acquire()



class PydanticStateModel(BaseModel):
    """
    Intended to be subclassed.

    Behaves like a normal Pydantic model, but also provides:
    1. save method that saves the state to a file in a thread on a timer.
    2. load method that loads the state from a file.
    3. logging capabilities
    """

    _logger: logging.Logger = PrivateAttr(default=get_logger('PydanticStateModel'))
    _save_timer: threading.Timer | None = PrivateAttr(default=None)
    _file_lock: Lock = PrivateAttr(default_factory=Lock)

    def model_post_init(self, __context: Any) -> None:
        """
        Resets the logger name to the subclass name.
        """
        self._logger = get_logger(self.__class__.__name__)

    def _save(self, filepath: str | Path) -> None:
        self._logger.info('[SAVE>]')
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            with open(filepath, 'w') as f:
                f.write(self.model_dump_json(indent=4))
        self._logger.info('[SAVE<]')
        
    def save(
        self, 
        filepath: str | Path, 
        on_thread: bool = True, 
        interval: float = 0.5,
    ) -> None:
        if self._save_timer is not None:
            self._save_timer.cancel()

        if on_thread:
            self._save_timer = threading.Timer(
                interval=interval,
                function=self._save, 
                args=(filepath,)
            )
            self._save_timer.start()
        else:
            self._save(filepath=filepath,)

    @classmethod
    def load(cls, filepath: str | Path) -> Self:
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            data = f.read()
        return cls.model_validate_json(data)