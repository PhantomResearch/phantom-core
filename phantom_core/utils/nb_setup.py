"""
This module provides a standard tooling set for Jupyter notebooks.

It includes imports for common data analysis and visualization libraries,
as well as utility functions for displaying DataFrames and setting the
working directory. This module is intended to be imported using
'from stonks.utils.nb_setup import *' in notebooks to provide quick
access to these tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display
import os
import json
import asyncio
import nest_asyncio
nest_asyncio.apply()

from .market_dataframe import MarketDataFrame
from .utils import configure_pandas_display
from .plotting.annotations import annotate_ax


np = np
pd = pd
plt = plt
Path = Path
json = json
MarketDataFrame = MarketDataFrame
annotate_ax = annotate_ax

configure_pandas_display()

def displaydfs(*args, n: int = 3):
    """
    Display multiple DataFrames or objects.

    Args:
        *args: Variable number of objects to display.
        n (int): Number of rows to display for each DataFrame (default: 3).
    """
    for obj in args:
        display(obj)


def set_cwd_to_phantom_root(rel_path: str = '.'):
    """
    Set the current working directory to the Phantom root path.

    Args:
        rel_path (str): Relative path from the Phantom root (default: '.').

    This function changes the current working directory to the specified
    path relative to the Phantom root, and adds the new directory to the
    Python path.
    """

    if 'PHANTOM_ROOT_DIR' not in os.environ:
        raise ValueError("PHANTOM_ROOT_DIR not set")
    PHANTOM_ROOT_PATH = Path(os.environ['PHANTOM_ROOT_DIR'])

    print(f'cwd before: {os.getcwd()}')
    new_dir = PHANTOM_ROOT_PATH / rel_path
    os.chdir(new_dir)
    print(f'cwd now: {os.getcwd()}')
    
    # Add the new working directory to Python path
    import sys
    if str(new_dir) not in sys.path:
        sys.path.append(str(new_dir))
        print(f'Added {new_dir} to Python path')