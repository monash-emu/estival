from typing import Callable, Iterable, Optional, Type

import multiprocessing as mp

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import cloudpickle


# initialize a worker in the process pool
def process_init_cloudpickle(custom_cpkl):
    """Initialize a mp.Process with custom cloudpickled data

    Args:
        custom_cpkl: The cloudpickle.dumps output of the custom data
    """
    # declare global variable
    global custom_data
    # assign the global variable
    custom_data = cloudpickle.loads(custom_cpkl)


def generic_cpkl_worker(*args):
    """Worker function to be used with multiprocessing pools, where
    custom_data is a global initialized to a function, usually via
    process_init_cloudpickle

    Returns:
        Return type of the custom global function
    """
    global custom_data
    run_func = custom_data

    return run_func(*args)


def map_parallel(
    run_func: Callable,
    input_iterator: Iterable,
    n_workers: Optional[int] = None,
    mode: Optional[str] = "process",
):
    """Map the values of input_iterator over a function run_func, using n_workers parallel workers
    Defaults to ProcessPoolExecutor; for non-Python-bound tasks, 'thread'

    Args:
        run_func: The function to call over the mapped inputs
        input_iterator: An iterable containing the values to map
        n_workers: Number of processes used by Pool
        mode: ProcessExecutor type; either 'thread' or 'process'.  'process' is required for
              non-thread-safe tasks, while 'thread' is usually faster for small jax-heavy tasks

    Returns:
        A list of values return by run_func
    """

    if n_workers is None:
        n_workers = int(mp.cpu_count())

    if mode is None:
        mode = "thread"

    if mode == "process":
        with ProcessPoolExecutor(  # type: ignore
            n_workers, initializer=process_init_cloudpickle, initargs=(cloudpickle.dumps(run_func),)
        ) as pool:
            pres = pool.map(generic_cpkl_worker, input_iterator)
            pres = [p for p in pres]
    elif mode == "thread":
        with ThreadPoolExecutor(n_workers) as pool:  # type: ignore
            pres = pool.map(run_func, input_iterator)
            pres = [p for p in pres]
    else:
        raise ValueError("Mode must be one of ['thread', 'process']")

    return pres
