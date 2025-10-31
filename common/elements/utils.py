import contextlib
import copy
import functools
import inspect
import io
import multiprocessing
import random
import subprocess
import sys
import unittest
import logging
from logging.handlers import RotatingFileHandler
import warnings
import logging
import os
import datetime
from enum import Enum
from subprocess import CalledProcessError
from typing import Any, Generator, Optional, Iterable

import torch
import time
import numpy as np

newline = "\n"

GLOBAL_CACHE_PATH = '/tmp/pipeline_cache'  # temporary cached files
GLOBAL_TMP_PATH = '/tmp/pipeline_output'
GLOBAL_DEPLOY_PATH = '/tmp/pipeline_deploy'
GLOBAL_CICD_MODE = False


class Timer(contextlib.ContextDecorator):
    # Usage: @Timer() decorator or 'with Timer('name'):' context manager
    def __init__(self, name: str = 'Timer', logger: logging.Logger = logging.getLogger(__name__)):
        self.name = name
        self.logger = logger

    def __enter__(self):
        if multiprocessing.parent_process() is None and torch.cuda.is_available():
            # synchronize can not be called in a forked subprocess
            torch.cuda.synchronize()
        self.start = time.perf_counter()

    def __exit__(self, _, __, ___):
        if multiprocessing.parent_process() is None and torch.cuda.is_available():
            # synchronize can not be called in a forked subprocess
            torch.cuda.synchronize()
        self.logger.info(f'{self.name}: {time.perf_counter() - self.start:.6f}s')


class TimerAdvanced:
    """
    This class gives the developer the option to easily time their code. Use it like a context manager.
    """

    def __init__(
            self,
            name: str,
            include_gpu: Optional[bool] = False,
            wait_time: Optional[int] = None,
            print_timings: Optional[bool] = False,
    ):
        self.logger = Logger.setup_logger()

        self.name = name
        self.wait_time = wait_time
        self.include_gpu = include_gpu
        self.print_timings = print_timings
        self.start_real = None
        self.start_cpu = None
        self.end_real = None
        self.end_cpu = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

        if self.print_timings:
            self.print_times()

    def start(self) -> None:
        """
        Start the timer
        """
        if self.include_gpu:
            if multiprocessing.parent_process() is None and torch.cuda.is_available():
                torch.cuda.synchronize()

        self.start_real = time.perf_counter()
        self.start_cpu = time.process_time()

    def stop(self) -> None:
        """
        Stop the timer
        """
        if self.include_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()

        self.end_real = time.perf_counter()
        self.end_cpu = time.process_time()

    def elapsed_real_time(self) -> float:
        """
        Calculated the time, should only be called after calling stop()
        """
        if self.end_real is None:
            self.stop()
        return self.end_real - self.start_real

    def print_times(self) -> None:
        """
        Prints the time of the code
        """
        real_time = self.elapsed_real_time()
        cpu_time = self.end_cpu - self.start_cpu
        self.logger.info(f"Real time for {self.name}: {real_time:.4f} seconds")
        # self.logger.info(f"CPU time for {self.name}: {cpu_time:.4f} seconds")

    def reset(self) -> None:
        """
        Resets the timer
        """
        self.start()

    def wait_until(self) -> None:
        """
        Wait until a certain amount of milliseconds is passed from the moment the timer started
        """
        if self.wait_time is None:
            self.logger.info("Wait time never set, thus can not wait forever")

        current_difference = time.perf_counter() - self.start_real

        if current_difference < self.wait_time:
            time.sleep(self.wait_time - current_difference)

    def get_timings(self) -> dict:
        """
        Get timing data for both real_time and cpu_time
        """
        return {
            "real_time": self.elapsed_real_time(),
            "cpu_time": self.end_cpu - self.start_cpu,
        }


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    blue = "\x1b[34;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class OutputCapture:
    """
    A context manager class to capture the output of print statements.

    Usage:
    ------
    with OutputCapture() as output:
        print("This will be captured")

    # After the context, the captured output is stored in the `output.captured_output` attribute
    print("Captured:", output.captured_output)
    """

    def __enter__(self):
        # Initialize a StringIO object to capture output
        self._output_capture = io.StringIO()

        # Save the current standard output
        self._original_stdout = sys.stdout

        # Redirect the standard output to the StringIO object
        sys.stdout = self._output_capture

        # Return self to be used within the context
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original standard output
        sys.stdout = self._original_stdout

        # Store the captured output for later use
        self.captured_output = self._output_capture.getvalue()

        # Close the StringIO object
        self._output_capture.close()


class Logger:
    """
    Provides a logger for informative print statements and saves them for further investigation
    """
    _logger = None

    @staticmethod
    def setup_logger():
        if Logger._logger is None:
            Logger._logger = logging.getLogger()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            os.makedirs(os.path.join("logs", ), exist_ok=True)

            file_handler = RotatingFileHandler(
                os.path.join(
                    "logs",
                    f"{str(datetime.datetime.now()).replace('-', '_').replace(':', '_')}.log",
                ),
                mode="a",
                maxBytes=10 * 1024 * 1024,
                backupCount=2,
                encoding="utf-8",
                delay=False,
            )
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            Logger._logger.setLevel(logging.INFO)
            Logger._logger.addHandler(file_handler)
            Logger._logger.addHandler(console_handler)
        return Logger._logger


class DevNull:
    """
    Object that can be used to set stdout to, if no output should be given.
    Incorporates possible functions that can be called on stdout.
    """

    def noop(*args, **kwargs): pass

    close = write = flush = writelines = noop


def set_cache_dir(p):
    global GLOBAL_CACHE_PATH
    GLOBAL_CACHE_PATH = os.path.abspath(p)
    get_cache_dir()


def get_cache_dir(sub_dir: str = None):
    global GLOBAL_CACHE_PATH
    p = GLOBAL_CACHE_PATH
    if sub_dir is not None:
        p = os.path.join(p, sub_dir)
    os.makedirs(p, exist_ok=True)
    return p


def get_cicd_tmp(subfolder: str = "output"):
    branchname = os.getenv('CI_COMMIT_REF_SLUG')  # GitLab CICD will set this var to the current branch/tag name
    if branchname is None:
        branchname = "unknown_branch"
    tmp_path = os.path.join(os.sep, "media", "private_data", "cicd", "pipeline", branchname, subfolder)
    return tmp_path


def set_tmp_dir(p):
    global GLOBAL_TMP_PATH
    GLOBAL_TMP_PATH = p
    get_tmp_dir()


def get_tmp_dir(sub_dir: str = None):
    global GLOBAL_TMP_PATH
    p = GLOBAL_TMP_PATH
    if sub_dir is not None:
        p = os.path.join(p, sub_dir)
    try:
        os.makedirs(p, exist_ok=True)
    except Exception as e:
        print(f"Warning could not create temporary folder: {p}: Exception: {e}")
    return p


def set_deploy_dir(p):
    global GLOBAL_DEPLOY_PATH
    GLOBAL_DEPLOY_PATH = p
    get_deploy_dir()


def get_deploy_dir(sub_dir: str = None):
    global GLOBAL_DEPLOY_PATH
    p = GLOBAL_DEPLOY_PATH
    if sub_dir is not None:
        p = os.path.join(p, sub_dir)
    try:
        os.makedirs(p, exist_ok=True)
    except Exception as e:
        print(f"\nWarning: {e}")
    return p


def set_cicd_mode(enabled: bool):
    global GLOBAL_CICD_MODE
    GLOBAL_CICD_MODE = enabled


def get_cicd_mode() -> bool:
    global GLOBAL_CICD_MODE
    return GLOBAL_CICD_MODE


class CICDTestType(Enum):
    SHORT = 1  # no pipelines at all
    FULL_QUICK = 2  # pipelines should run minimal number of epochs and with relaxed performance requirements
    FULL_COMPLETE = 3  # pipelines should run normal number of epochs and with more strict performance requirements


def _get_cicd_test_type() -> CICDTestType:
    if not get_cicd_mode():
        return CICDTestType.FULL_COMPLETE  # always run full code when starting manually
    else:
        ci_source = os.getenv("CI_PIPELINE_SOURCE")
        ci_branch_slug = os.getenv("CI_COMMIT_REF_SLUG")
        ci_test_type = os.getenv("CI_TEST_TYPE")

        if ci_test_type is None:
            if ci_branch_slug == "master":
                return CICDTestType.FULL_COMPLETE  # always run complete for master
            if ci_branch_slug == "dev" and ci_source in ["schedule"]: # , "merge_request_event"]:
                return CICDTestType.FULL_COMPLETE  # run complete for dev only for nightlies
            if ci_source == "merge_request_event":
                return CICDTestType.FULL_QUICK  # merge requests run quick test of pipelines

            # default
            return CICDTestType.SHORT  # for other events (e.g. push) don't run pipelines by default at all
        else:
            if ci_test_type == "SHORT":
                return CICDTestType.SHORT
            elif ci_test_type == "FULL_QUICK":
                return CICDTestType.FULL_QUICK
            else:
                return CICDTestType.FULL_COMPLETE


def get_cicd_test_type() -> CICDTestType:
    type = _get_cicd_test_type()
    return type


class ChangeWorkingDir(object):
    """
    Context manager for changing the current working directory. Will restore to the original working directory on exit.
    """

    def __init__(self, working_dir: str):
        if not os.path.isdir(working_dir):
            raise ValueError(f"Specified working directory does not exist: {working_dir}")

        self._working_dir = working_dir
        self._original_dir: Optional[str] = None

    def __enter__(self):
        self._original_dir = os.getcwd()
        os.chdir(self._working_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._original_dir)


def get_logger(name: str):
    return logging.getLogger(name)


def static_var(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


def wait_forever(msg: str = "Waiting forever."):
    wait_chars = ['|', '/', '-', '\\', '-', '*']
    print(f'{msg}... ', end="", flush=True)
    i = 0
    while True:
        print(wait_chars[i], end="", flush=True)
        time.sleep(2)
        print("\b", end="", flush=True)
        i = 0 if i > len(wait_chars) - 2 else i + 1


class PrettyPrint:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_bold(msg: str):
    print(PrettyPrint.BOLD + msg + PrettyPrint.END)


def disable_batchnorm_pt(model):
    bns = [module for module in model.modules() if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d)]
    for i in range(len(bns)):
        bns[i].momentum = 0.99
        bns[i].track_running_stats = False
    return


def select_device(dev: str = 'cpu'):
    # device = 'cpu' or '0' or '0,1,2,3,n' for the specific GPU device
    if isinstance(dev, torch.device):
        return dev
    if dev != 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = dev
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def reproduce_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_compute_device(CUDA_DEVICES="0", reproduce=False, seed=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES  # select CUDA cards [0 .. 3], multiple like: "2,3"
    if reproduce:
        reproduce_seed(seed)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def write_tensor_as_npy(tensor, file_name):  # convert to NxHxWxC
    if tensor.dim() == 2:
        pass
    elif tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    elif tensor.dim() == 4:
        tensor = tensor.permute(0, 2, 3, 1)
    else:
        raise NotImplementedError(f'Unexpected tensor dimension {tensor.dim()}')
    np.save(file_name, tensor.numpy())


def save_tensor_as_npy(*tensors):
    for t in tensors:
        write_tensor_as_npy(t[0], f'{t[1]}.npy')


def concat_np_arrays(*arrays):
    result = np.array([], arrays[0].dtype)
    for array in arrays:
        result = np.append(result, array)
    return result


def chunks(lst: list[Any], n: int) -> Generator[list[Any], None, None]:
    """
    Cuts a list into n chunks of len(lst).
    Not: the last chunk might be shorter

    :param lst: the input list
    :param n: the size of the chunks
    :return: a generator for list of chunks

    >>> for chunk in chunks([1, 2, 3, 4, 5, 6, 7, 8], 3):
    ...     print(chunk)
    [1, 2, 3]
    [4, 5, 6]
    [7, 8]
    """
    n = min(max(1, n), len(lst))
    return (lst[i:i + n] for i in range(0, len(lst), n))


def get_pt_mean(mult: float = 255) -> tuple[float, float, float]:
    """
    Get the RGB mean of the input images for the pretrained models from the PyTorch model zoo.

    :param mult: multiplier should equal to the max pixel value in the image (e.g. 255 for 8 bits)
    :return: the RGB mean.
    """
    return 0.485 * mult, 0.456 * mult, 0.406 * mult


def get_pt_std(mult: float = 255) -> tuple[float, float, float]:
    """
    Get the RGB standard deviation of the input images for the pretrained models from the PyTorch model zoo.

    :param mult: multiplier should equal to the max pixel value in the image (e.g. 255 for 8 bits)
    :return: the RGB std.
    """
    return 0.229 * mult, 0.224 * mult, 0.225 * mult


def split_dataset(dataset, split=(0.33, 0.33, 0)) -> list[torch.utils.data.Subset]:
    """
    This method splits a torch dataset in parts.

    :param dataset: the torch dataset
    :param split: the list of split items (if the last item is 0 the remaining part is put there)
    :return: a list of Subsets equal to the number of elements in split
    """
    indices = torch.randperm(len(dataset)).tolist()
    result = []
    first = 0
    for s in split:
        if s != 0:
            last = first + round(s * len(indices)) + 1
            split_set = torch.utils.data.Subset(dataset, indices[first:last])
            first = last
        else:
            split_set = torch.utils.data.Subset(dataset, indices[first:])

        result.append(split_set)
    return result


def split_generic_dataset(dataset, split=(0.33, 0.33, 0)) -> list:
    """
    Split samples from GenericDataset into multiple GenericDatasets.

    :param dataset: GenericDataset object.
    :param split: the list of split items (if the last item is 0 the remaining part is put there).
    :return: a list of GenericDatasets equal to the number of elements in split.
    """
    subsets = split_dataset(dataset=dataset, split=split)

    for i, subset in enumerate(subsets):
        subset.dataset = copy.deepcopy(subset.dataset)
        subset.dataset._samples = [subset.dataset.samples[i] for i in subset.indices]
        subsets[i] = subset.dataset
    return subsets


def create_tensor(x: float, dev="cuda:0") -> torch.Tensor:
    """
    Creates a scaler wrapped in a Tensor object with gradients

    :param x: scalar value
    :param dev: device
    :return: the Tensor

    >>> create_tensor(42, "cpu")
    tensor([42.], requires_grad=True)
    """
    return torch.tensor([float(x)], requires_grad=True, device=dev)


string_types = (type(b''), type(u''))


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))

def filter_out_indices(it: Iterable, indices: list):
    """
    Safely removes the specified indices from an iterable.

    Supports lists, NumPy arrays, and PyTorch tensors.

    :param it: Input iterable (list, numpy array, or PyTorch tensor)
    :param indices: List of indices to remove
    :return: A new iterable of the same type as the input but with specified indices removed
    """
    if isinstance(it, list):
        return [v for i, v in enumerate(it) if i not in indices]

    elif isinstance(it, np.ndarray):
        return np.delete(it, indices, axis=0)

    elif isinstance(it, torch.Tensor):
        mask = torch.ones(it.shape[0], dtype=torch.bool)
        mask[indices] = False
        return it[mask]
    else:
        raise TypeError(f"Unsupported type: {type(it)}. Expected list, numpy array, or PyTorch tensor.")

class RunShellCmd(unittest.TestCase):
    def __init__(self, test_script, python_path, cicd):
        super().__init__("test_command")
        self._cicd = cicd
        self._test_script = test_script
        self._python_path = python_path

    def test_command(self):
        custom_env = os.environ.copy()
        if "PYTHONPATH" in custom_env.keys():
            custom_env["PYTHONPATH"] = self._python_path + ":" + custom_env["PYTHONPATH"]
        else:
            custom_env["PYTHONPATH"] = self._python_path
        print(custom_env["PYTHONPATH"])

        script = os.path.abspath(self._test_script)
        if self._cicd:
            cmd = [sys.executable, script, '--cicd']
        else:
            cmd = [sys.executable, script]
        print(f"Running: {cmd}")
        try:
            subprocess.run(cmd, capture_output=True, check=True, env=custom_env)
        except CalledProcessError as e:
            raise RuntimeError(f"Subprocess '{cmd}' failed with STDERR: {e.stderr.decode('utf-8')}\n STDOUT: {e.stdout.decode('utf-8')}\n")


def assert_float_precision_ac(accelerator, model, precision=torch.float32):
    """
    Check if model wrapped with the accelerator decorator is loaded with a certain floating point precision.

    :param accelerator: accelerator from accelerator.Accelerator
    :param model: the model with the accelerator wrapper
    :param precision: the precision to check for
    """
    if accelerator.unwrap_model(model).dtype != precision:
        raise ValueError(
            f"Model loaded with incorrect floating point datatype {accelerator.unwrap_model(model).dtype}."
        )
