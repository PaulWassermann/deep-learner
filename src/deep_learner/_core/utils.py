from functools import partial, update_wrapper
from typing import Callable


def format_grad_func_name(func: Callable) -> str:
    """
    Format the name of a function for a better display.

    Parameters
    ----------
    func: a callable
        The callable must have the special attribute `__name__` set.

    Returns
    -------
    str:
        The formatted function name.
    """

    func_name_parts: list[str] = func.__name__.split("_")
    func_name = "".join([part.capitalize() for part in func_name_parts])
    return func_name


def indent_text(text: str, indent: int = 4) -> str:
    """
    Add the specified amount of white spaces before the provided text string.

    Parameters
    ----------
    text: str
        The string to be indented.

    indent: int
        Number of white spaces to prepend to the string.

    Returns
    -------
    str
        The indented text.
    """

    return "\n".join(indent * " " + line for line in text.splitlines())


def partial_wrapper(func: Callable, *args, **kwargs) -> Callable:
    """
    Return a partial function with the appropriate `__name__` and  __doc__` attributes.

    Parameters
    ----------
    func: callable
        Any callable.

    args
        Arguments passed to define the partial function.

    kwargs
        Arguments passed to define the partial function.

    Returns
    -------
    callable:
        A partial function wrapping the input function.
    """

    partial_func = partial(func, *args, **kwargs)
    return update_wrapper(partial_func, func)
