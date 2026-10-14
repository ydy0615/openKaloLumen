"""Exceptions shared by cancellable application computations."""


class Cancelled(Exception):
    """Signal cooperative cancellation without reporting a task failure."""
