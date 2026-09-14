"""Caught: a desk camera that learns your bad habits and shows you the chart."""

__version__ = "0.1.0"


def main() -> None:
    from .cli import main as _main

    _main()
