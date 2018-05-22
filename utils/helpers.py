"""Helper functions"""

from os import environ


def progress(*args):
    """Print progress message."""
    if 'DEBUG_LEVEL' in environ and int(environ['DEBUG_LEVEL']) >= 1:
        print('--- progress:', *args)


def debug(*args):
    """Print debug message."""
    if 'DEBUG_LEVEL' in environ and int(environ['DEBUG_LEVEL']) >= 2:
        print('--- debug:', *args)


def memoize(f):
    """Decorator to memoize functions by argument list."""
    memo = {}

    def mem(*args):
        key = tuple(args[1:])
        try:
            return memo[key]
        except KeyError:
            memo[key] = f(*args)
            return memo[key]

    return mem
