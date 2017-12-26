from contextlib import contextmanager
from logging import getLogger
import sys

logger = getLogger(__name__)


@contextmanager
def largerrecursionlimit(unroll_scan, seqlen):
    if unroll_scan:
        y0 = 1000  # default sys.getrecursionlimit()
        y1 = 2000
        x0 = 33    # seqlen that works with y0
        x1 = 50
        limit = int((y1 - y0) / (x1 - x0) * (seqlen - x0) + y0)
        orig_limit = sys.getrecursionlimit()
        if limit > orig_limit:
            logger.info(
                "sys.setrecursionlimit(%s) since seqlen=%s is too long",
                limit, seqlen)
            try:
                sys.setrecursionlimit(limit)
                yield
            finally:
                logger.info(
                    "sys.setrecursionlimit(%s)"
                    " --- resetting back to the original limit",
                    orig_limit)
                sys.setrecursionlimit(orig_limit)
            return

    yield
