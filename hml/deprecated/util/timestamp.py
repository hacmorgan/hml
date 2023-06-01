#!/usr/bin/env python3


"""
Learning rate schedules
"""


__author__ = "Hamish Morgan"


from typing import Tuple

import datetime


def iso_condensed() -> str:
    """
    Return the ISO timestamp of the current date and time, without any special
    characters, e.g. "20220709T150245.230325"
    """
    return datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f")
