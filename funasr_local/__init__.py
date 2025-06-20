"""Initialize funasr_local package."""

import os

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "version.txt")
if os.path.exists(version_file):
    with open(version_file, "r") as f:
        __version__ = f.read().strip()
else:
    __version__ = "0.4.4"