# stdlib
from datetime import datetime, timezone
import os
import sys

# third-party

# local


# every print follows "[{ts()}] [SCOPE] message" - SCOPE says what part
# of the run wrote that line e.g. (SETUP, EXTRACT, FOLD 1/3, etc)
# makes it easy to scan the log


def ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Tee design adapted from: https://shallowsky.com/blog/programming/python-tee.html
class Tee:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()  # flush immediately so a crash doesn't lose lines

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def write_log(scope, message, **kwargs):
    timestamp = ts()
    prefix = f"[{timestamp} - {scope}]"
    context = " | ".join(
        f"{key.replace('_', '.', 1)}={value}" for key, value in kwargs.items()
    )
    if context:
        print(f"{prefix} {message} :: {context}")
    else:
        print(f"{prefix} {message}")


def start_logging(log_dir, name):
    os.makedirs(
        log_dir, exist_ok=True
    )  # here to avoid circular dependency shenanigans: `setup.py` has a log but it also creates the logs folder.
    log_file = f"{log_dir}/{name}-T{int(datetime.now(timezone.utc).timestamp())}.log"
    sys.stdout = Tee(log_file)
    return log_file
