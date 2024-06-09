import time

import rich

console = rich.get_console()


class Timer:
    def __init__(self, name="Time taken"):
        self.start = time.time()
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type or exc_value or traceback:
            console.log(f"{self.name}: Exception occurred.")

        self.end = time.time()
        self.interval = self.end - self.start
        console.log(f"{self.name}: {self.interval:.2f} seconds.")
