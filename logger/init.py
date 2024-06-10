import logging

from rich.console import Console
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler


class BetterReprHighlighter(ReprHighlighter):
    def __init__(self):
        # 将 1.2s, 1.5s 这种时间字符串识别为数字
        super().__init__()
        self.highlights.append(r"(?P<number>\d+\.\d+s)")


def initialize():
    FORMAT = "%(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler(highlighter=BetterReprHighlighter())])
