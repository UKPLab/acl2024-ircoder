import sys


def input(string=""):
    print(string, file=sys.stdout)

    for line in sys.stdin:
        return line


