"""Cosmica - Professional Astrophotography Image Processing"""

import os

os.environ.setdefault('FLEXIBLAS', 'netlib')

import sys


def main():
    from cosmica.ui.app import run_application

    sys.exit(run_application(sys.argv))


if __name__ == "__main__":
    main()
