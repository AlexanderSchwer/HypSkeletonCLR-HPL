#!/usr/bin/env python
import argparse
import sys

from torchlight import import_class

from processor.processor import init_seed
from processor.registry import LEGACY_PROCESSORS, PROCESSORS
init_seed(0)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    processors = {name: import_class(path) for name, path in PROCESSORS.items()}

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    if arg.processor in LEGACY_PROCESSORS:
        print(
            f"WARNING: Processor '{arg.processor}' is legacy and is kept for reference and reproduction.",
            file=sys.stderr,
        )

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    if p.arg.phase == 'train':
        p.save_src()

    p.start()
