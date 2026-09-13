#!/usr/bin/env python
import argparse
import sys
import os
import shutil
import zipfile
import time

from torchlight import import_class

from processor.processor import init_seed
from processor.registry import LEGACY_PROCESSORS, PROCESSORS
init_seed(0)

def save_src(target_path):
    code_root = os.getcwd()
    srczip = zipfile.ZipFile('./src.zip', 'w')
    for root, dirnames, filenames in os.walk(code_root):
            for filename in filenames:
                if filename.split('\n')[0].split('.')[-1] == 'py':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'yaml':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
                if filename.split('\n')[0].split('.')[-1] == 'ipynb':
                    srczip.write(os.path.join(root, filename).replace(code_root, '.'))
    srczip.close()
    save_path = os.path.join(target_path, 'src_%s.zip' % time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))
    shutil.copy('./src.zip', save_path)


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
        # save src
        save_src(p.arg.work_dir)

    p.start()
