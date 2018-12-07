from __future__ import print_function

import os

from .loader import load_gandata


def param_grid(gangrid):
    gridtable = gangrid.to_dataframe()
    gridtable.loc[:, 'path'] = [os.path.dirname(data.main_logpath)
                                for data in gridtable['GANData']]
    del gridtable['GANData']
    return gridtable


def print_gangrid_params(paths, output):
    import pandas
    gangrid = load_gandata(paths)
    gridtable = param_grid(gangrid)
    with pandas.option_context("display.max_rows", None,
                               "display.max_columns", None):
        print(gridtable, file=output)


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=argparse.FileType('w'))
    parser.add_argument('paths', nargs='+')
    ns = parser.parse_args(args)
    print_gangrid_params(**vars(ns))


if __name__ == '__main__':
    main()
