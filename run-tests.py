#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import re


def get_source_files(testdir):
    files = []
    suffixes = [".c", ".C"]
    for suffix in suffixes:
        files += glob.glob(os.path.join(testdir, "*%s" % suffix))
    files.sort()
    return files


class TestRunner:
    def __init__(self, compiler):
        self.failed_tests = []
        self.compiler = compiler

    def compile(self, file):
        args = [self.compiler]
        args += ["-fplugin=python"]
        args += ["-fplugin-arg-python-script=plugin2.py"]
        args += ["-c"]
        args += ["-otest.o"]
        args += [file]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, errs = p.communicate()
        os.system("rm -f test.o")

        errs = errs.decode("utf-8")
        return errs

    def run_test(self, file, regex):
        res = self.compile(file)
        match = regex.match(res)
        if match:
            print(file + " ==> Success")
        else:
            print(file + " ==> Failure")
        return match

    def run_positive_tests(self):
        files = get_source_files("tests/positive")
        regex = re.compile(r".*Transformation seems to be correct", flags=re.DOTALL)
        for file in files:
            success = self.run_test(file, regex)
            if not success:
                self.failed_tests.append(file)

    def run_negative_tests(self):
        files = get_source_files("tests/negative")
        regex = re.compile(r".*Transformation is not correct", flags=re.DOTALL)
        for file in files:
            success = self.run_test(file, regex)
            if not success:
                self.failed_tests.append(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("compiler", help="the compiler to use")
    args = parser.parse_args()

    tr = TestRunner(args.compiler)
    tr.run_positive_tests()
    tr.run_negative_tests()
    print(f"Nof failures: {len(tr.failed_tests)}")
    for file in tr.failed_tests:
        print(f"  {file}")


if __name__ == "__main__":
    main()
