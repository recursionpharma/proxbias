from ast import literal_eval

from setuptools import setup


def _read_version():
    with open("VERSION", "r") as f:
        lines = f.read().splitlines()
        versions = {k: literal_eval(v) for k, v in map(lambda x: x.split("="), lines)}
        return ".".join(versions[i] for i in ["MAJOR", "MINOR", "PATCH"])


setup(name="proxbias", version=_read_version())
