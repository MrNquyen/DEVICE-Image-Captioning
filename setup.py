from setuptools import setup, Extension

setup(
    name="phoc_ext",
    packages=["utils.phoc.src"],
    ext_modules=[
        Extension(
            "utils.phoc.src.cphoc",
            ["utils/phoc/src/cphoc.c"],
            # you may need extra compile/link flags here
        )
    ]
)