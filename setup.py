from setuptools import setup

from server_api import __version__

setup(
    name="server_api",
    version=__version__,
    url="https://github.com/lemon-ai-com/server_api",
    author="Lemon AI",
    author_email="dev@lemon-ai.com",
    py_modules=["server_api"],
    install_requires=["requests==2.31.0", "pydantic==2.3.0"],
)
