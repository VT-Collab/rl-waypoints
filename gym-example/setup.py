from setuptools import setup

setup(name="gym_example",
      version="0.1",
      author="Collab",
      packages=["gym_example", "gym_example.envs"],
      install_requires = ["gym", "numpy"]
)
