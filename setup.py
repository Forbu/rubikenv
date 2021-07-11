from setuptools import setup
import setuptools

setup(name="rubikenv",
      version="0.1",
      description="Gym env for rubik cube",
      author="Adrien Bufort",
      author_email="adrienbufort@gmail.com",
      packages=setuptools.find_packages(),
      package_dir={"rubikenv": "rubikenv"},
      install_requires=[],
      extras_require={
            "dev": [],
      },
      license="Apache 2.0")