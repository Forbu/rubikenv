FROM anibali/pytorch:latest
USER root

RUN pip install wandb
RUN pip install pytorch_lightning
RUN pip install pytest
RUN pip install prettytable
RUN pip install pytest-benchmark
RUN pip install pandas
RUN pip install numpy
RUN pip install gym
RUN pip install tqdm
RUN pip install prettytable

ADD . /app/

WORKDIR /app
RUN python setup.py install