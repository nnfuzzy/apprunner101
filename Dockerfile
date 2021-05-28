
FROM ubuntu:latest

# System packages 
RUN apt-get update && apt-get install -y curl

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda config --add channels conda-forge
RUN conda install -y streamlit fbprophet xlrd openpyxl
#RUN conda install -y ipython jupyter jupyter-lab papermill
RUN conda install -y typer fsspec s3fs
RUN conda install -y pip
RUN pip install streamlit --upgrade

COPY . /home/
RUN ls -lhrt  /
RUN ls -lhrt  /home/src


WORKDIR /home/src

EXPOSE 8501

# Run the executable
ENTRYPOINT ["streamlit", "run"]
CMD ["forecaster_app.py"]
