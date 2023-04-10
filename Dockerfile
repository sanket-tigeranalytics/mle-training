FROM continuumio/miniconda3
WORKDIR /docker
LABEL maintainer="Sanket"
COPY deploy/conda/env.yml env.yml
RUN conda env create --file env.yml
RUN activate mle-dev
COPY . .
ENTRYPOINT ["/bin/bash"]
