FROM ubuntu:16.04

RUN apt-get update && apt-get install -y \
	software-properties-common python-software-properties \
	python3-pip

RUN add-apt-repository ppa:deadsnakes/ppa && \
        apt-get update && \
        apt-get install -y python3.7

RUN add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update && \
    apt-get install -y wget=1.* git=1:2.* python-protobuf=2.* python3-tk=3.* \
                       gdal-bin=2.2.* \
                       jq=1.5* \
                       build-essential libsqlite3-dev=3.11.* zlib1g-dev=1:1.2.* \
                       libopencv-dev=2.4.* python-opencv=2.4.* && \
    apt-get autoremove && apt-get autoclean && apt-get clean

# Setup GDAL_DATA directory, rasterio needs it.
ENV GDAL_DATA=/usr/share/gdal/2.2/

RUN apt-get install -y unzip

# Install Tippecanoe
RUN cd /tmp && \
    wget https://github.com/mapbox/tippecanoe/archive/1.32.5.zip && \
    unzip 1.32.5.zip && \
    cd tippecanoe-1.32.5 && \
    make && \
    make install

# Set WORKDIR and PYTHONPATH
WORKDIR /opt/src/
ENV PYTHONPATH=/opt/src:$PYTHONPATH

# Needed for click to work
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/list

# RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
#      chmod +x ~/miniconda.sh && \
#      ~/miniconda.sh -b -p /opt/conda && \
#      rm ~/miniconda.sh
#
# ENV PATH /opt/conda/bin:$PATH
# RUN conda install -y python=$PYTHON_VERSION
# RUN conda install -y -c pytorch magma-cuda100=2.5 torchvision=0.2
# RUN conda install -y -c fastai fastai=1.0.51
# RUN conda install -y -c conda-forge awscli=1.16.* boto3=1.9.*
# RUN conda install -y jupyter=1.0.*
# RUN conda clean -ya
RUN pip3 install git+git://github.com/azavea/raster-vision.git@3b96214f9a4ce38e08ac78d0a0113353ad02ebf0
RUN pip3 install deap==1.2.2 ptvsd==4.2.*

# See https://github.com/mapbox/rasterio/issues/1289
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

COPY ./genetic /opt/src/genetic
COPY ./examples /opt/src/examples

ENV PYTHONPATH /opt/src/fastai:$PYTHONPATH

CMD ["bash"]
