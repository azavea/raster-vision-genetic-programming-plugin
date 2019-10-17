FROM ubuntu:18.04

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntugis/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        wget git \
        python3 python3-setuptools python3-pip python3-protobuf python3-tk python3-dev \
        gdal-bin \
        libproj-dev \
        proj-bin \
        jq \
        unzip \
        build-essential libsqlite3-dev zlib1g-dev && \
    apt-get autoremove && apt-get autoclean && apt-get clean

RUN pip3 install Cython==0.29.13 wheel==0.33.4
RUN pip3 install --no-cache-dir --no-binary="pyproj" deap==1.2.2 ptvsd==4.2.* protobuf==3.8.0 opencv-python==4.1.0.25 pyproj==1.9.5.1
RUN pip3 install git+git://github.com/ddohler/raster-vision.git@99a9aaef9dd2040ee1feffe450a5d1e74f325674

# Install Tippecanoe
RUN cd /tmp && \
    wget https://github.com/mapbox/tippecanoe/archive/1.32.5.zip && \
    unzip 1.32.5.zip && \
    cd tippecanoe-1.32.5 && \
    make && \
    make install

# Setup GDAL_DATA directory, rasterio needs it.
ENV GDAL_DATA=/usr/share/gdal/

# Set WORKDIR and PYTHONPATH
WORKDIR /opt/src/
ENV PYTHONPATH=/opt/src:$PYTHONPATH
# In Ubuntu, python3 is always and forever python3, not python, but RV expects to find a "python" executable
RUN ln -s /usr/bin/python3 /usr/bin/python

# Needed for click to work
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# See https://github.com/mapbox/rasterio/issues/1289
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

COPY ./genetic /opt/src/genetic
COPY ./examples /opt/src/examples

ENV PYTHONPATH /opt/src/fastai:$PYTHONPATH

CMD ["bash"]
