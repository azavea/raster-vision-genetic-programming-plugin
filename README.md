# Raster Vision DEAP Genetic Programming Plugin

This plugin uses [DEAP](https://github.com/deap/deap) to implement a semantic segmentation backend plugin for [Raster Vision](https://rastervision.io/). The plugin operates by using Genetic Programming to evolve raster algebra formulas to perform specific detection tasks.

## Setup and Requirements

### Docker
You'll need `docker` (preferably version 18 or above) installed. After cloning this repo, to build the Docker images, run the following command:

```shell
> docker/build
```

Before running the container, set an environment variable to a local directory in which to store data.
```shell
> export RASTER_VISION_DATA_DIR="/path/to/data"
```
To run a Bash console in the Docker container, invoke:
```shell
> docker/run
```
This will mount the following local directories to directories inside the container:
* `$RASTER_VISION_DATA_DIR -> /opt/data/`
* `genetic/ -> /opt/src/genetic/`
* `examples/ -> /opt/src/examples/`

### Debug Mode

For debugging, it can be helpful to use a local copy of the Raster Vision source code rather than the version baked into the Docker image. To do this, you can set the `RASTER_VISION_REPO` environment variable to the location of the main repo on your local filesystem. If this is set, `docker/run` will mount `$RASTER_VISION_REPO/rastervision` to `/opt/src/rastervision` inside the container. You can then modify your local copy of Raster Vision in order to debug experiments running inside the container.

### Setup profile

Using the plugin requires making a Raster Vision profile which points to the location of the plugin module. You can make such a profile by creating a file at `~/.rastervision/plugin` containing something like the following.

```
[PLUGINS]
files=[]
modules=["genetic.semantic_segmentation_backend_config"]
```

## Running an experiment

To test the plugin, you can run an [experiment](examples/vegas.py) using the SpaceNet Vegas Buildings dataset. A test run can be executed locally using something like the following. The `-p plugin` flag says to use the `plugin` profile created above.

```
export RASTER_VISION_REPO=/path/to/raster-vision
export RASTER_VISION_DATA_DIR=/path/to/SpaceNet_Buildings_Competition_Round2_Sample
```
```
export RAW_URI="/opt/data/AOI_2_Vegas_Train"
export PROCESSED_URI="/opt/data/genetic/vegas/processed-data"
export ROOT_URI="/opt/data/genetic/vegas/local-output"
rastervision -p plugin run local -e examples.semantic_segmentation.vegas_buildings -a raw_uri $RAW_URI -a processed_uri $PROCESSED_URI -a root_uri $ROOT_URI -a test True --splits 2
```

The train output folder will then contain a text file with the best-performing formula for calculating the segmentation task.
