import re
import random
import os
from os.path import join

import rastervision as rv
from rastervision.utils.files import list_paths

from genetic.semantic_segmentation_backend_config import (
    GP_SEMANTIC_SEGMENTATION
)


class VegasBuildings(rv.ExperimentSet):
    def exp_main(self, raw_uri, root_uri):
        """Run an experiment on the Spacenet Vegas building dataset.

        This is a simple example of how to do semantic segmentation on data that
        doesn't require any pre-processing or special permission to access.

        Args:
            raw_uri: (str) directory of raw data (the root of the Spacenet dataset)
            root_uri: (str) root directory for experiment output
        """
        raster_uri = join(raw_uri, 'MUL')
        label_uri = join(raw_uri, 'geojson/buildings')
        raster_fn_prefix = 'MUL_AOI_2_Vegas_img'
        label_fn_prefix = 'buildings_AOI_2_Vegas_img'
        label_paths = list_paths(label_uri, ext='.geojson')
        label_re = re.compile(r'.*{}(\d+)\.geojson'.format(label_fn_prefix))
        scene_ids = [
            label_re.match(label_path).group(1)
            for label_path in label_paths]

        random.seed(5678)
        scene_ids = sorted(scene_ids)
        random.shuffle(scene_ids)
        # Workaround to handle scene 1000 missing on S3.
        if '1000' in scene_ids:
            scene_ids.remove('1000')
        num_train_ids = int(len(scene_ids) * 0.8)
        train_ids = scene_ids[0:num_train_ids]
        val_ids = scene_ids[num_train_ids:]

        exp_id = 'spacenet-simple-seg'
        chip_size = 162

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(chip_size) \
                            .with_classes({
                                'Building': (1, 'orange'),
                                'Background': (2, 'black')
                            }) \
            .with_chip_options(
                chips_per_scene=1,
                debug_chip_probability=0.25,
                negative_survival_probability=1.0,
                target_classes=[1],
                target_count_threshold=1000) \
            .build()

        config = {
            'band_count': 8,
            'num_generations': 50,
            'pop_size': 250,
            'num_individuals': 125,
            'num_offspring': 125,
            'mutation_rate': 0.3,
            'crossover_rate': 0.5,
            'debug': True
        }

        backend = rv.BackendConfig.builder(GP_SEMANTIC_SEGMENTATION) \
                                  .with_task(task) \
                                  .with_train_options(**config) \
                                  .build()

        def make_scene(id):
            train_image_uri = os.path.join(raster_uri,
                                           '{}{}.tif'.format(raster_fn_prefix, id))

            raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                .with_uri(train_image_uri) \
                .with_stats_transformer() \
                .build()

            vector_source = os.path.join(
                label_uri, '{}{}.geojson'.format(label_fn_prefix, id))
            label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                .with_vector_source(vector_source) \
                .with_rasterizer_options(2) \
                .build()

            label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                .with_raster_source(label_raster_source) \
                .build()

            scene = rv.SceneConfig.builder() \
                .with_task(task) \
                .with_id(id) \
                .with_raster_source(raster_source) \
                .with_label_source(label_source) \
                .build()

            return scene

        train_scenes = [make_scene(id) for id in train_ids]
        val_scenes = [make_scene(id) for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .build()

        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                    .build()

        # Need to use stats_analyzer because imagery is uint16.
        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_analyzer(analyzer) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
