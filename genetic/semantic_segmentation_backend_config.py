from copy import deepcopy
import math
import rastervision as rv

from genetic.semantic_segmentation_backend import (
    SemanticSegmentationBackend)
from genetic.simple_backend_config import (
    SimpleBackendConfig, SimpleBackendConfigBuilder)

GP_SEMANTIC_SEGMENTATION = 'GP_SEMANTIC_SEGMENTATION'


class TrainOptions():
    def __init__(self, num_generations=None, pop_size=None, band_count=None, debug=False):
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.band_count = band_count
        self.debug = debug

    def __setattr__(self, name, value):
        if name in ['band_count', 'pop_size', 'num_generations']:
            value = int(math.floor(value)) if isinstance(value, float) else value
        super().__setattr__(name, value)


class SemanticSegmentationBackendConfig(SimpleBackendConfig):
    train_opts_class = TrainOptions
    backend_type = GP_SEMANTIC_SEGMENTATION
    backend_class = SemanticSegmentationBackend


class SemanticSegmentationBackendConfigBuilder(SimpleBackendConfigBuilder):
    config_class = SemanticSegmentationBackendConfig

    def _applicable_tasks(self):
        return [rv.SEMANTIC_SEGMENTATION]

    def with_train_options(
            self,
            band_count=8,
            num_generations=100,
            pop_size=25,
            debug=False,):
        b = deepcopy(self)
        b.train_opts = TrainOptions(
            num_generations=num_generations,
            pop_size=pop_size,
            band_count=band_count,
            debug=debug)
        return b

    def with_pretrained_uri(self, pretrained_uri):
        """pretrained_uri should be uri of exported model file."""
        # "exported model file" is just a text representation of the algebraic function.
        return super().with_pretrained_uri(pretrained_uri)


# ? Need to change names here.
def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(
        rv.BACKEND, GP_SEMANTIC_SEGMENTATION,
        SemanticSegmentationBackendConfigBuilder)
