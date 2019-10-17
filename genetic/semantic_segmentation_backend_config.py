from copy import deepcopy
import math
import rastervision as rv

from genetic.semantic_segmentation_backend import (
    SemanticSegmentationBackend)
from genetic.simple_backend_config import (
    SimpleBackendConfig, SimpleBackendConfigBuilder)

GP_SEMANTIC_SEGMENTATION = 'GP_SEMANTIC_SEGMENTATION'


class TrainOptions():
    def __init__(
            self,
            num_generations=25,
            pop_size=50,
            band_count=8,
            num_individuals=25,
            num_offspring=25,
            mutation_rate=.3,
            crossover_rate=.3,
            debug=False):
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.band_count = band_count
        self.num_individuals = num_individuals
        self.num_offspring = num_offspring
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.debug = debug

    def __setattr__(self, name, value):
        if name in [
                'band_count',
                'pop_size',
                'num_generations',
                'num_offspring',
                'num_individuals']:
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

    def with_train_options(self, **kwargs):
        b = deepcopy(self)
        b.train_opts = TrainOptions(**kwargs)
        return b

    def with_pretrained_uri(self, pretrained_uri):
        """pretrained_uri should be uri of exported model file."""
        # "exported model file" is just a text representation of the algebraic function.
        return super().with_pretrained_uri(pretrained_uri)


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(
        rv.BACKEND, GP_SEMANTIC_SEGMENTATION,
        SemanticSegmentationBackendConfigBuilder)
