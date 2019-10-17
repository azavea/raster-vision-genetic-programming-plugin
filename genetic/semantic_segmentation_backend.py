import operator
from os.path import join, basename, dirname
import uuid
import zipfile
import glob
import multiprocessing
from pathlib import Path
import random
import math
from functools import partial

import numpy as np

import rasterio

from rastervision.utils.files import (
    get_local_path, make_dir, upload_or_copy, list_paths,
    download_if_needed, sync_from_dir, sync_to_dir, str_to_file)
from rastervision.utils.misc import save_img
from rastervision.backend import Backend
from rastervision.data.label import SemanticSegmentationLabels

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from genetic.utils import apply_to_raster, fitness


# DEAP infrastructure setup
# Define new functions
def protectedDiv(left, right):
    if right == 0:
        return 1
    return left / right


def protectedSqrt(val):
    return math.sqrt(abs(val))


def protectedLog10(val):
    if val == 0:
        return 0  # Returning infinity just pollutes everything
    return math.log10(abs(val))


def gt(left, right):
    if left > right:
        return 1
    return 0


def lt(left, right):
    if left < right:
        return 1
    return 0


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# This has to go last, otherwise the above items won't be found when pickling.
pool = multiprocessing.Pool()


class SemanticSegmentationBackend(Backend):
    def __init__(self, task_config, backend_opts, train_opts):
        self.task_config = task_config
        self.backend_opts = backend_opts
        self.train_opts = train_opts
        self.raster_func = None
        self._pset = gp.PrimitiveSet("MAIN", self.train_opts.band_count)
        self.chip_dir = None

        # Set up toolbox with evolution configuration.
        # TODO: Is this the best place to configure this? Can we auto-detect?
        self._pset = gp.PrimitiveSet("MAIN", self.train_opts.band_count)
        # TODO: Make these configurable (?)
        self._pset.addPrimitive(operator.add, 2)
        self._pset.addPrimitive(operator.sub, 2)
        self._pset.addPrimitive(operator.mul, 2)
        self._pset.addPrimitive(protectedDiv, 2)
        self._pset.addPrimitive(operator.neg, 1)
        self._pset.addPrimitive(math.cos, 1)
        self._pset.addPrimitive(math.sin, 1)
        self._pset.addPrimitive(protectedLog10, 1)
        self._pset.addPrimitive(protectedSqrt, 1)
        self._pset.addPrimitive(math.floor, 1)
        self._pset.addPrimitive(math.ceil, 1)
        self._pset.addPrimitive(round, 1)

        # Multiprocessing
        self._toolbox = base.Toolbox()

        self._toolbox.register("map", pool.map)
        # TODO: Make these configurable.
        self._toolbox.register("expr", gp.genHalfAndHalf, pset=self._pset, min_=1, max_=2)
        self._toolbox.register("individual", tools.initIterate,
                               creator.Individual, self._toolbox.expr)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)
        self._toolbox.register("compile", gp.compile, pset=self._pset)

        self._toolbox.register("select", tools.selTournament, tournsize=3)
        self._toolbox.register("mate", gp.cxOnePoint)
        self._toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self._toolbox.register("mutate", self.mut_random_operator)

        self._toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=5)
        )
        self._toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=5)
        )

    # Four methods to override: Two for making chips, train, load_model, predict.
    def print_options(self):
        print('Backend options')
        print('--------------')
        for k, v in self.backend_opts.__dict__.items():
            print('{}: {}'.format(k, v))
        print()

        print('Train options')
        print('--------------')
        for k, v in self.train_opts.__dict__.items():
            print('{}: {}'.format(k, v))
        print()

    def save_tiff(self, pixels, path):
        """Use rasterio to write data to a file at path."""
        # TODO: The transform parameter is just to get rasterio to stop complaining that the dataset
        # is unreferenced. Ideally, we should be using something else to do reading and writing
        # since none of the images have georeferencing information. However, I've found indications
        # that other libraries don't do well with non-RGB image files so I've stuck with rasterio
        # for now.
        with rasterio.open(
            path,
            'w',
            driver='GTiff',
            width=pixels.shape[0],
            height=pixels.shape[1],
            count=pixels.shape[2],
            dtype=str(pixels.dtype),
            transform=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ) as ds:
            ds.write(np.transpose(pixels, (2, 0, 1)))

    def process_scene_data(self, scene, data, tmp_dir):
        """Process each scene's training data.

        This writes {scene_id}/img/{scene_id}-{ind}.png and
        {scene_id}/labels/{scene_id}-{ind}.png

        Args:
            scene: Scene
            data: TrainingData

        Returns:
            backend-specific data-structures consumed by backend's
            process_sceneset_results
        """
        # ? Overall, what's the role of this function in the pipeline?
        # Takes in one Scene at a time. A Scene has a labelsource, a rastersource, and an ID
        # TrainingData is a list of tuples (chip, window, labels)
        # ? What are the formats and functionality of Scene and TrainingData
        # ? Are there any restrictions on what this should output or can I structure it however is
        # most convenient?
        # This is given the raw chunks of a scene (chips) and then it is responsible for writing
        # them out into files in a way that the training process is eventually going to be able to
        # use.
        # tmp_dir is a path
        scene_dir = join(tmp_dir, str(scene.id))
        img_dir = join(scene_dir, 'img')
        labels_dir = join(scene_dir, 'labels')

        make_dir(img_dir)
        make_dir(labels_dir)

        # A window is a box data structure, it's a bounding box. In pixel coordinates.
        # A chip is the numpy array containing raster data. It can be sliced out of a larger scene,
        # and then the window gives you the offsets of where that chip comes from in the larger
        # scene.
        # Labels has more than just the window, but chip and window should be aligned.
        for ind, (chip, window, labels) in enumerate(data):
            chip_path = join(img_dir, '{}-{}.tif'.format(scene.id, ind))
            label_path = join(labels_dir, '{}-{}.tif'.format(scene.id, ind))

            label_im = labels.get_label_arr(window).astype(np.uint8)
            save_img(label_im, label_path)
            self.save_tiff(chip, chip_path)

        return scene_dir

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """After all scenes have been processed, process the result set.

        This writes a zip file for a group of scenes at {chip_uri}/{uuid}.zip
        containing:
        train-img/{scene_id}-{ind}.png
        train-labels/{scene_id}-{ind}.png
        val-img/{scene_id}-{ind}.png
        val-labels/{scene_id}-{ind}.png

        Args:
            training_results: dependent on the ml_backend's process_scene_data
            validation_results: dependent on the ml_backend's
                process_scene_data
        """
        # This is responsible for aggregating the results of chipping several Scenes into a zip
        # file. Takes in the results from process_scene and makes a zip file out of them. Can do
        # whatever it needs to in order for the training process to be able to access the data.
        # Can probably avoid touching these, unless I decide to use 8-band data. In that case would
        # need to write out tiffs or some other multiband format. Compressed numpy arrays work
        # pretty well.
        # ? Overall, what's the role of this function in the pipeline?
        # ? Can you give examples of what training_results and validation_results might look like?
        # Does this mean rasters or vectors or accuracy metrics? What calls this?
        if self.train_opts.debug:
            self.print_options()

        group = str(uuid.uuid4())
        group_uri = join(self.backend_opts.chip_uri, '{}.zip'.format(group))
        group_path = get_local_path(group_uri, tmp_dir)
        make_dir(group_path, use_dirname=True)

        with zipfile.ZipFile(group_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            def _write_zip(results, split):
                for scene_dir in results:
                    scene_paths = glob.glob(join(scene_dir, '**/*.tif'))
                    for p in scene_paths:
                        zipf.write(p, join(
                            '{}-{}'.format(
                                split,
                                dirname(p).split('/')[-1]),
                            basename(p)))
            _write_zip(training_results, 'train')
            _write_zip(validation_results, 'val')

        upload_or_copy(group_path, group_uri)

    # ? What state of the world does this rely on? What can it assume exists? What should it return?
    # ? What should this return?
    # Basically, all this relies on is that process_sceneset_data has been run on all groups of
    # Scenes and that those zip files are available for download. This is responsible for
    # downloading those files, unzipping them, and then using them as appropriate to perform
    # training.
    def train(self, tmp_dir):
        """Train a model."""
        self.print_options()

        # Sync output of previous training run from cloud.
        # This will either be local or S3. This allows restarting the job if it has been shut down.
        train_uri = self.backend_opts.train_uri
        train_dir = get_local_path(train_uri, tmp_dir)
        make_dir(train_dir)
        sync_from_dir(train_uri, train_dir)

        # Get zip file for each group, and unzip them into chip_dir.
        self.chip_dir = join(tmp_dir, 'chips')
        make_dir(self.chip_dir)

        train_chip_dir = self.chip_dir + '/train-img'
        train_truth_dir = self.chip_dir + '/train-labels'
        fitness_func = partial(fitness, train_chip_dir, train_truth_dir, self._toolbox.compile)
        self._toolbox.register("evaluate", fitness_func)
        # This is the key part -- this is how it knows where to get the chips from.
        # backend_opts comes from RV, and train_opts is where you can define backend-specific stuff.
        for zip_uri in list_paths(self.backend_opts.chip_uri, 'zip'):
            zip_path = download_if_needed(zip_uri, tmp_dir)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(self.chip_dir)

        # Setup data loader.
        def get_label_path(im_path):
            return Path(str(im_path.parent)[:-4] + '-labels') / im_path.name

        class_map = self.task_config.class_map
        classes = class_map.get_class_names()
        if 0 not in class_map.get_keys():
            classes = ['nodata'] + classes

        # Evolve
        # Set up hall of fame to track the best individual
        hof = tools.HallOfFame(1)

        # Set up debugging
        mstats = None
        if self.train_opts.debug:
            stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
            stats_size = tools.Statistics(len)
            mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
            mstats.register("averageaverage", np.mean)
            mstats.register("stdeviation", np.std)
            mstats.register("minimumstat", np.min)
            mstats.register("maximumstat", np.max)

        pop = self._toolbox.population(n=self.train_opts.pop_size)
        pop, log = algorithms.eaMuPlusLambda(
            pop,
            self._toolbox,
            self.train_opts.num_individuals,
            self.train_opts.num_offspring,
            self.train_opts.crossover_rate,
            self.train_opts.mutation_rate,
            self.train_opts.num_generations,
            stats=mstats,
            halloffame=hof,
            verbose=self.train_opts.debug
        )

        # ? What should my model output be given that the output is just a string? Should I output a
        # text file?
        # RV uses file-presence based caching to figure out whether a stage has completed (kinda
        # like Makefiles). So since this outputs a file every epoch, it needs to use something else
        # to trigger done-ness.
        # Since model is exported every epoch, we need some other way to
        # show that training is finished.
        if self.train_opts.debug:
            print(str(hof[0]))
        str_to_file(str(hof[0]), self.backend_opts.train_done_uri)

        # Sync output to cloud.
        sync_to_dir(train_dir, self.backend_opts.train_uri)

    def mut_random_operator(self, individual):
        """Randomly select a mutation operator and apply it."""
        mutations = [
            partial(gp.mutUniform, individual, expr=self._toolbox.expr_mut, pset=self._pset),
            partial(gp.mutShrink, individual),
            partial(gp.mutNodeReplacement, individual, pset=self._pset),
            partial(gp.mutInsert, individual, pset=self._pset),
            partial(gp.mutEphemeral, individual, mode='one')
        ]
        operator = random.choice(mutations)
        return operator()

    # Takes in a text file containing a Python expression and compiles it to Python code.
    # Compiled function stored as self.raster_func.
    def load_model(self, tmp_dir):
        """Load the model in preparation for one or more prediction calls."""
        if self.raster_func is None:
            self.print_options()
            model_uri = self.backend_opts.model_uri
            model_path = download_if_needed(model_uri, tmp_dir)
            with open(model_path, 'r') as func_file:
                func_str = func_file.read()
                print('FUNC', func_str)
                parsed_func = self._toolbox.compile(expr=func_str)
                self.raster_func = parsed_func

    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.

        Return:
            Labels object containing predictions
        """
        # chips and windows both are arrays, but they always only contain one element (we think)
        # Labels should be an array of your classifications (integers), in the same shape as the
        # chip.  RV assumes that the class IDs start at 1, 0 is an "ignore" class, so it shouldn't
        # be included during training. They need to be integers. So I need to do the snapping here
        # (and it should happen in the training too, in the same way).
        self.load_model(tmp_dir)
        func_chips = [np.transpose(chip, (2, 0, 1)) for chip in chips]
        # The expected output shape is only one band with the same length and width dimensions
        label_arr = apply_to_raster(
            self.raster_func,
            func_chips[0],
            (chips[0].shape[0], chips[0].shape[1])
        )
        print(label_arr.shape, label_arr.dtype, str(label_arr))

        # Return "trivial" instance of SemanticSegmentationLabels that holds a single
        # window and has ability to get labels for that one window.
        # This is designed to access the prediction for a particular window lazily.
        # If the data isn't huge this is just a pass through for the results, which is what is
        # happening here.
        def label_fn(_window):
            if _window == windows[0]:
                return label_arr  # Do the prediction in here.
            else:
                raise ValueError('Trying to get labels for unknown window.')
        return SemanticSegmentationLabels(windows, label_fn)
