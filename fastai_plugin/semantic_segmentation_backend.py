import os
import sys
import operator
from os.path import join, basename, dirname
import uuid
import zipfile
import glob
from pathlib import Path
import random
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from rastervision.utils.files import (
    get_local_path, make_dir, upload_or_copy, list_paths,
    download_if_needed, sync_from_dir, sync_to_dir, str_to_file)
from rastervision.utils.misc import save_img
from rastervision.backend import Backend
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.data.label_source.utils import color_to_triple

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from fastai_plugin.utils import zipdir

matplotlib.use("Agg")


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


# Creates a representation of the input and the labels overlaid, it's RGB + input labels, labels
# colorized based on color map.
def make_debug_chips(data, class_map, tmp_dir, train_uri, debug_prob=1.0):
    # TODO get rid of white frame
    if 0 in class_map.get_keys():
        colors = [class_map.get_by_id(i).color
                  for i in range(len(class_map))]
    else:
        colors = [class_map.get_by_id(i).color
                  for i in range(1, len(class_map) + 1)]
        # use grey for nodata
        colors = ['grey'] + colors
    colors = [color_to_triple(c) for c in colors]
    colors = [tuple([x / 255 for x in c]) for c in colors]
    cmap = matplotlib.colors.ListedColormap(colors)

    def _make_debug_chips(split):
        debug_chips_dir = join(tmp_dir, '{}-debug-chips'.format(split))
        zip_path = join(tmp_dir, '{}-debug-chips.zip'.format(split))
        zip_uri = join(train_uri, '{}-debug-chips.zip'.format(split))
        make_dir(debug_chips_dir)
        ds = data.train_ds if split == 'train' else data.valid_ds
        for i, (x, y) in enumerate(ds):
            if random.uniform(0, 1) < debug_prob:
                plt.axis('off')
                plt.imshow(x.data.permute((1, 2, 0)).numpy())
                plt.imshow(y.data.squeeze().numpy(), alpha=0.4, vmin=0,
                           vmax=len(colors), cmap=cmap)
                plt.savefig(join(debug_chips_dir, '{}.png'.format(i)),
                            figsize=(3, 3))
                plt.close()
        zipdir(debug_chips_dir, zip_path)
        upload_or_copy(zip_path, zip_uri)

    _make_debug_chips('train')
    _make_debug_chips('val')


class SemanticSegmentationBackend(Backend):
    def __init__(self, task_config, backend_opts, train_opts):
        self.task_config = task_config
        self.backend_opts = backend_opts
        self.train_opts = train_opts
        # ? What's this?
        # This is set by load_model.
        # Four methods to override: Two for making chips, train, load_model, predict.
        self.inf_learner = None

    def print_options(self):
        # TODO get logging to work for plugins
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
            chip_path = join(img_dir, '{}-{}.png'.format(scene.id, ind))
            label_path = join(labels_dir, '{}-{}.png'.format(scene.id, ind))

            label_im = labels.get_label_arr(window).astype(np.uint8)
            save_img(label_im, label_path)
            save_img(chip, chip_path)

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
        self.print_options()

        group = str(uuid.uuid4())
        group_uri = join(self.backend_opts.chip_uri, '{}.zip'.format(group))
        group_path = get_local_path(group_uri, tmp_dir)
        make_dir(group_path, use_dirname=True)

        with zipfile.ZipFile(group_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            def _write_zip(results, split):
                for scene_dir in results:
                    scene_paths = glob.glob(join(scene_dir, '**/*.png'))
                    for p in scene_paths:
                        zipf.write(p, join(
                            '{}-{}'.format(
                                split,
                                dirname(p).split('/')[-1]),
                            basename(p)))
            _write_zip(training_results, 'train')
            _write_zip(validation_results, 'val')

        upload_or_copy(group_path, group_uri)

    def subset_training_data(self, chip_dir):
        """ Specify a subset of all the training chips that have been created

        This creates uses the train_opts 'train_count' or 'train_prop' parameter to
            subset a number (n) of the training chips. The function prioritizes
            'train_count' and falls back to 'train_prop' if 'train_count' is not set.
            It creates two new directories 'train-{n}-img' and 'train-{n}-labels' with
            subsets of the chips that the dataloader can read from.

        Args:
            chip_dir (str): path to the chip directory

        Returns:
            (str) name of the train subset image directory (e.g. 'train-{n}-img')
        """
        # Allows you to choose a subset of the training chips to actually get used.
        # This is called by the train method, this isn't RV-specific.
        # ? Overall, what's the role of this function in the pipeline?
        # ? What's the dataloader here? Do I need to modify that?

        all_train_uri = join(chip_dir, 'train-img')
        all_train = list(filter(lambda x: x.endswith(
            '.png'), os.listdir(all_train_uri)))
        all_train.sort()

        count = self.train_opts.train_count
        if count:
            if count > len(all_train):
                raise Exception('Value for "train_count" ({}) must be less '
                                'than or equal to the total number of chips ({}) '
                                'in the train set.'.format(count, len(all_train)))
            sample_size = int(count)
        else:
            prop = self.train_opts.train_prop
            if prop > 1 or prop < 0:
                raise Exception('Value for "train_prop" must be between 0 and 1, got {}.'.format(prop))
            if prop == 1:
                return 'train-img'
            sample_size = round(prop * len(all_train))

        random.seed(100)
        sample_images = random.sample(all_train, sample_size)

        def _copy_train_chips(img_or_labels):
            all_uri = join(chip_dir, 'train-{}'.format(img_or_labels))
            sample_dir = 'train-{}-{}'.format(str(sample_size), img_or_labels)
            sample_dir_uri = join(chip_dir, sample_dir)
            make_dir(sample_dir_uri)
            for s in sample_images:
                upload_or_copy(join(all_uri, s), join(sample_dir_uri, s))
            return sample_dir

        for i in ('labels', 'img'):
            d = _copy_train_chips(i)

        return d

    def fitness(self, individual, train_files):
        """
        Return a score representing the fitness of a particular program.

        Params individual: The individual to be evaluated.  train_files: A list of tuples of the
        form (raster, geojson), where raster is a filename of a GeoTIFF containing multiband raster
        data, and geojson is the name of a GeoJSON file representing ground-truth.
        """
        # Take a random choice from the possible train data to test against for this iteration We
        # need to get at least some mixture of classes in order to train. A good, dense image has
        # about 50 buildings (houses), so add images to the evaluation set until we have 50
        # buildings.
        eval_choices = assemble_eval_data(train_files, desired_features=100)

        total_error = 0
        func = toolbox.compile(expr=individual)
        for input_file, truth_file in eval_choices:
            # Rasterize JSON and cache raster
            # TODO: Shouldn't have to do this anymore, everything comes in rasterized
            with rasterio.open(RASTER_DIR + input_file, 'r') as input_ds:
                truth_pixels = rasterize_geojson(input_ds, truth_file)
                try:
                    output = apply_to_raster(func, input_ds, truth_pixels.shape)
                except (ValueError, OverflowError):
                    print(individual)
                    return (sys.float_info.max,)
            #print('output', output[0, 0], output[50, 50])
            #print('truth', truth_pixels[0, 0], truth_pixels[50, 50])
            errors = output - truth_pixels
            # Return sum of squared classification errors
            #total_error += np.sum(np.square(errors))
            # Return mean squared error
            total_error += np.mean(np.square(errors))
            #print(individual.__str__(), total_error)
        # if str(individual) in eval_cache:
        #     assert eval_cache[str(individual)] == total_error
        # else:
        #     eval_cache[str(individual)] = total_error
        return (total_error,)

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
        chip_dir = join(tmp_dir, 'chips')
        make_dir(chip_dir)
        # This is the key part -- this is how it knows where to get the chips from.
        # backend_opts comes from RV, and train_opts is where you can define backend-specific stuff.
        for zip_uri in list_paths(self.backend_opts.chip_uri, 'zip'):
            zip_path = download_if_needed(zip_uri, tmp_dir)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(chip_dir)

        # Setup data loader.
        def get_label_path(im_path):
            return Path(str(im_path.parent)[:-4] + '-labels') / im_path.name

        size = self.task_config.chip_size
        class_map = self.task_config.class_map
        classes = class_map.get_class_names()
        if 0 not in class_map.get_keys():
            classes = ['nodata'] + classes

        train_img_dir = self.subset_training_data(chip_dir)

        data = None  # TODO
        print(data)

        if self.train_opts.debug:
            make_debug_chips(data, class_map, tmp_dir, train_uri)

        # Setup GP.
        # Setup callbacks and train model.
        model_path = get_local_path(self.backend_opts.model_uri, tmp_dir)

        pretrained_uri = self.backend_opts.pretrained_uri
        if pretrained_uri:
            print('Loading weights from pretrained_uri: {}'.format(
                pretrained_uri))
            pretrained_path = download_if_needed(pretrained_uri, tmp_dir)
            # TODO: Do something with pretrained_path (read file and parse)

        # Evolve
        # Set up toolbox with evolution configuration.
        # TODO: Make these configurable
        pset = gp.PrimitiveSet("MAIN", BAND_COUNT)
        pset.addPrimitive(operator.add, 2)
        pset.addPrimitive(operator.sub, 2)
        pset.addPrimitive(operator.mul, 2)
        pset.addPrimitive(protectedDiv, 2)
        pset.addPrimitive(operator.neg, 1)
        pset.addPrimitive(math.cos, 1)
        pset.addPrimitive(math.sin, 1)
        pset.addPrimitive(protectedLog10, 1)
        pset.addPrimitive(protectedSqrt, 1)

        pset.addEphemeralConstant("rand101", lambda: random.randint(0, 65535))

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        # TODO: Make these configurable.
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        toolbox.register("evaluate", self.evalSymbReg, train_files=train_files)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", mut_random_operator)

        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
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

        pop = toolbox.population(n=self.train_opts.pop_size)
        pop, log = algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            self.train_opts.pop_size,  # TODO: Add parameter for generation size
            self.train_opts.pop_size,  # TODO: Add parameter for new individual count (I think?)
            0.5,  # TODO: Add parameter for mutation rate (?)
            0.4,  # TODO: Add param for crossover rate (?)
            30,  # TODO: Add param for this, I don't remember what it is
            stats=mstats,  # TODO: Make sure the non-debug case works
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
        # TODO: I'm doing both but I'd rather not used train_done_uri if I don't have to.
        str_to_file(hof[0], self.backend_opts.train_done_uri)
        str_to_file(hof[0], self.backend_opts.train_uri)

        # Sync output to cloud.
        sync_to_dir(train_dir, self.backend_opts.train_uri)

    # ? What is this expected to do? Does it need to set any particular attributes on self?
    # This would take in the text file and parse it to a Python expression.
    def load_model(self, tmp_dir):
        """Load the model in preparation for one or more prediction calls."""
        if self.raster_func is None:
            self.print_options()
            model_uri = self.backend_opts.model_uri
            model_path = download_if_needed(model_uri, tmp_dir)
            self.raster_func = self.load_function(  # TODO: Implement self.load_function()
                dirname(model_path), basename(model_path)
            )

    # ? What is the structure of a Labels object?
    # ? What is the structure of windows and how am I expected to use them? One prediction per
    # window?
    # chips and windows both are arrays, but they always only contain one element (we think)
    # Labels should be an array of your classifications (integers), in the same shape as the chip.
    # RV assumes that the class IDs start at 1, 0 is an "ignore" class, so it shouldn't be included
    # during training. They need to be integers. So I need to do the snapping here (and it should
    # happen in the training too, in the same way).
    def predict(self, chips, windows, tmp_dir):
        """Return predictions for a chip using model.

        Args:
            chips: [[height, width, channels], ...] numpy array of chips
            windows: List of boxes that are the windows aligned with the chips.

        Return:
            Labels object containing predictions
        """
        self.load_model(tmp_dir)
        # TODO: Call apply_to_raster with self.raster_func on chips[0] to generate label_arr
        label_arr = []

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
