"""Make Tensor Module"""

#
# Import standard libraries
#
import logging

import yaml
from pathlib import Path

#
# Import display classes/utilities
#
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

#
# Import fibertree libraries
#
from fibertree import Tensor

#
# Set up logging
#
module_logger = logging.getLogger('fibertree.notebook.tensor_maker')


class TensorMaker():
    """TensorMaker

    This class is used to create sets of Jupyter notebook controls to
    control the creation of random tensors.

    Constructor
    -----------

    Parameters
    ----------

    name: string, default=None
        The name of this tensor maker for saving/restoring control settings

    autoload: Bool, default: False
        Control whether saved values are automatically loaded


    Notes
    -----

    This method only creates tensors filled with integers.

    """

    def __init__(self, name=None, autoload=False):
        """ __init__ """

        #
        # Set up logging
        #
        self.logger = logging.getLogger('fibertree.notebook.tensor_maker')


        #
        # Save parameters
        #
        self.name = name
        self.autoload = autoload

        #
        # Tensor creation variables
        #
        self.controls = {}

        self.rank_ids = {}
        self.variables = {}
        self.reset = {}

        self.colors = ["blue",
                       "green",
                       "orange",
                       "purple",
                       "red",
                       "yellow"]

        self.default2value = {"0": 0,
                              "None": None}

        self.control_names = ["SHAPE",
                              "DENSITY",
                              "INTERVAL",
                              "DEFAULT",
                              "COLOR",
                              "SEED"]

        self.tooltips = {"SHAPE": "The shape of this rank's fibers",
                         "DENSITY": "The probablility that a leaf element will be non-zero",
                         "INTERVAL": "The range of integer values to use, e.g., 0 to N",
                         "DEFAULT": "The 'empty' value for compression, may be NONE for fully-populated tensors",
                         "COLOR": "The color used for displaying this tensor",
                         "SEED": "The random number generator seed"}
        #
        # Create directory for load/save configuration information
        #
        etcdir = Path("etc/tensormaker")
        etcdir.mkdir(mode=0o755, parents=True, exist_ok=True)
        self.etcdir = etcdir



#
# Methods to create interactive controls for specifying the attributes
# of a tensor.
#
# Note: the next set of methods are convenience functions for creating
# interactive controls for tensors with standard names and rank names
#

#
# Matrix multiply (by convention)
#
    def addA_MK(self,
                shape=[16, 8],
                density=0.2,
                interval=5,
                seed=10,
                color="green"):

        t = self.addTensor("A",
                           ["M", "K"],
                           shape=shape,
                           density=density,
                           interval=interval,
                           seed=seed,
                           color=color)

        return t


    def addB_KN(self,
                shape=[8,12],
                density=0.2,
                interval=5,
                seed=10,
                color="blue"):

        t = self.addTensor("B",
                           ["K", "N"],
                           shape=shape,
                           density=density,
                           interval=interval,
                           seed=seed,
                           color=color)

        return t


    #
    # Convolution (by convention)
    #
    def addI_CHW(self,
                 shape=[3,8,8],
                 density=1.0,
                 interval=5,
                 seed=10,
                 color="blue"):

        t = self.addTensor("I",
                           ["C", "H", "W"],
                           shape=shape,
                           density=density,
                           interval=interval,
                           seed=seed,
                           color=color)


        return t


    def addF_KCRS(self,
                  shape=[2,4,3,3],
                  density=1.0,
                  interval=5,
                  seed=10,
                  color="green"):

        t = self.addTensor("F",
                           ["K", "C", "R", "S"],
                           shape=shape,
                           density=density,
                           interval=interval,
                           seed=seed,
                           color=color)

        return t

    #
    # Graphs
    #
    def addG_SD(self,
                shape=[10,10],
                density=0.2,
                interval=5,
                seed=5,
                color="orange"):

        t = self.addTensor("G",
                           ["S", "D"],
                           shape=shape,
                           density=density,
                           interval=interval,
                           seed=seed,
                           color=color)

        return t

#
# Generic method to create interactive controls for specifying the
# attributes of a tensor.
#
    def addTensor(self, name, rank_ids, **kwargs):
        """addTensor

        Create a set of interactive controls for the given tensor name

        Parameters
        ----------

        name: str
            The name of the tensor to be created

        rank_ids: list
            The "rank ids" for the tensor

        shape: list
            The "shape" (i.e., size) of each level of the tree

        density: float
            The probability that leaf elements will not be 0

        interval: integer
            The closed range [0:`interval`] of each value at the leaf
            level of the tree

        default: int or None, default=0
            The default empty value, None means no empty value

        seed: a valid argument for `random.seed`
            A seed to pass to `random.seed`

        Notes
        -----

        All tenors are only populated with integer values.

        The density always is referring to the density of zeros
        irrespective of the `default` empty value for the tensor. Note
        further, that if the `default` is not 0 then setting a non-unit
        density for upper ranks does not make sense.

        """

        #
        # Convert simple kwargs into full label names for:
        #

        kwargs = self._convert_kwargs(name, rank_ids, kwargs)

        self.controls[name] = widgets.Label(value=f"Tensor {name}")

        #
        # Create "shape" controls
        #
        self.rank_ids[name] = rank_ids

        for r in rank_ids:
            vname = r + "_SHAPE"

            if vname in self.controls:
                control = widgets.Text(description=f'Shape {r}:',
                                       value="This shape defined above",
                                       disabled=True,
                                       tooltip=self.tooltips["SHAPE"])

                self.controls[f"{vname}_{name}"] = control
            else:

                control = widgets.IntSlider(description=f'Shape {r}:',
                                            min=1,
                                            max=64,
                                            step=1,
                                            value=kwargs.get(vname, 16),
                                            tooltip=self.tooltips["SHAPE"])

                self.controls[vname] = control


        #
        # Create "density" control
        #
        vname = name + "_DENSITY"

        if vname in self.controls:
            del self.controls[vname]

        control = widgets.FloatSlider(description='Density:',
                                      min=0,
                                      max=1,
                                      step=0.02,
                                      value=kwargs.get(vname, 0.2),
                                      tooltip=self.tooltips["DENSITY"])

        self.controls[vname] = control


        #
        # Create "default" control
        #
        vname = name + "_DEFAULT"

        if vname in self.controls:
            del self.controls[vname]

        val = kwargs.get(vname, 0)
        value = f"{val}"
        options = ["0", "None"]
        if value not in options:
            self.default2value[value] = val
            options = [value] + options


        control = widgets.Dropdown(options=options,
                                   value=value,
                                   description='Default:',
                                   disabled=False,
                                   tooltip=self.tooltips["DEFAULT"])

        self.controls[vname] = control

        #
        # Create "interval" control
        #
        vname = name + "_INTERVAL"

        if vname in self.controls:
            del self.controls[vname]

        control = widgets.IntSlider(description='Interval:',
                                    min=1,
                                    max=100,
                                    step=1,
                                    value=kwargs.get(vname, 5),
                                    tooltip=self.tooltips["INTERVAL"])

        self.controls[vname] = control


        #
        # Create "color" control
        #
        vname = name + "_COLOR"

        if vname in self.controls:
            del self.controls[vname]

        self.controls[vname] = widgets.Dropdown(options=self.colors,
                                                value=kwargs.get(vname, "red"),
                                                description='Color:',
                                                disabled=False,
                                                tooltip=self.tooltips["COLOR"])

        #
        # Create "seed" control
        #
        vname = name + "_SEED"

        if vname in self.controls:
            del self.controls[vname]

        control = widgets.IntSlider(description='Seed:',
                                    min=0,
                                    max=100,
                                    step=1,
                                    value=kwargs.get(vname, 10),
                                    tooltip=self.tooltips["SEED"])

        self.controls[vname] = control

        #
        # Return the tensor information
        #
        return {'name': name,
                'rank_ids': rank_ids}



    def _convert_kwargs(self, name, rank_ids, kwargs):
        """Canonicalize kwargs

        Convert simple (lowercase) kwargs into expanded key names
        (partially uppercaed) that can be freely mixed with names from
        all tensors being created by the class. This includes adding
        tensor names and rank_ids to the keys names.

        Some examples for the tensor `A`: the `density` keyword will
        be expanded into `A_DENSITY` and the shape of rank `N`, which
        is found in the list from the `shape` keyword, will be
        expanded to `N_SHAPE`.

        These names match the names used in the `self.controls`
        dictionary.

        The keywords supported include:

             - shape
             - density
             - interval
             - seed
             - color
             - default

        """

        new_kwargs = {}

        for key, value in kwargs.items():

            key_upper = key.upper()

            if key_upper in self.control_names:

                if key == "shape":
                    #
                    # Rank shape information is global across all tensors
                    #
                    for rank_id, shape in zip(rank_ids, value):
                        new_kwargs[f"{rank_id}_SHAPE"] = shape

                    continue
                else:
                    #
                    # Other information is per tensor name
                    #
                    new_kwargs[f"{name}_{key_upper}"] = value
                    continue

            #
            # Preserve other global keys
            #
            if key not in kwargs:
                new_kwargs[key] = value

        return new_kwargs


#
# Display all of the interactive controls to set tensor attributes
#
    def displayControls(self):
        """displayControls

        Create and display the interactive controls

        Parameters
        ----------

        None

        """

        #
        # Display the tensor configuration controls
        #
        controls = self._getControls()

        display(controls)

        #
        # Display the reset button
        #
        load = widgets.Button(description='Load',
                              tooltip='Load all controls values from a file')

        load.on_click(lambda arg: self.loadControls())

        store = widgets.Button(description='Store',
                               tooltip='Store all controls to a file')

        store.on_click(lambda arg: self.storeControls())

        reset = widgets.Button(description='Reset',
                               tooltip='Reset all control values to their default state')

        reset.on_click(lambda arg: self.resetControls())


        display(widgets.HBox([load, store, reset]))


    def _getControls(self):

        title = widgets.Label(value="Tensor Creation Controls")

        controls = interactive(self._set_params,
                               Title=title,
                               **self.controls)

        #
        # Collect reset values for all controls
        #
        for name, control in self.controls.items():
            self.reset[name] = control.value

        #
        # Optionally load controls from file
        #
        if self.autoload:
            self.loadControls()

        return controls


    def _set_params(self, **kwargs):

        for variable, value in kwargs.items():
            self.variables[variable] = value




    def storeControls(self):
        """ storeControls """

        filename = self._getFilename()
        if filename is None:
            return

        state = {name: control.value for (name, control) in self.controls.items()}
        state_yaml = yaml.dump(state, Dumper=yaml.SafeDumper)

        with open(filename, "w") as control_file:
            control_file.write(state_yaml)


    def loadControls(self):
        """ loadControls """

        filename = self._getFilename(exists=True)
        if filename is None:
            return

        with open(filename, "r") as control_file:
            state_yaml = control_file.read()

        state = yaml.load(state_yaml, Loader=yaml.SafeLoader)

        for name, value in state.items():
            self.controls[name].value = value


    def resetControls(self):
        """ resetControls """

        for name, control in self.controls.items():
            control.value = self.reset[name]


    def _getFilename(self, exists=False):
        if self.name is None:
            self.logger.warning("No filename specified at init time")
            return None

        filename = self.etcdir / Path(self.name+".yaml")

        if exists and not filename.is_file():
            self.logger.warning(f"Control file ({filename}) does not exist")
            return None

        return filename



#
# Methods to create a tensor from the interactively set attributes
#


    def makeA_MK(self):

        return self.makeTensor("A")



    def makeB_KN(self):

        return self.makeTensor("B")



    def makeTensor(self, name):
        """makeTensor

        Create a tensor from the current interactively set attributes
        for the tensor named `name`.

        Parameters
        ----------

        name: str
            The name of the tensor to be created


        Returns
        -------

        new_tensor: tensor
            The newly created tensor

        """

        rank_ids = self.rank_ids[name]
        shape = [self.variables[r + "_SHAPE"] for r in rank_ids]
        density = (len(rank_ids) - 1) * [1.0] + [self.variables[name + "_DENSITY"]]
        default = self.default2value[self.variables[name + "_DEFAULT"]]
        interval = self.variables[name + "_INTERVAL"]
        color = self.variables[name + "_COLOR"]
        seed = self.variables[name + "_SEED"]

        t = Tensor.fromRandom(name=name,
                              rank_ids=rank_ids,
                              shape=shape,
                              density=density,
                              default=default,
                              interval=interval,
                              color=color,
                              seed=seed)

        return t
