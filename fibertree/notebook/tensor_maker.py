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
        """ Create the set of interactive controls for the given tensor """

        #
        # Convert simple kwargs into full label names for:
        #

        kwargs = self._convert_kwargs(name, rank_ids, kwargs)

        self.controls[name] = widgets.Label(value=f"Tensor {name}")

        self.rank_ids[name] = rank_ids

        for r in rank_ids:
            vname = r+"_SHAPE"

            if vname in self.controls:
                self.controls[f"{vname}_{name}"] = widgets.Text(description=f'Shape {r}:',
                                                                value="This shape defined above",
                                                                disabled=True)
            else:
                self.controls[vname]=widgets.IntSlider(description=f'Shape {r}:',
                                                       min=1,
                                                       max=64,
                                                       step=1,
                                                       value=kwargs.get(vname, 16))

        vname = name+"_DENSITY"

        if vname in self.controls:
            del self.controls[vname]

        self.controls[vname]=widgets.FloatSlider(description='Density:',
                                                 min=0,
                                                 max=1,
                                                 step=0.02,
                                                 value=kwargs.get(vname, 0.2))


        vname = name+"_INTERVAL"

        if vname in self.controls:
            del self.controls[vname]

        self.controls[vname]=widgets.IntSlider(description='Interval:',
                                               min=1,
                                               max=100,
                                               step=1,
                                               value=kwargs.get(vname, 5))

        vname = name+"_SEED"

        if vname in self.controls:
            del self.controls[vname]

        self.controls[vname]=widgets.IntSlider(description='Seed:',
                                               min=0,
                                               max=100,
                                               step=1,
                                               value=kwargs.get(vname, 10))


        vname = name+"_COLOR"

        if vname in self.controls:
            del self.controls[vname]

        self.controls[vname]=widgets.Dropdown(options=self.colors,
                                              value=kwargs.get(vname, "red"),
                                              description='Color:',
                                              disabled=False)
        
        return {'name':     name,
                'rank_ids': rank_ids}



    def _convert_kwargs(self, name, rank_ids, kwargs):
        """ Canonicalize kwargs
                        
        
        Convert simple kwargs into full label names for:
             - shape
             - density
             - interval
             - seed
             - color
        
        """

        new_kwargs = {}

        for key, value in kwargs.items():
            if key == "shape":
                for rank_id, shape in zip(rank_ids, value):
                    new_kwargs[f"{rank_id}_SHAPE"] = shape

                continue

            if key == "density":
                new_kwargs[f"{name}_DENSITY"] = value
                continue

            if key == "seed":
                new_kwargs[f"{name}_SEED"] = value
                continue

            if key == "interval":
                new_kwargs[f"{name}_INTERVAL"] = value
                continue

            if key == "color":
                new_kwargs[f"{name}_COLOR"] = value
                continue

            if not key in kwargs:
                new_kwargs[key] = value

        return new_kwargs


#
# Display all of the interactive controls to set tensor attributes
#

    def displayControls(self):
        """Create and display the interactive controls"""

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
        """ Create a tensor from the current interactively set attributes """ 

        rank_ids = self.rank_ids[name]

        t = Tensor.fromRandom(name=name,
                              rank_ids=rank_ids,
                              shape=[self.variables[r+"_SHAPE"] for r in rank_ids],
                              density=(len(rank_ids)-1)*[1.0]+[self.variables[name+"_DENSITY"]],
                              interval=self.variables[name+"_INTERVAL"],
                              seed=self.variables[name+"_SEED"],
                              color=self.variables[name+"_COLOR"])

        return t



