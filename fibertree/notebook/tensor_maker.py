"""Make Tensor Module"""

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from fibertree import Tensor

class TensorMaker():

    def __init__(self):
        """ __init__ """

        #
        # Tensor creation variables
        #
        self.controls = {}
        self.controls["Title"] = widgets.Label(value="Tensor Creation Controls")

        self.rank_ids = {}
        self.variables = {}

        self.colors = ["blue",
                       "green",
                       "orange",
                       "purple",
                       "red",
                       "yellow"]



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

        # Note repeated rank names will only appear once

        self._getControls()

        display(self.controls)


    def _getControls(self):
        self.controls = interactive(self._set_params, **self.controls)

        return self.controls


    def _set_params(self, **kwargs):

        for variable, value in kwargs.items():
            self.variables[variable] = value

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



