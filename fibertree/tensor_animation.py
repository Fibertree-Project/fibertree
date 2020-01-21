import numpy
import cv2

from PIL import Image, ImageDraw, ImageFont

from fibertree import Tensor
from fibertree import TensorImage
from fibertree import Fiber
from fibertree.payload import Payload

class TensorAnimation():
    """TensorAnimation"""

    def __init__(self, tensors):
        """__init__"""

        #
        # Conditionally unwrap Payload objects
        #
        self.tensors = []
        self.image_list_per_tensor = []
        for tensor in tensors:
            self.tensors.append(Payload.get(tensor))
            self.image_list_per_tensor.append([])
        
        #
        # Add an initial frame with nothing highlighted (it looks good)
        #
        self.add()


    def add(self, highlighted_coords_per_tensor=None):
        
        #
        # Handle the case where nothing should be highlighted anywhere.
        #
        if highlighted_coords_per_tensor is None:
            highlighted_coords_per_tensor = []
            for n in range(len(self.tensors)):
              highlighted_coords_per_tensor.append([])
        
        assert(len(highlighted_coords_per_tensor) == len(self.tensors))
        
        for n in range(len(self.tensors)):
            tensor = self.tensors[n]
            highlighted_coords = highlighted_coords_per_tensor[n]
            im = TensorImage(tensor, highlighted_coords).im
            self.image_list_per_tensor[n].append(im)


    def finalize(self, filename="tmp.gif"):
    
        #
        # Set all images to the max canvas size to ensure smooth  animations
        # 

        final_dims = []
        for n in range(len(self.tensors)):
            max_width = 0
            max_height = 0
            for image in self.image_list_per_tensor[n]:
                max_height = image.height if (image.height > max_height) else max_height
                max_width  = image.width  if (image.width  > max_width)  else max_width
            final_dims.append((max_width, max_height))


        #
        # Take max of width, but concatenate height
        #
        final_width = 0
        final_height = 0
        flattened_height = []

        for w, h in final_dims:
            final_width = w if w > final_width else final_width
            final_height = final_height + h
            flattened_height.append(final_height)


        #
        # Create empty frames for pasting
        #
        final_images = []
        for image in self.image_list_per_tensor[0]:
            final_images.append(Image.new("RGB", (final_width, final_height), "wheat"))

        #
        # Dump individual frames into the same image so they stay in sync.
        #
        for n in range(len(final_images)):
            for t in range(len(self.tensors)):
                image = self.image_list_per_tensor[t][n]
                x_center = final_width // 2 - (image.width // 2)
                # Start where the last image finished.
                y_final = 0 if t == 0 else flattened_height[t-1]
                final_images[n].paste(image, (x_center, y_final))

    def movie(self, filename):
    
        #
        # Set all images to the max canvas size to ensure smooth  animations
        # 

        final_dims = []
        for n in range(len(self.tensors)):
            max_width = 0
            max_height = 0
            for image in self.image_list_per_tensor[n]:
                max_height = image.height if (image.height > max_height) else max_height
                max_width  = image.width  if (image.width  > max_width)  else max_width
            final_dims.append((max_width, max_height))


        #
        # Take max of width, but concatenate height
        #
        final_width = 0
        final_height = 0
        flattened_height = []

        for w, h in final_dims:
            final_width = w if w > final_width else final_width
            final_height = final_height + h
            flattened_height.append(final_height)
         
        #   
        # Add a little padding at the bottom for when the controls are visible.
        #
        final_height = final_height + 75

        #
        # Create empty frames for pasting
        #
        final_images = []
        for image in self.image_list_per_tensor[0]:
            final_images.append(Image.new("RGB", (final_width, final_height), "wheat"))

        #
        # Dump individual frames into the same image so they stay in sync.
        #
        for n in range(len(final_images)):
            for t in range(len(self.tensors)):
                image = self.image_list_per_tensor[t][n]
                x_center = final_width // 2 - (image.width // 2)
                # Start where the last image finished.
                y_final = 0 if t == 0 else flattened_height[t-1]
                final_images[n].paste(image, (x_center, y_final))

        fourcc = cv2.VideoWriter_fourcc(*"vp09")
        out = cv2.VideoWriter(filename,fourcc, 1, (final_width, final_height))
        
        for image in final_images:
            for duplication_cnt in range(1):
                out.write(cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR))
        out.release()


if __name__ == "__main__":

    a = Tensor("../examples/data/draw-a.yaml")
    b = Tensor("../examples/data/draw-b.yaml")
    anim  = TensorAnimation([a, b])
    anim.add([[0], [0]])
    anim.add([[10], [4]])
    anim.add([[10,40], [4,1]])
    anim.add([[10,40,1], [4,1,0]])
    anim.add([[0], [0]])
    anim.movie()
