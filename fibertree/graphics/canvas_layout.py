"""Canvas Layout Module"""

import re

class CanvasLayout():
    """CanasLayout

    A class to create a layout for a set of tensors in a canvas. This
    class is used by the `MovieCanvas` and SpacetimeCanvas classes to find
    the layout characteristics of each row of tensors on the canvas image.

    Constructor
    -----------

    Parameters
    ----------
    image_list_per_tensor: list of lists
        A list for each tensor where each item of the list is a list of images
        of that tensor - one for each frame of the output

    layout: list
        A list of the number of tensors to display in each row of the canvas

    """
    
    def __init__(self, image_list_per_tensor, layout):
        """__init__"""

        #
        # Save tensor images and layout
        #
        self.image_list_per_tensor  = image_list_per_tensor

        self.layout = layout

    def getLayout(self, start, end):
        
        return self._calcShapes(start,end)
    
    def _calcShapes(self, start, end):

        layout = self.layout

        tensor_dims = self._calcTensorSizes()

        #
        # Set up layout pattern
        #
        if isinstance(layout, str):
            # Define the regex pattern to match "auto:<number>"
            pattern = r'auto:(\d+)'
    
            # Search for the pattern in the input string
            match = re.search(pattern, layout)
    
            if match:
                # Extract the number and convert it to an integer
                frame_width = int(match.group(1))
                layout = self._autoLayout(tensor_dims, frame_width)
            else:
                layout = []
                
        layout_tensors = sum(layout)
        total_tensors = len(self.image_list_per_tensor)
        
        # Add rows to cover all tensors
        if layout_tensors  < total_tensors:
            layout.extend([1] * (total_tensors - layout_tensors))

        #
        # Given a layout determine the:
        #    1) final width of image for all frames
        #    2) final height of image for all frames
        #    3) widths of the regions used by each tensor in each row
        #
        final_width = 0
        final_height = 0
        row_shapes = []

        current_tensor = 0
        
        for row_length in layout:

            if current_tensor >= len(tensor_dims):
                break

            #
            # Extract out tensor dimensions for tensors in this row
            #
            row_dims = tensor_dims[current_tensor:current_tensor+row_length]

            #
            # Calculate the width/height of row
            # Record width of each tensor in the row
            # Track maximum width and total height
            #
            row_width = 0
            row_height = 0
            row_widths = []
            
            for tensor_width, tensor_height in row_dims:
            
                row_width += tensor_width
                row_height = max(row_height, tensor_height)

                row_widths.append(tensor_width)


            final_width = max(final_width, row_width)
            final_height += row_height
            row_shapes.append([row_width, row_height, row_widths])

            current_tensor += row_length

        #
        # Add a little padding at the bottom for when the controls are visible.
        #
        final_height = final_height + 75

        return (final_width, final_height, row_shapes)
    
    
    def _calcTensorSizes(self):
        """_calcTensorSizes"""

        #
        # For each tensor find the image of that tensor with the maximum width and height
        #
        tensor_dims = []

        for t in range(len(self.image_list_per_tensor)):
            max_width = 0
            max_height = 0
            for image in self.image_list_per_tensor[t]:
                max_height = image.height if (image.height > max_height) else max_height
                max_width  = image.width  if (image.width  > max_width)  else max_width

            tensor_dims.append((max_width, max_height))

        return tensor_dims


    def _autoLayout(self, tensor_dims, frame_width=1500):
        """_autoLayout"""

        layout = []

        #
        # Put first tensor as first tensor in first row_count
        #
        layout.append(1)
        row_width = tensor_dims[0][0]

        #
        # For remaining tensors, add them to row if they fit
        # otherwise start a new row
        #
        for width, _ in tensor_dims[1:]:

            if row_width + width < frame_width:
                layout[-1] += 1
                row_width = row_width + width
            else:
                layout.append(1)
                row_width = width

        return layout
        
