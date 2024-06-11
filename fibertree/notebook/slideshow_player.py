""" Imaage Player Module """

from IPython.display import display, Javascript
from ipywidgets import widgets, Button, VBox, HBox
from PIL import Image
import io
import asyncio
import nest_asyncio

nest_asyncio.apply()

class SlideshowPlayer:
    def __init__(self, canvas, layout=None):

        self.frames = canvas.getAllFrames(layout=layout)

        self.index = 0
        self.is_playing = False

        # Create button widgets
        self.next_button = Button(description="Next")
        self.back_button = Button(description="Back")
        self.play_button = Button(description="Play")
        self.full_screen_button = Button(description="Full Screen")

        # Attach click event handlers
        self.next_button.on_click(self.next_image)
        self.back_button.on_click(self.prev_image)
        self.play_button.on_click(self.toggle_play)
        self.full_screen_button.on_click(self.full_screen)

        # Create an output widget to display the image
        self.output = widgets.Output()

        # Display the first image initially
        self.show_image()

        # Create a HBox layout to hold the buttons
        self.button_box = HBox([self.back_button,
                                self.next_button,
                                self.play_button,
                                self.full_screen_button])

        # Create a VBox layout to hold the output and the button box
        self.layout = VBox([self.output, self.button_box])

    def show_image(self):
        self.output.clear_output(wait=True)

        with self.output:
            index = self.index

            img = self.frames[self.index]
            display(img)

    def next_image(self, b=None):
        # Move to the next image
        self.index = (self.index + 1) % len(self.frames)
        self.show_image()

    def prev_image(self, b=None):
        # Move to the previous image
        self.index = (self.index - 1) % len(self.frames)
        self.show_image()

    def toggle_play(self, b=None):
        if not self.is_playing:
            self.is_playing = True
            self.play_button.description = "Pause"
            asyncio.ensure_future(self.play_images())
        else:
            self.is_playing = False
            self.play_button.description = "Play"

    async def play_images(self):
        while self.is_playing:
            self.next_image()
            await asyncio.sleep(1)  # Pause for 1 second



    def full_screen(self, b=None):
        display(Javascript('''
            var cell = this.closest('.output');
            var button = this;
            if (cell.requestFullscreen) {
                cell.requestFullscreen();
            } else if (cell.mozRequestFullScreen) { // Firefox
                cell.mozRequestFullScreen();
            } else if (cell.webkitRequestFullscreen) { // Chrome, Safari, and Opera
                cell.webkitRequestFullscreen();
            } else if (cell.msRequestFullscreen) { // IE/Edge
                cell.msRequestFullscreen();
            }
        '''))

    def display(self):
        display(self.layout)
