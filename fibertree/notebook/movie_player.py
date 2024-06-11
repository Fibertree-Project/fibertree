""" Movie Player Module """
#
# Import standard libraries
#
import os
import string
import random
import tempfile
import datetime
import re

from pathlib import Path

#
# Import display classes/utilities
#
from IPython.display import display # to display images
from IPython.display import Image
from IPython.display import HTML
from IPython.display import Javascript
from IPython.display import Video

from base64 import b64encode


class MoviePlayer():

    def __init__(self, canvas, layout=None, filename=None):

        self.canvas = canvas

        self.rand = random.Random()

        #
        # Create filename (if not given)
        #
        if filename is None:
            filename = self._createFilename()

        posix_filename = filename.as_posix()
        self.posix_filename = posix_filename

        canvas.saveMovie(posix_filename) # , layout=layout)

    def display(self,
                width="100%",
                loop=True,
                autoplay=True,
                controls=True,
                center=False):

        posix_filename = self.posix_filename

        # TBD: Actually pay attention to width and centering
        final_width = "" if width is None else " width=\"{0}\"".format(width)
        final_center = "" if not center else " style=\"display:block; margin: 0 auto;\""

        final_loop = "" if not loop else " loop"
        final_autoplay = "" if not autoplay else " autoplay"
        final_controls = "" if not controls else " controls"

        final_attributes = f"{final_loop}{final_autoplay}{final_controls}"

        if 'COLAB_JUPYTER_IP' in os.environ:
            #
            # Running in a Google Colab (note status is ignored)
            #
            video_file = open(posix_filename, "r+b").read()

            video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"

#            video = HTML(f"""<video width=600 controls loop autoplay><source src="{video_url}"></video>""")
            video = HTML(f"""<video width=800 {final_attributes}><source src="{video_url}"></video>""")
            display(video)
            return

        #
        # Running in a regular Jupyter notebook
        #
        video = Video(f"./{posix_filename}",
                      html_attributes=final_attributes,
                      width=800)

        display(video)

    def _createFilename(self):
        """
        Create a filename for the movie

        File will have a timestamp followed by the title of the video
        or a random strin (for empty titles)

        """

        title = self.canvas.title

        #
        # Create tmp directory for movies
        #
        tmpdir = Path("tmp")
        tmpdir.mkdir(mode=0o755, exist_ok=True)
        self.tmpdir = tmpdir

        #
        # Create filename
        #
        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y.%m.%d_%H%M%S")

        if title == "":
            basename = Path(f"{date_time_str}.{self._random_string(10)}.mp4")
        else:
            #
            # Remove illegal characters and replace spaces with underscore and lowercase
            #
            cleaned_title = re.sub(r'[^\w\s-]', '', title)    # Remove illegal characters
            cleaned_title = cleaned_title.replace(' ', '_')   # Replace spaces with underscores
            cleaned_title = cleaned_title.lower()
            #
            # Construct the filename
            #
            basename = Path(f"{date_time_str}.{cleaned_title}.mp4")

        filename = self.tmpdir / basename

        return filename

    def _random_string(self, length):
        return ''.join(self.rand.choice(string.ascii_letters) for m in range(length))
