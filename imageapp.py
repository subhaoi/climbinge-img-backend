from flask import *  
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import click
import requests
from io import BytesIO
from pathlib import Path
import pickle
from PIL import Image as pil_img
import numpy as np
from fastai.vision.data import ImageDataBunch
from fastai.vision.transform import get_transforms
from fastai.vision.learner import create_cnn
from fastai.vision import models
from fastai.vision.image import pil2tensor, Image
import matplotlib
import matplotlib.pyplot as plt
import cv2
from imutils import resize
import os
import gc

app = Flask(__name__)  
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
@limiter.limit("10/minute", override_defaults=False)
def success():  
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename) 
        print(f.filename)
        find_similar_images('images',f.filename, './static/output.png', False, 1)
        os.remove(f.filename)
        return render_template("success.html", name = f.filename)  


def find_similar_images(input_path, img_url, output_path, show_image, n_items):
    '''
    Function to run everything together
    '''
    resp, url_img = download_img_from_url(img_url)
    if resp:
        print("Load databunch")
        data_bunch = load_image_databunch(input_path, classes)

        print("Create a model")
        learner = load_model(data_bunch, models.resnet34, "stg2-rn34")

        print("Add a Hook")
        sf = SaveFeatures(learner.model[1][5])

        print("Load LSH table")
        lsh = pickle.load(open(Path(input_path) / "lsh.p", "rb"))
        print(lsh,output_path)
        print("Return similar items")
        get_similar_images(
            url_img, learner, sf, lsh, show_image, output_path, n_items=n_items
        )

    else:
        print(
            "Image cannot be downloaded from URL please check the url link and try again."
        )


def load_image_databunch(input_path, classes):
    """
    Code to define a databunch compatible with model
    """
    tfms = get_transforms(
        do_flip=False,
        flip_vert=False,
        max_rotate=0,
        max_lighting=0,
        max_zoom=1,
        max_warp=0,
    )

    data_bunch = ImageDataBunch.single_from_classes(
        Path(input_path), classes, ds_tfms=tfms, size=224
    )

    return data_bunch


def load_model(data_bunch, model_type, model_name):
    """
    Function to create and load pretrained weights of convolutional learner
    """
    learn = create_cnn(data_bunch, model_type, pretrained=False)
    learn.load(model_name)
    return learn


def download_img_from_url(url):
    '''
    Function to download image given a valid url
    '''
    try:
        img = pil_img.open(url)
        print(img)
        resp = True
    except:
        resp = False
        img = np.nan
    return resp, img


class SaveFeatures:
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))

    def remove(self):
        self.hook.remove()


def image_to_vec(url_img, hook, learner):
    '''
    Function to convert image to vector
    '''
    print("Convert image to vec")
    _ = learner.predict(Image(pil2tensor(url_img, np.float32).div_(255)))
    vect = hook.features[-1]
    return vect


def get_similar_images(
    url_img, conv_learn, hook, lsh, show_image, output_path, n_items=5
):
    ## Converting Image to vector
    vect = image_to_vec(url_img, hook, conv_learn)

    ## Finding approximate nearest neighbours using LSH
    response = lsh.query(vect, num_results=n_items + 1, distance_func="hamming")

    ## Dimension calculation for plotting
    columns = 3
    rows = int(np.ceil(n_items + 1 / columns)) + 1

    gc.collect()
    ## Plotting function
    fig = plt.figure(figsize=(2 * rows, 3 * rows))
    for i in range(1, columns * rows + 2):
        ## Plotting the url_img in center of first row
        if i == 1:
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(url_img)
            plt.axis("off")
            plt.title("Input Image")
        ## Plotting similar images row 2 onwards
        elif i < n_items + 2:
            ret_img = pil_img.open(response[i - 1][0][1])
            fig.add_subplot(rows, columns, i + 2)
            plt.imshow(ret_img)
            plt.axis("off")
            plt.title(response[i - 1][0][1])
            print(response[i - 1][0][1])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)

    ## Display if show_image is mentioned in argument
    if show_image:
        img = cv2.imread(output_path, 1)
        img = resize(img, width=600)
        cv2.imshow("Similar images output", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


classes = ['Arjuns route ', 'Avathi PhD Crack', 'Avathi PhD Straightup', 'Avathi PhD Warmup', 'Avathi cinema', 'Avathi cinema traverse', 'Avathi special', 'Avathi special slab', 'Beginner Warmup', 'Beginner slab', 'Bharat pls name', 'Boulder Project next to Manbun', 'Butterfly crack', 'Bye Bye Boulder', 'Crack near Yabadabadoo', 'Crack next to Magic Cookie', 'Crimpy as hell', 'Dark Slab', 'Defying sanity', 'Dhag dhag', 'Dragon Wings', 'Easy Crack', 'Easy Layback', 'Easy Slab', 'Easy high slab', 'Easy offwidth', 'Easy overhang', 'Easy step slab', 'Easy traverse', 'Elephant Routes 1', 'Elephant Routes 2', 'Evening climb', 'Evening slab', 'Evening steps', 'Fly in the mouth', 'Gujju Short overhang', 'Gujju pls name', 'High Slab', 'Jump Route', 'Kiddy Slab', "Likhit's Birthday", 'Lizard Traverse', 'Loose motion', 'Machine Gun side route', 'Magic Cookie', 'Man Bun', 'Mighty slab', 'Mini sitstart', 'Nightlife', 'Obi Wan', 'Overhang nightlife', 'Power of Boti', 'Project next to Magic Cookie', 'Project next to Yabadabadoo', 'Raja Huli Slab', 'Raja Huli Warm up', 'Samosa chutney', 'Sit start', 'Sitstart', 'Slab next to Manbun', 'Sunset undercut', 'TBC - Bharat Route', 'TBC - Bharat balancy Route', 'The boxer guy', 'The invincible hands', 'Traverse behind samosa', 'Tree Route', 'Tricky Slab', 'Unnamed top of Avathi', 'Wall traverse', 'Warm up', 'Warm up slab', 'Workshop Warmup', 'Yabadabadoo', 'random']


if __name__ == '__main__':  
    app.run(debug = True)  
