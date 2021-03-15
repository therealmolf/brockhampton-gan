from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL
from io import BytesIO
import base64
from inference.helpers import display_image, to_data_uri, flattened_pca
from inference.deep import create_model
import re


progan = hub.load("https://tfhub.dev/google/progan-128/1").signatures['default']


posts = [
    {
        'author': 'Your Mum',
        'title': 'BlogPost1',
        'content': 'Geh geh geh',
        'date_posted': 'December 11, 1999'
    },
    {
        'author': 'Soytiet',
        'title': 'BlogPost2',
        'content': '41 42 43 44 4fiiiive',
        'date_posted': 'December 11, 1998'
    }
]


def home(request):
    context = {
        'posts': posts
    }
    return render(request, 'inference/home.html', context)

def generate_image(request):
    file_obj = request.FILES['filepath']
    print(file_obj)
    fs = FileSystemStorage()
    filepath_name = fs.save(file_obj.name,file_obj)
    filepath_name = fs.url(filepath_name)
    # hacky format thing I had to do
    filepath_name = re.sub("%20", " ", filepath_name)
    filepath_name = "/home/therealmolf/projects/gango" + filepath_name

    # filepath_name = "/home/therealmolf/projects/gango/media/glaive - dnd (audio).mp3"


    model = create_model()
    
    vector = flattened_pca(filepath_name)
    output = model(vector)[0]
    pil_img = display_image(output)
    music_uri = to_data_uri(pil_img)

    # print(filepath_name)

    context = {'filepath_name': filepath_name,
                'music_uri': music_uri}
    # print(request.POST)
    return render(request, 'inference/home.html', context)

def progan_image(request):
    v1 = tf.random.normal([512])
    output = progan(v1)['default'][0]
    pil_img = display_image(output)
    image_uri = to_data_uri(pil_img)

    context = {'image_uri': image_uri}
    return render(request, 'inference/home.html', context)

def about(request):
    return render(request, 'inference/about.html')

