import numpy as np
import PIL
from io import BytesIO
import base64
import tensorflow as tf


# probably best to move this to a separate pyfile
def to_data_uri(img):
    data = BytesIO()
    img.save(data, "JPEG") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8')

def display_image(image):
    output = tf.constant(image)
    output = tf.image.convert_image_dtype(output, tf.uint8)
    return PIL.Image.fromarray(output.numpy(), 'RGB')
    