import numpy as np
import PIL
from io import BytesIO
import base64
import tensorflow as tf


# probably best to move this to a separate pyfile
def to_data_uri(img):
    """
        Returns an HTML-readable image
    """
    data = BytesIO()
    img.save(data, "JPEG") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8')


def display_image(image):
    '''
        Returns a PIL image from the image tensor
    '''
    output = tf.constant(image)
    output = tf.image.convert_image_dtype(output, tf.uint8)
    return PIL.Image.fromarray(output.numpy(), 'RGB')

    
def flattened_pca(mp3_filename):
  '''
    loads mp3 file, gets magnitude mel spectrogram, flattens mel and reduces dimensionality of particular mel
    Returns a flattened numpy array that represents the mFcc
  '''
  y, sr = librosa.load(mp3_filename)
  mel = librosa.feature.melspectrogram(y=y, sr=sr)
  mel_db = librosa.power_to_db(mel, ref=np.max)
  mel_db = mel_db.flatten()
  mel_db = np.reshape(mel_db, (1, -1))

  return mel_db