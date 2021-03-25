import numpy as np
import PIL
from io import BytesIO
import base64
import tensorflow as tf
import librosa
from pathlib import Path


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
      oads mp3 file, gets magnitude mel spectrogram, 
      flattens mel and reduces dimensionality of particular mel
      Returns a flattened numpy array that represents the mFcc
    '''
    mp3 = Path(mp3_filename)
    y, sr = librosa.load(mp3)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db.flatten()
    mel_db = np.reshape(mel_db, (1, -1))

    return mel_db


def generate_vectors():
    
    '''Generates noise vectors as inputs for each frame'''

    '''
    We can probably turn this into a class
    WAV refers to librosa loaded song along with sr
    fpm refers
    USE MEL to get max values then add pulse noise list to initialized vectors list


    # Get number of noise vectors to initialize (based on speed_fpm)
    num_init_noise = round(
        librosa.get_duration(self.wav, 
                             self.sr)/60*self.speed_fpm)
    
    # Initialize vectors
      init_noise = [self.truncation * \
                    truncnorm.rvs(-2, 2, 
                                  size=(self.batch_size, self.input_shape)) \
                             .astype(np.float32)[0]\
                    for i in range(num_init_noise)]

      # Compute number of steps between each pair of vectors
      steps = int(np.floor(len(self.spec_norm_class))/len(init_noise)- 1)
    


    # Interpolate
      noise = full_frame_interpolation(init_noise, 
                                       steps,
                                       len(self.spec_norm_class))
    '''

    pass