import librosa



# Song data, sample rate data
# Speed fpm data, batch size data
# Generate initial noise vectors from song
# Interpolate between these noise vectors
# Model takes in noise vectors
# Predicted vectors turn into images
# images are compiled as video with song


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



    # Write temporary audio file
    soundfile.write('tmp.wav',wav_output, sr_output)

    # Generate final video
    audio = mpy.AudioFileClip('tmp.wav', fps = self.sr*2)
    video = mpy.ImageSequenceClip(self.frames_dir, 
                                  fps=self.sr/self.frame_duration)
    video = video.set_audio(audio)
    video.write_videofile(file_name,audio_codec='aac')

    # Delete temporary audio file
    os.remove('tmp.wav')


    '''

    pass