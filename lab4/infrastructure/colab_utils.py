import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import glob
import io
import base64
import os
import shutil
from IPython.display import HTML
from IPython import display as ipythondisplay

## modified from https://colab.research.google.com/drive/1flu31ulJlgiRL1dnN2ir8wGh9p7Zij2t#scrollTo=TCelFzWY9MBI

def show_video():
  mp4list = glob.glob('./video/*.mp4')
  print(mp4list)
  if len(mp4list) > 0:
    mp4 = mp4list[0]
    video = io.open(mp4, 'r+b').read()
    encoded = base64.b64encode(video)
    ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
  else: 
    print("Could not find video")
    

def wrap_env(env):
  # env = RecordVideo(env, '/content/video')
  # return env
  video_path = './video'
  # Manually 'force' the directory to be clean
  if os.path.exists(video_path):
      shutil.rmtree(video_path)
  
  # We pass only the required arguments to avoid TypeErrors
  env = RecordVideo(env, video_folder=video_path, episode_trigger=lambda x: True)
  return env