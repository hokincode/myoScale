import mujoco.viewer
import pandas as pd
from myosuite.logger.grouped_datasets import Trace as Motion_Trace
import h5py
import cv2
import numpy as np

def simulate_skeleton():
    # enable joint visualization option:
    duration = 3.8  # (seconds)
    framerate = 60  # (Hz)
    # Simulate and display video.
    frames = []
    model = mujoco.MjModel.from_xml_path('./myosuite/simhive/myo_model/myoskeleton/myoskeleton.xml') # create model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model) # create renderer
    trace = Motion_Trace("Motion Trajectories") # create trace
    while data.time < duration: # Simulate and save data
        group_key = 'Frame_' + str(data.time)
        trace.create_group(group_key)
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            datum_dict = dict(
                image=renderer.render()
            )
        trace.append_datums(group_key=group_key, dataset_key_val=datum_dict)
    trace.save('myo_skeleton_simulation_output.h5', verify_length=True)

def motion2video():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change to 'mp4v' codec for MP4 format
    out = cv2.VideoWriter('output_video.mp4', fourcc, 300.0, (320, 240))  # Change the file extension to .mp4
    h5 = h5py.File('myo_skeleton_simulation_output.h5', 'r')  # h5 file setting
    for frame in list(h5.keys()):
        frame_data = h5[frame]
        image = frame_data['image']
        pixels_array = image[0]
        out.write(pixels_array)
    h5.close()
    out.release()

if __name__ == "__main__":
    simulate_skeleton()
    motion2video()