import mujoco.viewer
import pandas as pd
from myosuite.logger.grouped_datasets import Trace as Motion_Trace
import h5py
import cv2
import numpy as np

def simulate_skeleton():
    # create model
    model = mujoco.MjModel.from_xml_path('./myosuite/simhive/myo_model/myoskeleton/myoskeleton.xml')
    data = mujoco.MjData(model)
    # create renderer
    renderer = mujoco.Renderer(model)
    # number of steps
    n_steps = 1000
    # create trace
    trace = Motion_Trace("Motion Trajectories")
    # Simulate and save data
    for i in range(n_steps):
        group_key = 'Frame_' + str(i)
        trace.create_group(group_key)
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        datum_dict = dict(
            time=data.time,
            angular_velocity=data.qvel,
            stem_height=data.geom_xpos,
            image=renderer.render()
        )
        trace.append_datums(group_key=group_key, dataset_key_val=datum_dict)
    trace.save('myo_skeleton_simulation_output.h5', verify_length=True)

def motion2video():
    h5 = h5py.File('myo_skeleton_simulation_output.h5', 'r')
    # List all frames available in the file
    frames = list(h5.keys())
    # Loop through each trial and display its contents
    for frames in frames:
        frame_data = h5[frames]
        image = frame_data['image']
        pixels_array = image[0]
        print('time:', type(pixels_array))
        print(pixels_array.shape)
        # Display the image using OpenCV
        cv2.imshow('Image', pixels_array)
        # Save the image to a file

    h5.close()

if __name__ == "__main__":
    simulate_skeleton()
    motion2video()


