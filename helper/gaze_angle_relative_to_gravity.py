import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import json
import pandas as pd
import sys

def angle_between_vectors(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    
    cos_angle = dot_product / magnitude_product
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def get_gaze_angle_relative_to_gravity(world_gaze):
    """
    This function takes in a list of 3d vectors in the world frame where [x, y, z] == [0, 0, -1] is the direction of gravity
    and returns a list of angles between the gaze vector and the gravity vector.
    
    The gravity vector direction is based on the Pupil Labs Neon world coordinate system.
    Source: https://docs.pupil-labs.com/neon/data-collection/data-streams/#movement-imu-data
    
    :param world_gaze: a list of 3d vectors in the world frame where [x, y, z] == [0, 0, -1] is the direction of gravity
    :return: a list of angles between the gaze vector and the gravity vector
    """
    # Gravity vector based on Pupil Labs Neon world coordinate system
    gravity_vector = np.array([0, 0, -1])
    angles = []
    for gaze_vector in world_gaze:
        angles.append(angle_between_vectors(gaze_vector, gravity_vector))
    return angles

def get_world_gaze(root_dir, gaze_file):
    """
    This function takes in a root directory of the dataset and returns a list of 3d vectors in the world frame where [x, y, z] == [0, 0, -1] is the direction of gravity
    :param root_dir: root directory of the dataset
    :return: a list of 3d vectors in the world frame where [x, y, z] == [0, 0, -1] is the direction of gravity
    """
    # Camera Intrinsics (replace with your values)
    # just grab from scene_camera.json json.load(open('scene_camera.json'))['camera_matrix']
    cam_to_img = np.array(json.load(open(root_dir + 'scene_camera.json'))['camera_matrix'])

    # Distortion coefficients (replace with your values)
    # just grab from scene_camera.json
    distortion = np.array(json.load(open(root_dir + 'scene_camera.json'))['distortion_coefficients'])

    # List of quaternions and their timestamps
    # need to grab from imu.csv
    # pd.from_csv('imu.csv')['quaternion w', 'quaternion x', 'quaternion y', 'quaternion z']
    quats = pd.read_csv(root_dir + 'imu.csv', usecols=['quaternion w', 'quaternion x', 'quaternion y', 'quaternion z']).values
    # pd.from_csv('imu.csv')['timestamp [ns]']
    quat_ts = pd.read_csv(root_dir + 'imu.csv', usecols=['timestamp [ns]']).values.reshape(-1)

    # List of image points and their timestamps
    # grab gaze and remove blinks
    gaze_points = pd.read_csv(root_dir + gaze_file)
    # blinks = pd.read_csv(root_dir + 'blinks.csv')
    # for b in range(len(blinks)):
    #     gaze_points = gaze_points.drop(gaze_points[(gaze_points['timestamp [ns]'] >= blinks['start timestamp [ns]'][b]) & (gaze_points['timestamp [ns]'] <= blinks['end timestamp [ns]'][b])].index)
    
    gaze_ts = gaze_points['timestamp [ns]'].values.reshape(-1)
    gaze_points = gaze_points[['gaze x [px]', 'gaze y [px]']].values
    
    # Define the additional rotation of -102 degrees about the x-axis (cam_to_imu)
    cam_to_imu = Rotation.from_euler('x', -102, degrees=True).inv()

    # Process the gaze vectors
    world_gaze = [] # this will be a list of 3d vectors in the world frame where [x, y, z] == [0, 0, -1] is the direction of gravity
    for i, gaze_t in enumerate(gaze_ts):
        idx = np.searchsorted(quat_ts, gaze_t)
        if idx == len(quat_ts):
            closest_idx = idx - 1
        else:
            closest_idx = idx if idx == 0 or abs(quat_ts[idx] - gaze_t) < abs(quat_ts[idx-1] - gaze_t) else idx - 1

        closest_quaternion = quats[closest_idx]

        # imu_to_world transformation
        imu_to_world = Rotation.from_quat(closest_quaternion).as_matrix()

        gaze_point = gaze_points[i]
        # Undistorting the image point
        undistorted_point = cv2.undistortPoints(
            np.array([gaze_point], dtype=np.float32),
            cameraMatrix=cam_to_img,
            distCoeffs=distortion,
            P=cam_to_img
        )[0][0]

        # Convert to homogeneous coordinates (2D -> 3D)
        # this creates a new axis perpendicular to the image plane
        undistorted_point_homogeneous = np.append(undistorted_point, 1)

        # Apply the inverse of the cam_to_img transformation
        # this gives us the 3D gaze vector in terms of the camera frame
        # there is no need to normalize the vector (by dividing by the third component) 
        # because we are only interested in the gaze *direction* not the gaze *length* (which is ambiguous without depth information)
        img_to_cam = np.linalg.inv(cam_to_img)
        cam_point = img_to_cam.dot(undistorted_point_homogeneous)

        # Apply the cam_to_imu transformation
        # this gives us the gaze/img vector in terms of imu-relative-coordinates
        imu_point = cam_to_imu.apply(cam_point)

        # Apply the imu_to_world transformation
        # this gives us the gaze vector in terms of the world frame (which is relative to gravity)
        world_point = imu_to_world.dot(imu_point)

        world_gaze.append(world_point)

    return world_gaze

def main():
    # check if the data directory is given
    if len(sys.argv) < 2:
        print("Usage: python3 get_gaze.py <saving_dir>")
        sys.exit(1)
    
    # change this to the root directory of your dataset
    root_dir = sys.argv[-1] + '/'
    save_dir = sys.argv[-1] + '/'    
    gaze_file = 'gaze_calibrated.csv'

    # save the world gaze to a csv file
    world_gaze = get_world_gaze(root_dir, gaze_file)
    df = pd.DataFrame(world_gaze, columns=['x', 'y', 'z'])
    df.to_csv(save_dir + 'world_gaze.csv', index=False)
    print('world gaze saved to ' + save_dir + 'world_gaze.csv')

    # save the angle between gaze and gravity to a csv file
    angles = get_gaze_angle_relative_to_gravity(world_gaze)
    df = pd.DataFrame(angles, columns=['angle'])
    df.to_csv(save_dir + 'gaze_angle_relative_to_gravity.csv', index=False)
    print('gaze angle relative to gravity saved to ' + save_dir + 'gaze_angle_relative_to_gravity.csv') 
    

if __name__ == '__main__':
    main()