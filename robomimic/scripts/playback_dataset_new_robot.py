"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use-obs argument.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    use-obs (bool): if flag is provided, visualize trajectories with dataset 
        image observations instead of simulator

    use-actions (bool): if flag is provided, use open-loop action playback 
        instead of loading sim states

    render (bool): if flag is provided, use on-screen rendering during playback
    
    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to 
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

from IPython import embed
import os
import json
import h5py
import argparse
import imageio
import numpy as np

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.envs.env_base import EnvBase, EnvType

import robosuite
from robosuite.controllers import load_controller_config

# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}

SITE_MAPPER = {
             'robot0_ee'  :'gripper0_ee'  ,
             'robot0_ee_x':'gripper0_ee_x',
             'robot0_ee_y':'gripper0_ee_y',
             'robot0_ee_z':'gripper0_ee_z',
           }

def playback_trajectory_with_env(
    env, 
    env_target,
    initial_state, 
    states, 
    actions=None, 
    video_writer=None, 
    video_skip=5, 
    camera_names=None,
    first=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert isinstance(env, EnvBase)

    write_video = (video_writer is not None)
    video_count = 0

    # load the initial state
    print("RESET  ENV")
    env.reset()
    print("===================")
    ## TODO update keys
    model_str = initial_state['model']
    for key, item in SITE_MAPPER.items():
        model_str = model_str.replace(key, item)
    initial_state['model'] = model_str
    ## No "site" with name gripper0_ee_x exists.
    print("RESET  ENV TO INITiAL STATE")
    states = initial_state['states']
    env.reset_to({'states':states})
    env_target.reset()

    for name in env_target.env.sim.model.joint_names:
        if 'world' not in name and 'robot' not in name and 'gripper' not in name:
            qpos_val = env.env.sim.data.get_joint_qpos(name)
            to_jt = env_target.env.sim.model.get_joint_qpos_addr(name)
            env_target.env.sim.data.qpos[to_jt[0]:to_jt[1]] = qpos_val  
    print("===================")

    traj_len = actions.shape[0]
    action_playback = (actions is not None)
    o = env.get_observation()
    ot = env_target.get_observation()
    #o = env.env._get_observations(force_update=True)
    #ot = env_target.env._get_observations(force_update=True)
    last_eef_pos = o['robot0_eef_pos']
    last_target_eef_pos = ot['robot0_eef_pos']


    for i in range(traj_len):
        o,r,d,_ = env.env.step(actions[i])
        #o = env.env._get_observations(force_update=True)
        eef_pos = o['robot0_eef_pos']
        #eef_quat = o['robot0_eef_quat']
        target_action_pos = eef_pos - last_target_eef_pos 
        target_action_ori = np.zeros(3) 
        target_action_grip = actions[i,6:] # get actions
        target_action = np.hstack((target_action_pos, target_action_ori, target_action_grip))    
        ot,rt,dt,_ = env_target.env.step(target_action)
        #ot = env.env._get_observations(force_update=True)
        target_eef_pos = ot['robot0_eef_pos']
        #target_eef_quat = ot['robot0_eef_quat']
        if i < traj_len - 1:
            # check whether the actions deterministically lead to the same recorded states
            state_playback = env.get_state()["states"]
        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env_target.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        last_eef_pos = eef_pos
        last_target_eef_pos = target_eef_pos
        if first:
            break


def playback_trajectory_with_obs(
    traj_grp,
    video_writer, 
    video_skip=5, 
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert image_names is not None, "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["actions"].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k)][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def playback_dataset(args):
    # some arg checking
    write_video = (args.video_path is not None)

    # Auto-fill camera rendering info if not specified
    if args.render_image_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_image_names = DEFAULT_CAMERAS[env_type]

    ## need to make sure ObsUtils knows which observations are images, but it doesn't matter 
    ## for playback since observations are unused. Pass a dummy spec here.
    dummy_spec = dict(
        obs=dict(
                low_dim=["robot0_eef_pos",  'robot0_eef_quat', 'robot0_joint_pos', 'robot0_joint_vel',  'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_proprio-state', 'object-state'],
                rgb=['frontview_image'],
            ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env_meta_target = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
     
    # Choose controller
    control_freq = 5
    controller_file = "jaco_osc_pose_%shz.json"%control_freq
    controller_fpath = os.path.join(
               os.path.split(robosuite.__file__)[0], 'controllers', 'config',
               controller_file)
    assert os.path.exists(controller_fpath)
    env_meta_target['env_kwargs']['robots'][0] = args.target_robot
    env_meta_target['env_kwargs']['controller_configs'] = load_controller_config(custom_fpath=controller_fpath)
    env_meta_target['env_kwargs']['control_freq'] = control_freq
    env_meta_target['env_kwargs']['use_camera_obs'] = True
    env_meta_target['env_kwargs']['use_object_obs'] = True
    env_meta_target['env_kwargs']['reward_shaping'] = True
    env_meta_target['env_kwargs']['has_offscreen_renderer'] = True
    env_meta['env_kwargs']['use_camera_obs'] = True
    env_meta['env_kwargs']['use_object_obs'] = True
    env_meta['env_kwargs']['reward_shaping'] = True
    env_meta['env_kwargs']['has_offscreen_renderer'] = True
    print("CREATING ENV FROM METADATA")
    #env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render_offscreen=write_video)
    #env = robosuite.make(env_name=env_meta['env_name'], **env_meta['env_kwargs'])
    #env_target = robosuite.make(env_name=env_meta['env_name'], **env_meta['env_kwargs'])
    env= EnvUtils.create_env_from_metadata(env_meta=env_meta, render_offscreen=write_video)
    env_target = EnvUtils.create_env_from_metadata(env_meta=env_meta_target, render_offscreen=write_video)
    print("===================")

    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)



    f = h5py.File(args.dataset, "r")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))
        if args.use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)], 
                video_writer=video_writer, 
                video_skip=args.video_skip,
                image_names=args.render_image_names,
                first=args.first,
            )
            continue


        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = None
        actions = f["data/{}/actions".format(ep)][()]

        playback_trajectory_with_env(
            env=env, 
            env_target=env_target,
            initial_state=initial_state, 
            states=states, actions=actions, 
            video_writer=video_writer, 
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            first=args.first,
        )

    f.close()
    if write_video:
        video_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--target-robot",
        type=str,
        default='Jaco', 
        choices = ['Panda', 'Jaco'],
        help="new robot to use for data collection",
    )

    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )
    # Use image observations instead of doing playback using the simulator env.
    parser.add_argument(
        "--use-obs",
        action='store_true',
        help="visualize trajectories with dataset image observations instead of simulator",
    )

    # Playback stored dataset actions open-loop instead of loading from simulation states.
    parser.add_argument(
        "--use-actions",
        action='store_true',
        help="use open-loop action playback instead of loading sim states",
    )

 
    # number of trajectories to playback. If omitted, playback all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are played",
    )


    # Dump a video of the dataset playback to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render trajectories to this video file path",
    )

    # How often to write video frames during the playback
    parser.add_argument(
        "--video_skip",
        type=int,
        default=2,
        help="render frames to video every n steps",
    )

    # camera names to render, or image observations to use for writing to video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default=['frontview'],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # Only use the first frame of each episode
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    args = parser.parse_args()
    playback_dataset(args)
