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

    render (bool): if flag is provided, use on-screen rendering during playback

    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_camera_names (str or [str]): camera name(s) / image observation(s) to
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_camera_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_camera_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_camera_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_camera_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""
import numpy as np
np.random.seed(1111)
import torch
torch.manual_seed(1111)

from IPython import embed
from copy import deepcopy
import os
import json
import h5py
import argparse
import imageio

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase, EnvType

import robosuite
from robosuite.controllers import load_controller_config
from robosuite.utils import transform_utils

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
    render_camera_names=None,
    return_obs=True,
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
        return_obs (bool): if True, return possibly high-dimensional observations along the trajectoryu.
            They are excluded by default because the low-dimensional simulation states should be a minimal
            representation of the environment.

    """
    assert isinstance(env, EnvBase)

    write_video = (video_writer is not None)
    video_count = 0

    # load the initial state
    print("RESET  ENV")
    env.reset()
    print("===================")
    model_str = initial_state['model']
    for key, item in SITE_MAPPER.items():
        model_str = model_str.replace(key, item)
    initial_state['model'] = model_str
    ## No "site" with name gripper0_ee_x exists.
    print("RESET  ENV TO INITiAL STATE")
    states = initial_state['states']
    env.reset_to({'states':states})
    env_target.reset()

    # set target env to be same for objects
    for name in env_target.env.sim.model.joint_names:
        if 'world' not in name and 'robot' not in name and 'gripper' not in name:
            qpos_val = env.env.sim.data.get_joint_qpos(name)
            to_jt = env_target.env.sim.model.get_joint_qpos_addr(name)
            env_target.env.sim.data.qpos[to_jt[0]:to_jt[1]] = qpos_val
    print("===================")

    traj_len = actions.shape[0]
    o = env.get_observation()
    #target_obs = env_target.env._get_observations()
    target_obs = env_target.get_observation()
    target_eef_pos = target_obs['robot0_eef_pos']
    target_eef_quat = target_obs['robot0_eef_quat']

    video_count = 0  # video frame counter
    total_reward = 0.
    target_state_dict = deepcopy(env_target.get_state())


    initial_target_state_dict = deepcopy(env_target.get_state())
    initial_state_dict = deepcopy(env.get_state())

    print(target_state_dict)
    traj = dict(actions=[], rewards=[], dones=[], states=[], initial_state_dict=target_state_dict)
    if return_obs:
        # store observations too
        traj.update(dict(obs=[], next_obs=[]))

    for i in range(traj_len):
        o,r,d,_ = env.step(actions[i])
        #from_image = deepcopy(env.render(mode="rgb_array", height=512, width=512, camera_name='frontview'))
        #from_image = deepcopy(env.render(mode="rgb_array", height=84, width=84, camera_name='frontview'))
        next_eef_pos = o['robot0_eef_pos']
        next_eef_quat = o['robot0_eef_quat']

        target_action_pos = next_eef_pos - target_eef_pos
        target_action_ori = transform_utils.quat2axisangle(transform_utils.quat_distance(next_eef_quat, target_eef_quat))
        target_action_grip = actions[i,6:] # get action/s
        target_action = np.hstack((target_action_pos, target_action_ori, target_action_grip))
        next_target_obs,target_reward,target_done,_ = env_target.step(target_action)
        next_target_eef_pos =  next_target_obs['robot0_eef_pos']
        next_target_eef_quat = next_target_obs['robot0_eef_quat']

        # compute reward
        total_reward += target_reward
        success = env_target.is_success()["task"]

        # video render
        if write_video:
            if video_count % video_skip == 0:
                #video_img = [from_image]
                video_img = []
                for cam_name in render_camera_names:
                    video_img.append(next_target_obs[cam_name+'_image'])
                    #video_img.append(env_target.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                    #video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        # collect transition
        traj["actions"].append(target_action)
        traj["rewards"].append(target_reward)
        traj["dones"].append(target_done)
        traj["states"].append(target_state_dict["states"])
        if return_obs:
            # Note: We need to "unprocess" the observations to prepare to write them to dataset.
            #       This includes operations like channel swapping and float to uint8 conversion
            #       for saving disk space.
            #traj["obs"].append(ObsUtils.unprocess_obs_dict(target_obs))
            #traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_target_obs))
            traj["obs"].append(deepcopy(target_obs))
            traj["next_obs"].append(deepcopy(next_target_obs))


        target_state_dict = deepcopy(env_target.get_state())
        target_obs = deepcopy(next_target_obs)
        target_eef_pos = deepcopy(next_target_eef_pos)
        target_eef_quat = deepcopy(next_target_eef_quat)
    stats = dict(Return=total_reward, Horizon=(i + 1), Success_Rate=float(success))

    if return_obs:
        # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
        traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
        traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        print("TRYING", k)
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])
    return stats, traj



def playback_dataset(args):
    # some arg checking
    write_video = (args.video_path is not None)

    # Auto-fill camera rendering info if not specified
    if args.render_camera_names is None:
        # We fill in the automatic values
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
        env_type = EnvUtils.get_env_type(env_meta=env_meta)
        args.render_camera_names = DEFAULT_CAMERAS[env_type]

    ## need to make sure ObsUtils knows which observations are images, but it doesn't matter
    ## for playback since observations are unused. Pass a dummy spec here.
    #dummy_spec = dict(
    #    obs=dict(
    #            low_dim=[
    #                     'robot0_joint_pos',
    #                      'robot0_eef_pos', 'robot0_eef_quat',
    #                      'robot0_gripper_qpos',
    #                      'Can_pos', 'Can_quat',
    #                      'Can_to_robot0_eef_pos', 'Can_to_robot0_eef_quat',
    #            ],
    #            rgb=[],
    #        ),
    #)

    dummy_spec = dict(
        obs=dict(
                low_dim=[
                         'robot0_joint_pos', 'robot0_joint_pos_cos',
                          'robot0_joint_pos_sin', 'robot0_joint_vel',
                          'robot0_eef_pos', 'robot0_eef_quat',
                          'robot0_gripper_qpos', 'robot0_gripper_qvel',
                          'Can_pos', 'Can_quat',
                          'Can_to_robot0_eef_pos', 'Can_to_robot0_eef_quat',
                          'robot0_proprio-state', 'object-state'],
                rgb=['frontview_image', 'sideview_image', 'agentview_image', 'birdview_image'],
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
    env_meta_target['env_kwargs']['camera_names'] = ['agentview', 'frontview', 'birdview']
    env_meta_target['env_kwargs']['use_object_obs'] = True
    env_meta_target['env_kwargs']['reward_shaping'] = True
    env_meta_target['env_kwargs']['has_offscreen_renderer'] = True
    env_meta['env_kwargs']['use_camera_obs'] = False
    env_meta['env_kwargs']['use_object_obs'] = True
    env_meta['env_kwargs']['reward_shaping'] = True
    env_meta['env_kwargs']['has_offscreen_renderer'] = False
    env_meta['env_kwargs']['camera_names'] = [] #args.render_camera_names
    print("CREATING ENV FROM METADATA")
    # JRH can't figure out how to render correctly for both envs, therefore,
    # use_camera_obs=False for source env if you want to create a dataset in
    # target_env
    #env = EnvUtils.create_env_from_metadata(env_meta=env_meta,
    #                                        render_offscreen=True,
    #                                        use_image_obs=True)

    env = EnvUtils.create_env_from_metadata(env_meta=env_meta,
                                            render_offscreen=False,
                                            use_image_obs=False)
    env_target = EnvUtils.create_env_from_metadata(env_meta=env_meta_target,
                                                   render_offscreen=True,
                                                   use_image_obs=True)
    print("===================")

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

    # maybe open hdf5 to write rollouts
    write_dataset = (args.dataset_path is not None)
    if write_dataset:
        data_writer = h5py.File(args.dataset_path, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0

    rollout_stats = []


    for ind in range(len(demos)):
        ep = demos[ind]
        print("Playing back episode: {}".format(ep))

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # supply actions if using open-loop action playback
        actions = None
        actions = f["data/{}/actions".format(ep)][()]

        stats, traj = playback_trajectory_with_env(
            env=env,
            env_target=env_target,
            initial_state=initial_state,
            states=states, actions=actions,
            video_writer=video_writer,
            video_skip=args.video_skip,
            render_camera_names=args.render_camera_names,
        )
        rollout_stats.append(stats)

        if write_dataset:
            # store transitions
            ep_data_grp = data_grp.create_group("demo_{}".format(ind))
            ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
            ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
            ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
            ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
            if args.dataset_obs:
                for k in traj["obs"]:
                    ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
                    ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))

            # episode metadata
            if "model" in traj["initial_state_dict"]:
                ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
            total_samples += traj["actions"].shape[0]

    rollout_stats = TensorUtils.list_of_flat_dict_to_dict_of_list(rollout_stats)
    avg_rollout_stats = { k : np.mean(rollout_stats[k]) for k in rollout_stats }
    avg_rollout_stats["Num_Success"] = np.sum(rollout_stats["Success_Rate"])
    print("Average Rollout Stats")
    print(json.dumps(avg_rollout_stats, indent=4))



    f.close()
    if write_video:
        video_writer.close()

    if write_dataset:
        # global metadata
        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env_target.serialize(), indent=4) # environment info
        data_writer.close()
        print("Wrote dataset trajectories to {}".format(args.dataset_path))



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

    # If provided, an hdf5 file will be written with the rollout data
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="(optional) if provided, an hdf5 file will be written at this path with the rollout data",
    )

    # If True and @dataset_path is supplied, will write possibly high-dimensional observations to dataset.
    parser.add_argument(
        "--dataset_obs",
        action='store_true',
        help="include possibly high-dimensional observations in output dataset hdf5 file (by default,\
            observations are excluded and only simulator states are saved)",
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
        "--render_camera_names",
        type=str,
        nargs='+',
        default=['frontview'],
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    args = parser.parse_args()
    playback_dataset(args)
