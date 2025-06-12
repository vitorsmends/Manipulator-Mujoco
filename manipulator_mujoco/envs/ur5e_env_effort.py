import time
import os
import numpy as np
from dm_control import mjcf
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces
from manipulator_mujoco.arenas import StandardArena
from manipulator_mujoco.robots import Arm
from manipulator_mujoco.mocaps import Target
from manipulator_mujoco.controllers import JointEffortController

class UR5eEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": None,
    }  # TODO add functionality to render_fps

    def __init__(self, render_mode=None):
        # TODO come up with an observation space that makes sense
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )

        # TODO come up with an action space that makes sense
        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=(6,), dtype=np.float64
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self._render_mode = render_mode

        ############################
        # create MJCF model
        ############################
        
        # checkerboard floor
        self._arena = StandardArena()

        # mocap target that OSC will try to follow
        self._target = Target(self._arena.mjcf_model)

        # ur5e arm
        self._arm = Arm(
            xml_path= os.path.join(
                os.path.dirname(__file__),
                '../assets/robots/ur5e/ur5e.xml',
            ),
            eef_site_name='eef_site',
            attachment_site_name='attachment_site'
        )

        # attach arm to arena
        self._arena.attach(
            self._arm.mjcf_model, pos=[0,0,0], quat=[0.7071068, 0, 0, -0.7071068]
        )
       
        # generate model
        self._physics = mjcf.Physics.from_mjcf_model(self._arena.mjcf_model)

        # set up effort controller
        self._controller = JointEffortController(
            physics=self._physics,
            joints=self._arm.joints,
            min_effort=np.full(len(self._arm.joints), -150.0),
            max_effort=np.full(len(self._arm.joints), 150.0),
        )

        # for GUI and time keeping
        self._timestep = self._physics.model.opt.timestep
        self._viewer = None
        self._step_start = None

    def _get_obs(self):
        eef_pos = self._physics.bind(self._arm.eef_site).xpos
        return np.concatenate([
            self._physics.bind(self._arm.joints).qpos,
            self._physics.bind(self._arm.joints).qvel,
            eef_pos,
            self._goal_position,
        ])

    def _get_info(self) -> dict:
        joint_qpos = self._physics.bind(self._arm.joints).qpos
        joint_qvel = self._physics.bind(self._arm.joints).qvel
        joint_tau = self._physics.bind(self._arm.joints).qfrc_applied

        eef_pos = self._physics.bind(self._arm.eef_site).xpos
        eef_vel = self._physics.bind(self._arm.eef_site).xvel
        goal = self._goal_position

        position_error = np.linalg.norm(eef_pos - goal)

        return {
            "joint_positions": joint_qpos.copy(),
            "joint_velocities": joint_qvel.copy(),
            "joint_torques_applied": joint_tau.copy(),
            "eef_position": eef_pos.copy(),
            "eef_velocity": eef_vel.copy(),
            "goal_position": goal.copy(),
            "position_error": position_error,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        with self._physics.reset_context():
            self._physics.bind(self._arm.joints).qpos = [
                0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0,
            ]

            # Alvo aleatório dentro de um intervalo
            x = np.random.uniform(0.4, 0.7)
            y = np.random.uniform(-0.2, 0.2)
            z = np.random.uniform(0.25, 0.4)
            self._goal_position = np.array([x, y, z])

        return self._get_obs(), self._get_info()


    def set_target_pose(self, pose: np.ndarray):
        """
        Atualiza a pose alvo do controlador operacional.

        Args:
            pose (np.ndarray): vetor [x, y, z, qx, qy, qz, qw]
        """
        position = pose[:3]
        quaternion = pose[3:]
        self._target.set_mocap_pose(self._physics, position, quaternion)

    def step(self, action: np.ndarray) -> tuple:
        # TODO use the action to control the arm

        # get mocap target pose
        target_pose = self._target.get_mocap_pose(self._physics)

        # run OSC controller to move to target pose
        self._controller.run(target_pose)

        # step physics
        self._physics.step()

        # render frame
        if self._render_mode == "human":
            self._render_frame()
        
        # TODO come up with a reward, termination function that makes sense for your RL task
        observation = self._get_obs()
        reward = 0
        terminated = False
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self) -> np.ndarray:
        """
        Renders the current frame and returns it as an RGB array if the render mode is set to "rgb_array".

        Returns:
            np.ndarray: RGB array of the current frame.
        """
        if self._render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> None:
        """
        Renders the current frame and updates the viewer if the render mode is set to "human".
        """
        if self._viewer is None and self._render_mode == "human":
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
            )
        if self._step_start is None and self._render_mode == "human":
            # initialize step timer
            self._step_start = time.time()

        if self._render_mode == "human":
            # render viewer
            self._viewer.sync()

            # TODO come up with a better frame rate keeping strategy
            time_until_next_step = self._timestep - (time.time() - self._step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            self._step_start = time.time()

        else:  # rgb_array
            return self._physics.render()

    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()