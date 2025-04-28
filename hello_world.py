# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from isaacsim.examples.interactive.base_sample import BaseSample
import numpy as np
# Can be used to create a new cube or to point to an already existing cube in stage.
from isaacsim.core.api.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.robots import Robot
import carb

# Note: checkout the required tutorials at https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html


class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._start_time = None  # 시뮬레이션 시작 시간
        self._robot_stopped = False  # 로봇이 이미 멈췄는지 체크
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        """
        fancy_cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube", # The prim path of the cube in the USD stage
                name="fancy_cube", # The unique name used to retrieve the object from the scene later on
                position=np.array([0, 0, 1.0]), # Using the current stage units which is in meters by default.
                scale=np.array([0.5015, 0.5015, 0.5015]), # most arguments accept mainly numpy arrays.
                color=np.array([0, 0, 1.0]), # RGB channels, going from 0-1
            ))
        """
        # you configure a new server with /Isaac folder in it
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            # Use carb to log warnings, errors and infos in your application (shown on terminal)
            carb.log_error("Could not find nucleus server with /Isaac folder")
        asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        # This will create a new XFormPrim and point it to the usd file as a reference
        # Similar to how pointers work in memory
        add_reference_to_stage(usd_path=asset_path, prim_path="/World/Fancy_Robot")
        # Wrap the jetbot prim root under a Robot class and add it to the Scene
        # to use high level api to set/ get attributes as well as initializing
        # physics handles needed..etc.
        # Note: this call doesn't create the Jetbot in the stage window, it was already
        # created with the add_reference_to_stage
        jetbot_robot = world.scene.add(Robot(prim_path="/World/Fancy_Robot", name="fancy_robot"))
        # Note: before a reset is called, we can't access information related to an Articulation
        # because physics handles are not initialized yet. setup_post_load is called after
        # the first reset so we can do so there
        print("Num of degrees of freedom before first reset: " + str(jetbot_robot.num_dof)) # prints None


        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._jetbot = self._world.scene.get_object("fancy_robot")
        # This is an implicit PD controller of the jetbot/ articulation
        # setting PD gains, applying actions, switching control modes..etc.
        # can be done through this controller.
        # Note: should be only called after the first reset happens to the world
        self._jetbot_articulation_controller = self._jetbot.get_articulation_controller()
        # Adding a physics callback to send the actions to apply actions with every
        # physics step executed.
        self._world.add_physics_callback("sending_actions", callback_fn=self.send_robot_actions)

        # Reset이 완료된 시점의 현재 시뮬레이션 시간 기록
        self._start_time = self._world.current_time
        self._robot_stopped = False

        # Print info about the jetbot after the first reset is called
        print("Num of degrees of freedom after first reset: " + str(self._jetbot.num_dof)) # prints 2
        print("Joint Positions after first reset: " + str(self._jetbot.get_joint_positions()))

        """
        self._cube = self._world.scene.get_object("fancy_cube")
        self._world.add_physics_callback("sim_step", callback_fn=self.print_cube_info) #callback names have to be unique
        """
        return
    
    def send_robot_actions(self, step_size):
        # Every articulation controller has apply_action method
        # which takes in ArticulationAction with joint_positions, joint_efforts and joint_velocities
        # as optional args. It accepts numpy arrays of floats OR lists of floats and None
        # None means that nothing is applied to this dof index in this step
        # ALTERNATIVELY, same method is called from self._jetbot.apply_action(...)
        if self._jetbot_articulation_controller is None:
            return

        # 현재 시뮬레이션 시간 가져오기
        current_time = self._world.current_time

        if not self._robot_stopped:
            if self._start_time is not None and (current_time - self._start_time) >= 5.0:
                # 5초 경과 → 로봇 멈춤
                print("🛑 5초 경과: 로봇 정지!")
                self._jetbot_articulation_controller.apply_action(
                    ArticulationAction(
                        joint_positions=None,
                        joint_efforts=None,
                        joint_velocities=[0.0, 0.0]  # 속도를 0으로 설정
                    )
                )
                self._robot_stopped = True
            else:
                # 아직 5초 안 됐으면 랜덤 속도 부여
                self._jetbot_articulation_controller.apply_action(
                    ArticulationAction(
                        joint_positions=None,
                        joint_efforts=None,
                        joint_velocities=5 * np.random.rand(2,)
                    )
                )

    """
    # here we define the physics callback to be called before each physics step, all physics callbacks must take
    # step_size as an argument
    def print_cube_info(self, step_size):
        position, orientation = self._cube.get_world_pose()
        linear_velocity = self._cube.get_linear_velocity()
        # will be shown on terminal
        print("Cube position is : " + str(position))
        print("Cube's orientation is : " + str(orientation))
        print("Cube's linear velocity is : " + str(linear_velocity))

    def post_reset_button_event(self):
        # Reset 버튼을 누른 뒤 로봇 위치 출력
        # /home/beearth/isaacsim/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/base_sample/base_sample_extension.py
        # 위의 코드에서 def post_reset_button_event(self): 내부에 self._sample.post_reset_button_event() 라고 업데이트 필요
        if self._jetbot is not None:
            current_joint_positions = self._jetbot.get_joint_positions()
            print("✅ Reset 이후 로봇 초기 위치: ", current_joint_positions)
        else:
            print("⚠️ Warning: 아직 로봇이 로드되지 않았습니다.")
    """
    async def setup_pre_reset(self):
        return

    async def setup_post_reset(self):
        return

    def world_cleanup(self):
        return
