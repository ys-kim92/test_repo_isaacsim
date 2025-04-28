from isaacsim.robot.manipulators.examples.franka.tasks import PickPlace
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.tasks import BaseTask
import numpy as np

#from isaacsim.examples.interactive.base_sample import RoaiBaseSample
#from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
#from isaacsim.robot.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
#from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
#from isaacsim.core.utils.types import ArticulationAction
#from isaacsim.core.api.objects.cuboid import VisualCuboid


class RelayTask(BaseTask):
    def __init__(
        self,
        name
    ):
        super().__init__(name=name, offset=None)
        self._jetbot_goal_position = np.array([1.3, 0.3, 0])
        self._task_event = 0
        self._is_done = False
        # 큐브 시작 위치와 Franka가 큐브를 놓을 위치 설정
        self._pick_place_task = PickPlace(cube_initial_position=np.array([0.1, 0.3, 0.05]),
                                        target_position=np.array([1, -0.5, 0.0515 / 2.0]))
        return

    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        assets_root_path = get_assets_root_path()
        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        self._jetbot = scene.add(
            WheeledRobot(
                prim_path="/World/Fancy_Jetbot",
                name="fancy_jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=jetbot_asset_path,
                position=np.array([0, 0.3, 0]),
            )
        )

        self._pick_place_task.set_up_scene(scene)
        pick_place_params = self._pick_place_task.get_params()
        self._franka = scene.get_object(pick_place_params["robot_name"]["value"])
        self._franka.set_world_pose(position=np.array([1.0, 0, 0]))
        self._franka.set_default_state(position=np.array([1.0, 0, 0]))
        return

    def get_observations(self):
        current_jetbot_position, current_jetbot_orientation = self._jetbot.get_world_pose()
        observations= {
            "task_event": self._task_event,
            "is_relay_done": self._is_done,
            self._jetbot.name: {
                "position": current_jetbot_position,
                "orientation": current_jetbot_orientation,
                "goal_position": self._jetbot_goal_position
            }
        }
        # add the subtask's observations as well
        observations.update(self._pick_place_task.get_observations())
        return observations

    def get_params(self):
        pick_place_params = self._pick_place_task.get_params()
        params_representation = pick_place_params
        params_representation["jetbot_name"] = {"value": self._jetbot.name, "modifiable": False}
        params_representation["franka_name"] = pick_place_params["robot_name"]
        return params_representation

    def pre_step(self, control_index, simulation_time):
        if self._task_event == 0:
            current_jetbot_position, _ = self._jetbot.get_world_pose()
            if np.mean(np.abs(current_jetbot_position[:2] - self._jetbot_goal_position[:2])) < 0.04:
                self._task_event += 1
                self._cube_arrive_step_index = control_index
        elif self._task_event == 1:
            if control_index - self._cube_arrive_step_index == 200:
                self._task_event += 1
        return

    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._task_event = 0
        self._is_done = False
        return
    
    def is_done(self):
        return self._is_done

    def mark_as_done(self):
        self._is_done = True