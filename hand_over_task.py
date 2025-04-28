from isaacsim.examples.interactive.base_sample import RoaiBaseSample
from isaacsim.robot.manipulators.examples.franka.tasks import PickPlace
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.robot.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.api.objects.cuboid import VisualCuboid
from typing import Optional
import isaacsim.core.api.tasks as tasks
import numpy as np

# 기존 패키지 상속 후 modify
class PickPlaceExistingCube(tasks.PickPlace):
    def __init__(
        self,
        name: str = "franka_pick_place",
        cube_prim_path: Optional[str] = None,
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ):
        super().__init__(
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=cube_size,
            offset=offset,
        )
        self._cube_prim_path = cube_prim_path  # 추가된 부분

    def set_robot(self) -> Franka:
        """[summary]

        Returns:
            Franka: [description]
        """
        franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        franka_robot_name = find_unique_string_name(
            initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        return Franka(prim_path=franka_prim_path, name=franka_robot_name)

    def set_up_scene(self, scene):
        """Overriding to prevent creating a new cube."""
        super().set_up_scene(scene)
        
        # 기존 PickPlace는 여기서 cube를 만들었을 것임
        # 대신 우리는 이미 존재하는 cube를 찾아서 쓸 것임
        self._cube = scene.get_object(self._cube_prim_path)
        if self._cube is None:
            raise RuntimeError(f"Cube at path {self._cube_prim_path} not found in the scene.")

        # 로봇은 원래대로 생성
        self._robot = self.set_robot()
        scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self._task_objects[self._cube.name] = self._cube  # 기존 cube도 등록
        self._move_task_objects_to_their_frame()


class HandOverTask(BaseTask):
    def __init__(self, name, offset=None, is_first=False, cube_path=None, previous_robot=None, next_robot=None):
        super().__init__(name=name, offset=offset)
        self._task_event = 0
        self._is_done = False
        self._is_first = is_first  # 첫 번째 HandoverTask인지 여부
        self._cube_path = cube_path  # 큐브의 경로
        self._previous_robot = previous_robot  # 이전 로봇 이름
        self._next_robot = next_robot  # 다음 로봇 이름
        
        # 초기 위치와 목표 위치 설정
        initial_pos = np.array([1, -1, 0.05])
        if is_first:
            # 첫 번째 HandoverTask인 경우 RelayTask에서 놓은 위치
            initial_pos = np.array([1, -0.5, 0.0515 / 2.0])
            
        # 다음 로봇에게 넘길 목표 위치 설정
        target_pos = np.array([0.7, -0.3, 0.0515 / 2.0])
        if offset is not None:
            target_pos = target_pos + offset
            """
        self._pick_place_task = PickPlace(cube_initial_position=initial_pos,
                                         target_position=target_pos,
                                         offset=offset)
        """
        self._pick_place_task = PickPlaceExistingCube(
                name=self.name + "_pickplace",     # 반드시 name을 넘겨야 함
                cube_prim_path=self._cube_path,
                target_position=target_pos,
                offset=offset
            )
        
        return

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._pick_place_task.set_up_scene(scene)
        
        # Franka 로봇 가져오기
        pick_place_params = self._pick_place_task.get_params()
        self._franka = scene.get_object(pick_place_params["robot_name"]["value"])
        
        # Franka 로봇 위치 설정 (오프셋 적용)
        current_position, _ = self._franka.get_world_pose()
        if self._offset is not None:
            self._franka.set_world_pose(position=current_position + self._offset)
            self._franka.set_default_state(position=current_position + self._offset)
        
        # 작업 객체들을 프레임으로 이동
        self._move_task_objects_to_their_frame()
        return

    def get_observations(self):
        observations = {
            self.name + "_event": self._task_event,
            "is_" + self.name + "_done": self._is_done,
            "is_first_handover": self._is_first,
            "previous_robot": self._previous_robot,
            "next_robot": self._next_robot
        }
        
        # 하위 작업의 관찰 추가
        observations.update(self._pick_place_task.get_observations())
        return observations

    def get_params(self):
        pick_place_params = self._pick_place_task.get_params()
        params_representation = pick_place_params
        params_representation["franka_name"] = pick_place_params["robot_name"]
        params_representation["cube_path"] = {"value": self._cube_path, "modifiable": False}
        return params_representation
    
    def pre_step(self, control_index, simulation_time):
        # 작업 상태에 따른 로직 구현
        return

    def post_reset(self):
        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
        self._task_event = 0
        self._is_done = False
        
    def is_done(self):
        return self._is_done
        
    def mark_as_done(self):
        self._is_done = True
