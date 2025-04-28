from isaacsim.examples.interactive.base_sample import RoaiBaseSample
from isaacsim.robot.manipulators.examples.franka.tasks import PickPlace
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.robot.wheeled_robots.controllers.wheel_base_pose_controller import WheelBasePoseController
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.api.objects.cuboid import VisualCuboid
import numpy as np
from isaacsim.examples.interactive.user_examples.relay_task import RelayTask
from isaacsim.examples.interactive.user_examples.hand_over_task import HandOverTask
from omni.isaac.core.utils.stage import get_current_stage
from pxr import PhysxSchema


class RoaiTest4(RoaiBaseSample):
    def __init__(self) -> None:
        super().__init__()
        self._current_task_index = 0  # 현재 실행 중인 작업 인덱스
        self._tasks = []
        self._num_of_handover_tasks = 3  # HandoverTask 개수
        self._franka_controllers = []
        self._frankas = []
        self._cubes = []
        return

    def setup_scene(self):
        world = self.get_world()
        
        # RelayTask 추가
        relay_task = RelayTask(name="start_task")
        world.add_task(relay_task)
        self._tasks.append("start_task")
        
        # HandoverTask 추가
        for i in range(self._num_of_handover_tasks):
            is_first = (i == 0)
            previous_robot = f"franka_{i-1}" if i > 0 else "franka"
            next_robot = f"franka_{i+1}" if i < self._num_of_handover_tasks - 1 else None
            
            task_name = f"handover_task_{i}"
            task = HandOverTask(
                name=task_name,
                offset=np.array([0.5, -0.5 * (1+i), 0]),  # X 축으로 배치
                is_first=is_first,
                cube_path="/World/cube",
                previous_robot=previous_robot,
                next_robot=next_robot
            )
            world.add_task(task)
            self._tasks.append(task_name)
        

        return

    async def setup_post_load(self):
        self._world = self.get_world()
        
        # RelayTask 컨트롤러 설정
        task_params = self._world.get_task("start_task").get_params()
        self._franka = self._world.scene.get_object(task_params["franka_name"]["value"])
        self._jetbot = self._world.scene.get_object(task_params["jetbot_name"]["value"])
        self._cube_name = task_params["cube_name"]["value"]
        
        # RelayTask의 Franka 컨트롤러 추가
        self._franka_controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self._franka.gripper,
            robot_articulation=self._franka
        )
        self._franka_controllers.append(self._franka_controller)
        self._frankas.append(self._franka)
        
        # Jetbot 컨트롤러 설정
        self._jetbot_controller = WheelBasePoseController(
            name="cool_controller",
            open_loop_wheel_controller=DifferentialController(
                name="simple_control",
                wheel_radius=0.03, 
                wheel_base=0.1125
            )
        )

        # HandoverTask의 Franka 컨트롤러 추가
        for i in range(self._num_of_handover_tasks):
            task_name = f"handover_task_{i}"
            task_params = self._world.get_task(task_name).get_params()
            franka = self._world.scene.get_object(task_params["franka_name"]["value"])
            
            controller = PickPlaceController(
                name=f"pick_place_controller_{i}",
                gripper=franka.gripper,
                robot_articulation=franka
            )
            
            self._franka_controllers.append(controller)
            self._frankas.append(franka)
        
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_post_reset(self):
        # 모든 컨트롤러 리셋
        self._jetbot_controller.reset()

        self._franka_controller.reset()
        for controller in self._franka_controllers:
            controller.reset()
            
        self._current_task_index = 0
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        current_observations = self._world.get_observations()
        current_task_name = self._tasks[self._current_task_index]
        
        # RelayTask 실행
        if current_task_name == "start_task":
            if current_observations["task_event"] == 0:
                # Jetbot이 목표 위치로 이동
                self._jetbot.apply_wheel_actions(
                    self._jetbot_controller.forward(
                        start_position=current_observations[self._jetbot.name]["position"],
                        start_orientation=current_observations[self._jetbot.name]["orientation"],
                        goal_position=current_observations[self._jetbot.name]["goal_position"]
                    )
                )
            elif current_observations["task_event"] == 1:
                # Jetbot이 도착했으니 후진하고, Franka가 큐브를 집음
                self._jetbot.apply_wheel_actions(ArticulationAction(joint_velocities=[-8, -8]))
                
                actions = self._franka_controller.forward(
                    picking_position=current_observations[self._cube_name]["position"],
                    placing_position=current_observations[self._cube_name]["target_position"],
                    current_joint_positions=current_observations[self._franka.name]["joint_positions"]
                )
                self._franka.apply_action(actions)
            elif current_observations["task_event"] == 2:
                # Jetbot 정지, Franka가 큐브를 목표 위치에 놓음
                self._jetbot.apply_wheel_actions(ArticulationAction(joint_velocities=[0.0, 0.0]))

                actions = self._franka_controller.forward(
                    picking_position=current_observations[self._cube_name]["position"],
                    placing_position=current_observations[self._cube_name]["target_position"],
                    current_joint_positions=current_observations[self._franka.name]["joint_positions"]
                )
                self._franka.apply_action(actions)
                
                # RelayTask가 완료되었는지 확인
                if self._franka_controller.is_done():
                    task = self._world.get_task(current_task_name)
                    task.mark_as_done()
                    self._current_task_index += 1
        
        # HandoverTask 실행
        elif "handover_task_" in current_task_name:
            task_index = int(current_task_name.split("_")[-1])
            task = self._world.get_task(current_task_name)
            task_params = task.get_params()
            
            # 현재 Franka와 컨트롤러 가져오기
            franka = self._frankas[task_index + 1]  # +1은 RelayTask의 Franka가 첫 번째이기 때문
            controller = self._franka_controllers[task_index + 1]
            
            # 큐브 정보
            cube_name = task_params["cube_name"]["value"]
            
            # Handover 작업 실행
            actions = controller.forward(
                picking_position=current_observations[cube_name]["position"],
                placing_position=current_observations[cube_name]["target_position"],
                current_joint_positions=current_observations[franka.name]["joint_positions"]
            )
            franka.apply_action(actions)
            
            # 현재 HandoverTask가 완료되었는지 확인
            if controller.is_done():
                task.mark_as_done()
                
                # 다음 작업이 있으면 이동
                if self._current_task_index < len(self._tasks) - 1:
                    self._current_task_index += 1
                else:
                    # 모든 작업이 완료되면 시뮬레이션 일시 중지
                    self._world.pause()
        
        return

    def world_cleanup(self):
        self._current_task_index = 0
        self._tasks = []
        self._franka_controllers = []
        self._frankas = []
        self._cubes = []
        return