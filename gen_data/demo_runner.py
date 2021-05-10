# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import multiprocessing
import os
import random
import time
from enum import Enum

import numpy as np
from PIL import Image
from settings import default_sim_settings, make_cfg

import habitat_sim
import habitat_sim.agent
from habitat_sim.nav import ShortestPath
from habitat_sim.physics import MotionType
from habitat_sim.utils.common import (
    d3_40_colors_rgb,
    download_and_unzip,
    quat_from_angle_axis,
)

_barrier = None


class DemoRunnerType(Enum):
    BENCHMARK = 1
    EXAMPLE = 2
    AB_TEST = 3


class ABTestGroup(Enum):
    CONTROL = 1
    TEST = 2


class DemoRunner:
    def __init__(self, sim_settings, simulator_demo_type):
        if simulator_demo_type == DemoRunnerType.EXAMPLE:
            self.set_sim_settings(sim_settings)
        self._demo_type = simulator_demo_type

    def set_sim_settings(self, sim_settings):
        self._sim_settings = sim_settings.copy()

    def save_color_observation(self, obs, total_frames):
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        if self._demo_type == DemoRunnerType.AB_TEST:
            if self._group_id == ABTestGroup.CONTROL:
                color_img.save("test.rgba.control.%06d.png" % total_frames)
            else:
                color_img.save("test.rgba.test.%06d.png" % total_frames)
        else:
            # color_img.save("test.rgba.%05d.png" % total_frames)
            color_img.save(os.path.join(self._sim_settings['save_subdir'], "%06d.png") % total_frames)
            # color_img.save(os.path.join(self._sim_settings['save_dir'], "%05d.png") % \
                            # (total_frames + self._sim_settings["seed"] * self._sim_settings['max_frames']))

    def save_semantic_observation(self, obs, total_frames):
        semantic_obs = obs["semantic_sensor"]
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        if self._demo_type == DemoRunnerType.AB_TEST:
            if self._group_id == ABTestGroup.CONTROL:
                semantic_img.save("test.sem.control.%05d.png" % total_frames)
            else:
                semantic_img.save("test.sem.test.%05d.png" % total_frames)
        else:
            semantic_img.save("test.sem.%05d.png" % total_frames)

    def save_depth_observation(self, obs, total_frames):
        depth_obs = obs["depth_sensor"]
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        if self._demo_type == DemoRunnerType.AB_TEST:
            if self._group_id == ABTestGroup.CONTROL:
                depth_img.save("test.depth.control.%05d.png" % total_frames)
            else:
                depth_img.save("test.depth.test.%05d.png" % total_frames)
        else:
            depth_img.save("test.depth.%05d.png" % total_frames)

    def output_semantic_mask_stats(self, obs, total_frames):
        semantic_obs = obs["semantic_sensor"]
        counts = np.bincount(semantic_obs.flatten())
        total_count = np.sum(counts)
        print(f"Pixel statistics for frame {total_frames}")
        for object_i, count in enumerate(counts):
            sem_obj = self._sim.semantic_scene.objects[object_i]
            cat = sem_obj.category.name()
            pixel_ratio = count / total_count
            if pixel_ratio > 0.01:
                print(f"obj_id:{sem_obj.id},category:{cat},pixel_ratio:{pixel_ratio}")

    def init_agent_state(self, agent_id):
        # initialize the agent at a random start state
        agent = self._sim.initialize_agent(agent_id)
        start_state = agent.get_state()

        # force starting position on first floor (try 100 samples)
        num_start_tries = 0
        while start_state.position[1] > 0.5 and num_start_tries < 100:
            start_state.position = self._sim.pathfinder.get_random_navigable_point()
            num_start_tries += 1
        agent.set_state(start_state)

        if not self._sim_settings["silent"]:
            print(
                "start_state.position\t",
                start_state.position,
                "start_state.rotation\t",
                start_state.rotation,
            )

        return start_state

    def compute_shortest_path(self, start_pos, end_pos):
        self._shortest_path.requested_start = start_pos
        self._shortest_path.requested_end = end_pos
        self._sim.pathfinder.find_path(self._shortest_path)
        print("shortest_path.geodesic_distance", self._shortest_path.geodesic_distance)

    def init_physics_test_scene(self, num_objects):
        object_position = np.array(
            [-0.569043, 2.04804, 13.6156]
        )  # above the castle table

        # turn agent toward the object
        print("turning agent toward the physics!")
        agent_state = self._sim.get_agent(0).get_state()
        agent_to_obj = object_position - agent_state.position
        agent_local_forward = np.array([0, 0, -1.0])
        flat_to_obj = np.array([agent_to_obj[0], 0.0, agent_to_obj[2]])
        flat_dist_to_obj = np.linalg.norm(flat_to_obj)
        flat_to_obj /= flat_dist_to_obj
        # move the agent closer to the objects if too far (this will be projected back to floor in set)
        if flat_dist_to_obj > 3.0:
            agent_state.position = object_position - flat_to_obj * 3.0
        # unit y normal plane for rotation
        det = (
            flat_to_obj[0] * agent_local_forward[2]
            - agent_local_forward[0] * flat_to_obj[2]
        )
        turn_angle = math.atan2(det, np.dot(agent_local_forward, flat_to_obj))
        agent_state.rotation = quat_from_angle_axis(turn_angle, np.array([0, 1.0, 0]))
        # need to move the sensors too
        for sensor in agent_state.sensor_states:
            agent_state.sensor_states[sensor].rotation = agent_state.rotation
            agent_state.sensor_states[
                sensor
            ].position = agent_state.position + np.array([0, 1.5, 0])
        self._sim.get_agent(0).set_state(agent_state)

        # hard coded dimensions of maximum bounding box for all 3 default objects:
        max_union_bb_dim = np.array([0.125, 0.19, 0.26])

        # add some objects in a grid
        object_library = self._sim.get_object_template_manager()
        object_lib_size = object_library.get_num_templates()
        object_init_grid_dim = (3, 1, 3)
        object_init_grid = {}
        assert (
            object_lib_size > 0
        ), "!!!No objects loaded in library, aborting object instancing example!!!"

        # clear the objects if we are re-running this initializer
        for old_obj_id in self._sim.get_existing_object_ids():
            self._sim.remove_object(old_obj_id)

        for _obj_id in range(num_objects):
            # rand_obj_index = random.randint(0, object_lib_size - 1)
            # rand_obj_index = 0  # overwrite for specific object only
            rand_obj_index = self._sim_settings.get("test_object_index")
            if rand_obj_index < 0:  # get random object on -1
                rand_obj_index = random.randint(0, object_lib_size - 1)
            object_init_cell = (
                random.randint(-object_init_grid_dim[0], object_init_grid_dim[0]),
                random.randint(-object_init_grid_dim[1], object_init_grid_dim[1]),
                random.randint(-object_init_grid_dim[2], object_init_grid_dim[2]),
            )
            while object_init_cell in object_init_grid:
                object_init_cell = (
                    random.randint(-object_init_grid_dim[0], object_init_grid_dim[0]),
                    random.randint(-object_init_grid_dim[1], object_init_grid_dim[1]),
                    random.randint(-object_init_grid_dim[2], object_init_grid_dim[2]),
                )
            object_id = self._sim.add_object(rand_obj_index)
            object_init_grid[object_init_cell] = object_id
            object_offset = np.array(
                [
                    max_union_bb_dim[0] * object_init_cell[0],
                    max_union_bb_dim[1] * object_init_cell[1],
                    max_union_bb_dim[2] * object_init_cell[2],
                ]
            )
            self._sim.set_translation(object_position + object_offset, object_id)
            print(
                "added object: "
                + str(object_id)
                + " of type "
                + str(rand_obj_index)
                + " at: "
                + str(object_position + object_offset)
                + " | "
                + str(object_init_cell)
            )

    def do_time_steps(self):

        total_sim_step_time = 0.0
        total_frames = 0
        start_time = time.time()
        time_per_step = []


        collided = False            
        action_names = ['move_forward', 'turn_left', 'turn_right']

        while total_frames < self._sim_settings["max_frames"]:
            if total_frames == 1:
                start_time = time.time()

            if self._sim_settings['mode'] == 'nav':
                if collided:
                    collided = False
                    action = random.choice(['turn_left', 'turn_right'])
                    repeat_num = random.choice([5, 6, 7, 8, 9, 10, 11, 12])
                else:
                    action = random.choice(action_names)
                    if action == 'stop':
                        repeat_num = random.choice([1, 2])
                    else:
                        repeat_num = random.choice([1, 2, 3, 4, 5])
            else: # mode ptz
                action = 'spin'
                repeat_num = 10
                collided = False

            while repeat_num > 0 and total_frames < self._sim_settings['max_frames']:
                with open(self._sim_settings['save_subdir'] + '_action.txt', 'a') as file:
                    file.write('%s\n' % action)

                if not self._sim_settings["silent"]:
                    print("action", action)

                start_step_time = time.time()
                # get "interaction" time
                total_sim_step_time += time.time() - start_step_time

################################################################change start
                # observations = self._sim.step(action)
                # time_per_step.append(time.time() - start_step_time)
                collided = self._sim._default_agent.act(action)
                observations = self._sim.get_sensor_observations()
                pos = self._sim.agents[0].state.sensor_states['color_sensor'].position
                rot = self._sim.agents[0].state.sensor_states['color_sensor'].rotation
                # calc different 
                dist_from_origin = np.linalg.norm([pos[0], pos[2]]) # y is the height
                angle_along_y_axis = np.arctan2(rot.y, rot.w) * 2 / np.pi * 180 % 360 # convert from quaternion
                if not self._sim_settings['silent']:
                    print('position: ', dist_from_origin, ' rotation: ', angle_along_y_axis)
                with open(self._sim_settings['save_subdir'] + '_state.txt', 'a') as file:
                    file.write('%.2f %.2f %.2f %.0f\n' % (pos[0], pos[1], pos[2], angle_along_y_axis))                
################################################################change end

                # get simulation step time without sensor observations
                total_sim_step_time += self._sim._previous_step_time

                if self._sim_settings["save_png"]:
                    if self._sim_settings["color_sensor"]:
                        self.save_color_observation(observations, total_frames)
                    if self._sim_settings["depth_sensor"]:
                        self.save_depth_observation(observations, total_frames)
                    if self._sim_settings["semantic_sensor"]:
                        self.save_semantic_observation(observations, total_frames)

                state = self._sim.last_state()

                if not self._sim_settings["silent"]:
                    print("position\t", state.position, "\t", "rotation\t", state.rotation)

                if self._sim_settings["compute_shortest_path"]:
                    self.compute_shortest_path(
                        state.position, self._sim_settings["goal_position"]
                    )

                if self._sim_settings["compute_action_shortest_path"]:
                    self._action_path = self.greedy_follower.find_path(
                        self._sim_settings["goal_position"]
                    )
                    print("len(action_path)", len(self._action_path))

                if (
                    self._sim_settings["semantic_sensor"]
                    and self._sim_settings["print_semantic_mask_stats"]
                ):
                    self.output_semantic_mask_stats(observations, total_frames)

                total_frames += 1

###############################################################change start
                repeat_num -= 1
                if collided:
                    break
##############################################################change end                    

        end_time = time.time()
        perf = {}
        perf["total_time"] = end_time - start_time
        perf["frame_time"] = perf["total_time"] / total_frames
        perf["fps"] = 1.0 / perf["frame_time"]
        perf["time_per_step"] = time_per_step
        perf["avg_sim_step_time"] = total_sim_step_time / total_frames

        return perf

    def print_semantic_scene(self):
        if self._sim_settings["print_semantic_scene"]:
            scene = self._sim.semantic_scene
            print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
            for level in scene.levels:
                print(
                    f"Level id:{level.id}, center:{level.aabb.center},"
                    f" dims:{level.aabb.sizes}"
                )
                for region in level.regions:
                    print(
                        f"Region id:{region.id}, category:{region.category.name()},"
                        f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
                    )
                    for obj in region.objects:
                        print(
                            f"Object id:{obj.id}, category:{obj.category.name()},"
                            f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                        )
            input("Press Enter to continue...")

    def init_common(self):
        self._cfg = make_cfg(self._sim_settings)
        scene_file = self._sim_settings["scene"]

        if (
            not os.path.exists(scene_file)
            and scene_file == default_sim_settings["scene"]
        ):
            print(
                "Test scenes not downloaded locally, downloading and extracting now..."
            )
            download_and_unzip(default_sim_settings["test_scene_data_url"], ".")
            print("Downloaded and extracted test scenes data.")

        # create a simulator (Simulator python class object, not the backend simulator)
        self._sim = habitat_sim.Simulator(self._cfg)

        random.seed(self._sim_settings["seed"])
        self._sim.seed(self._sim_settings["seed"])

        recompute_navmesh = self._sim_settings.get("recompute_navmesh")
        if recompute_navmesh or not self._sim.pathfinder.is_loaded:
            if not self._sim_settings["silent"]:
                print("Recomputing navmesh")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            self._sim.recompute_navmesh(self._sim.pathfinder, navmesh_settings)

        # initialize the agent at a random start state
        return self.init_agent_state(self._sim_settings["default_agent"])

    def _bench_target(self, _idx=0):
        self.init_common()

        best_perf = None
        for _ in range(3):

            if _barrier is not None:
                _barrier.wait()
                if _idx == 0:
                    _barrier.reset()

            perf = self.do_time_steps()
            # The variance introduced between runs is due to the worker threads
            # being interrupted a different number of times by the kernel, not
            # due to difference in the speed of the code itself.  The most
            # accurate representation of the performance would be a run where
            # the kernel never interrupted the workers, but this isn't
            # feasible, so we just take the run with the least number of
            # interrupts (the fastest) instead.
            if best_perf is None or perf["frame_time"] < best_perf["frame_time"]:
                best_perf = perf

        self._sim.close()
        del self._sim

        return best_perf

    @staticmethod
    def _pool_init(b):
        global _barrier
        _barrier = b

    def benchmark(self, settings, group_id=ABTestGroup.CONTROL):
        self.set_sim_settings(settings)
        nprocs = settings["num_processes"]
        # set it anyway, but only be used in AB_TEST mode
        self._group_id = group_id

        barrier = multiprocessing.Barrier(nprocs)
        with multiprocessing.Pool(
            nprocs, initializer=self._pool_init, initargs=(barrier,)
        ) as pool:
            perfs = pool.map(self._bench_target, range(nprocs))

        res = {k: [] for k in perfs[0].keys()}
        for p in perfs:
            for k, v in p.items():
                res[k] += [v]

        return dict(
            frame_time=sum(res["frame_time"]),
            fps=sum(res["fps"]),
            total_time=sum(res["total_time"]) / nprocs,
            avg_sim_step_time=sum(res["avg_sim_step_time"]) / nprocs,
        )

    def example(self):
        start_state = self.init_common()

        # initialize and compute shortest path to goal
        if self._sim_settings["compute_shortest_path"]:
            self._shortest_path = ShortestPath()
            self.compute_shortest_path(
                start_state.position, self._sim_settings["goal_position"]
            )

        # set the goal headings, and compute action shortest path
        if self._sim_settings["compute_action_shortest_path"]:
            agent_id = self._sim_settings["default_agent"]
            self.greedy_follower = self._sim.make_greedy_follower(agent_id=agent_id)

            self._action_path = self.greedy_follower.find_path(
                self._sim_settings["goal_position"]
            )
            print("len(action_path)", len(self._action_path))

        # print semantic scene
        self.print_semantic_scene()

##########################################################change start                    
        for curr_loc_idx in range(self._sim_settings['init_loc'], 
                                  self._sim_settings['init_loc'] + self._sim_settings['loc_num']): 
            # reinitalization
            self._sim_settings['seed'] = curr_loc_idx
            random.seed(curr_loc_idx)
            self._sim.seed(curr_loc_idx)
            self._sim_settings['save_subdir'] = os.path.join(self._sim_settings['save_dir'], 
                                                            str(curr_loc_idx).zfill(3))
            if not os.path.exists(self._sim_settings["save_subdir"]):
                os.makedirs(self._sim_settings["save_subdir"])
            self.init_agent_state(self._sim_settings["default_agent"])
##########################################################change end

            perf = self.do_time_steps() # reinitialize after each run
        self._sim.close()
        del self._sim

        return perf
