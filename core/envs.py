import numpy as np
import highway_env


class Highway(highway_env.highway_env.envs.HighwayEnv):
    def __init__(self, mode='train'):
        super(highway_env.highway_env.envs.HighwayEnv, self).__init__()
        self.mode = mode
        self.default_config()
        self.num_constraints = len(self.get_constraint_values())

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "absolute": False,
            },
            # "action": {
            #     "type": "ContinuousAction",
            # },
            "lanes_count": 4,
            "vehicles_count": 15,
            "policy_frequency": 10,
            "duration": 500,
            "manual_control": True,
            "offroad_terminal": True
        })
        return config

    @classmethod
    def config_train(cls) -> dict:
        config = super().default_config()
        config.update({
            "action": {
                "type": "ContinuousAction",
            },
            "manual_control": False,
        })
        return config

    def get_long_distance(self):
        '''returns longitudinal distance to closest car in front of ego vehicle'''
        # We define all the constraints such that C_i = 0
        # distance_closest >= d_min -----> d_min - distance_closest <= 0
        d_min = 15  # m
        front_vehicle, _ = self.road.neighbour_vehicles(self.vehicle)
        distance = self.vehicle.lane_distance_to(front_vehicle)
        c = (d_min - distance) / d_min
        c = np.clip(-1, c, 1)
        return c

    def get_lane_distance(self):
        '''returns lateral distance to closest road limit'''
        d_min = - 0.4  # discount a bit because of the car width
        ego_position = self.vehicle.position
        lanes = self.road.network.lanes_list()
        _, distance_left = lanes[0].local_coordinates(ego_position)
        _, distance_right = lanes[-1].local_coordinates(ego_position)
        min_distance = min(- distance_right, distance_left)
        c = d_min - min_distance  # distance to lane limits
        return c

    def get_constraint_values(self):
        long_dist_const = self.get_long_distance()
        lane_dist_const = self.get_lane_distance()
        return np.array([long_dist_const, lane_dist_const])