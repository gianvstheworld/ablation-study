#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Point
from random import random

class Node:
    def __init__(self, position):
        self.position = position
        self.parent = None

class RRTPlanner:
    def __init__(self):
        rospy.init_node('rrt_planner')
        self.pub_path = rospy.Publisher('path', String, queue_size=10)
        self.sub_goal = rospy.Subscriber('goal', Point, self.goal_callback)
        self.goal = None

    def goal_callback(self, msg):
        self.goal = msg

    def run(self):
        while not rospy.is_shutdown():
            if self.goal is not None:
                path = self.plan_path()
                self.publish_path(path)
                self.goal = None

    def plan_path(self):
        if self.goal is None:
            return None

        start_position = Point(0, 0, 0)
        max_iterations = 1000
        goal_threshold = 0.5
        growth_factor = 0.1

        rrt_tree = []

        start_node = Node(start_position)
        rrt_tree.append(start_node)

        for i in range(max_iterations):
            sample_position = self.generate_random_position()

            nearest_node = self.find_nearest_node(rrt_tree, sample_position)

            new_node = self.expand_tree(
                nearest_node, sample_position, growth_factor)

            if self.distance_to_goal(new_node.position) < goal_threshold:
                rrt_tree.append(new_node)
                path = self.retrieve_path(rrt_tree, new_node)
                return path

    def publish_path(self, path):
        if path is not None:
            path_msg = String()
            path_msg.data = ','.join(
                [str(node.position.x) + ',' + str(node.position.y) for node in path])

            self.pub_path.publish(path_msg)

    def generate_random_position(self):
        x = random() * 10
        y = random() * 10
        return Point(x, y, 0)

    def find_nearest_node(self, rrt_tree, sample_position):
        nearest_node = None
        nearest_distance = float('inf')

        for node in rrt_tree:
            distance = self.distance_between(node.position, sample_position)
            if distance < nearest_distance:
                nearest_node = node
                nearest_distance = distance

        return nearest_node

    def expand_tree(self, nearest_node, sample_position, growth_factor):
        new_node = Node(nearest_node.position)
        new_node.position.x += (sample_position.x -
                                nearest_node.position.x) * growth_factor
        new_node.position.y += (sample_position.y -
                                nearest_node.position.y) * growth_factor
        return new_node

    def distance_to_goal(self, position):
        return self.distance_between(position, self.goal)

    def distance_between(self, position_a, position_b):
        return ((position_a.x - position_b.x) ** 2 + (position_a.y - position_b.y) ** 2) ** 0.5

    def retrieve_path(self, rrt_tree, node):
        path = []
        while node.parent is not None:
            path.append(node)
            node = node.parent
        path.append(node)
        path.reverse()
        return path

if __name__ == '__main__':
    planner = RRTPlanner()
    planner.run()