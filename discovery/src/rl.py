#/usr/bin/env python
'''
Created on Apr 28, 2011

@author: reza
'''
import random
import math
#import numpy
import copy
import multiprocessing
import os
import subprocess
import sys
import time

USE_NUMPY = False
USE_MULTIPROCESSING = True
NUM_CORES = 8

# default trial and episode numbers
DEFAULT_NUM_TRIALS = 20
DEFAULT_NUM_GENERATIONS = 15
DEFAULT_POPULATION_SIZE = 100
DEFAULT_NUM_GENERATION_EPISODES = 200
DEFAULT_NUM_EPISODES = DEFAULT_NUM_GENERATIONS * DEFAULT_NUM_GENERATION_EPISODES
DEFAULT_NUM_CHAMPION_TRIALS = 1

# evolutionary params
SURVIVAL_RATE = .15
TAU = 2

# debug
DEBUG_PROGRESS = True
DEBUG_EPISODE_TRACE = False
DEBUG_EPISODE_REWARD = False
DEBUG_ALG_VALUES = False
DEBUG_VERBOSE = False
DEBUG_CHAMPION = True
DEBUG_REPORT_ON_EPISODE = 100

# reporting
REPORT_RESULTS = True
PLOT_INTERVALS = 100

# what portion of the training rewards to include in calculating average rewards
TRAINING_SLACK = 0.0

# the computational cost parameter 
DEFAULT_ETA = 1.00

# gamma
DEFAULT_GAMMA = 1.0

# default multiplier for initial Q values based on maximum possible reward
INIT_Q_VALUE_MULTIPLIER = 1.0

WEIGHTS_ZERO = "Zero"
WEIGHTS_OPTIMISTIC = "Optimistic"
WEIGHTS_COPY = "Copy"

BASE_FEATURE_WEIGHTS = WEIGHTS_OPTIMISTIC
MUTATE_NEW_WEIGHTS_MULT = 0.0
MUTATE_CROSS_OVER_WEIGHTS = WEIGHTS_COPY

# the probability by which the algorithm tries to pick a dynamic state variable
# when it has both options
FORCE_DYNAMIC_PROB = 0.75

class AgentStateBased(object):

    def __init__(self, actions, environment, algorithm=None):
        self.state = None
        self.last_state = None
        
        self.actions = actions
        self.environment = environment
        self.algorithm = algorithm

        self.episode_steps = 0
        self.gamma_multiplier = 1.0
        self.episode_reward = 0
        self.episode_trace = ""
        self.reward_log = []
        self.average_reward = 0
        self.average_reward_normalized = 0
        self.training_time = 0
        
#    def set_algorithm(self, algorithm):
#        self.algorithm = algorithm
    
    def begin_episode(self, state):
        self.episode_steps = 0
        self.gamma_multiplier = 1.0
        self.episode_reward = 0
        self.episode_trace = ""
        self.state = state
        self.last_state = copy.deepcopy(self.state)
        if self.algorithm is not None:
            self.algorithm.begin_episode(state)
        
        if DEBUG_EPISODE_TRACE or DEBUG_CHAMPION:
            chunk = "starting at %s" % state
            self.episode_trace += chunk + "\n"
            if DEBUG_EPISODE_TRACE:
                print chunk
        
    def get_episode_trace(self):
        return self.episode_trace
    
#    def end_episode(self, state, action, reward):
#        self.algorithm.end_episode(state, action, reward)
    
#    def do_episode(self, state):
#        return self.algorithm.do_episode(state)
#    
#    def is_episode_ended(self):
#        return self.state.is_terminal()

    def all_actions(self):
        return self.actions.all_actions()

    def select_action(self):
        (a, v) = self.algorithm.select_action(self.state) #@UnusedVariable
        return a
    
    def transition(self, action):

        #last_state = copy.deepcopy(self.state)
        reward = self.environment.respond(self.state, self.last_state, action)
        if self.algorithm is not None:
            self.algorithm.cached_action = None
            self.algorithm.cached_action_value = None
            self.algorithm.cached_action_valid = False
        
        self.episode_steps += 1
        self.episode_reward += reward * self.gamma_multiplier
        self.gamma_multiplier *= self.environment.gamma
        
        next_state = self.state
        next_action = None
        if not next_state.is_final():
            next_action = self.select_action()
        if self.algorithm is not None:
            self.algorithm.transition(self.last_state, action, reward,
                                      next_state, next_action)

        if DEBUG_EPISODE_TRACE or DEBUG_CHAMPION:
            chunk = " (%s) -> %s" % (action, next_state)
            self.episode_trace += chunk + "\n"
            if DEBUG_EPISODE_TRACE:
                print chunk
        
        return reward
    
    def pause_learning(self):
        if self.algorithm is not None:
            self.algorithm.pause_learning()

    def resume_learning(self):
        if self.algorithm is not None:
            self.algorithm.resume_learning()
    
    def save_learning_state(self):
        if self.algorithm is not None:
            self.algorithm.save_learning_state()

    def restore_learning_state(self):
        if self.algorithm is not None:
            self.algorithm.restore_learning_state()

    def reset_learning(self):
        if self.algorithm is not None:
            self.algorithm.reset_learning()

class AgentFeatureBased(AgentStateBased):

    def __init__(self, actions, environment, feature_set, algorithm=None):
        super(AgentFeatureBased, self).__init__(actions, environment, algorithm)
        self.feature_set = feature_set

    def get_name(self):
        return str(self.feature_set)
    
    def __str__(self):
        return self.get_name()

    def clone(self):
        agent = copy.deepcopy(self)
#        algorithm_copy = copy.deepcopy(self.algorithm)
#        feature_set_copy = FeatureSet(list(self.feature_set.feature_list))
#        agent_class = self.__class__
#        agent = agent_class(feature_set_copy)
#        agent.set_algorithm(algorithm_copy)
#        algorithm_copy.agent = agent
        return agent
    
    def add_feature(self, feature):
        self.feature_set.add_feature(feature)
        self.algorithm.add_feature(feature)
        
class Environment(object):
    
    def __init__(self, state_class, gamma = DEFAULT_GAMMA):
        self.gamma = gamma
        self.state_class = state_class

    def generate_start_state(self):
        return self.state_class.generate_start_state()
    
#    def generate_terminal_state(self):
#        return self.state_class.generate_terminal_state()

    def get_environment_vars(self):
        return NotImplemented
    
    def respond(self, state, last_state, action):
        return NotImplemented
    
class Actions(object):
    
    def __init__(self, actions_list):
        self.actions_list = actions_list
    
    def all_actions(self):
        return self.actions_list

    def random_action(self):
        return random.choice(self.actions_list)
    
class State(object):
    
    def __init__(self):
        pass
    
    def __str__(self):
        return NotImplemented
    
#    def make_terminal(self):
#        return NotImplemented

    def is_final(self):
        return NotImplemented
    
#    def is_terminal(self):
#        return NotImplemented
    
class ModularState(object):

    def __init__(self, state_variables):
        self.state_variables = state_variables
        self.index = {}
        for state_variable in state_variables:
            self.index[state_variable.name] = state_variable
#        self.terminal = False
    
    def __str__(self):
        result = ""
        if self.is_final():
            result += "*"
        is_first = True
        for state_var in self.state_variables:
            if state_var.is_dynamic:
                if is_first:
                    is_first = False
                else:
                    result += "|"
                result += str(state_var)
        return result
    
#    def make_terminal(self):
#        self.terminal = True
        
    def is_final(self):
        return NotImplemented
    
#    def is_terminal(self):
#        return self.terminal
    
    @classmethod
    def generate_start_state(cls):
        return NotImplemented

#    @classmethod
#    def generate_terminal_state(cls):
#        state = cls.generate_start_state()
#        state.make_terminal()
#        return state

class StateVar(object):
    
    def __init__(self, name, is_dynamic, is_continuous):
        self.name = name
        self.is_dynamic = is_dynamic
        self.is_continuous = is_continuous
        
    def __str__(self):
        return NotImplemented
    
    @classmethod 
    def get_random_var(cls, state_var_class, state_vars, exclude_vars, 
                       is_dynamic=None, is_continuous=None):
        while True:
            be_dynamic = is_dynamic
            if random.random() < FORCE_DYNAMIC_PROB:
                be_dynamic = True
            var_index = int(random.random() * len(state_vars))
            state_var = state_vars[var_index]
            if isinstance(state_var, state_var_class):
                if state_var not in exclude_vars:
                    var_meets_criteria = True
                    if be_dynamic is not None:
                        if state_var.is_dynamic != be_dynamic:
                            var_meets_criteria = False
                    if is_continuous is not None:
                        if state_var.is_continuous != is_continuous:
                            var_meets_criteria = False
                    if var_meets_criteria:
                        return state_var
        
class StateVarPoint2D(StateVar):
    
    def __init__(self, name, x, y, point_range, is_dynamic, is_continuous):
        super(StateVarPoint2D, self).__init__(name, is_dynamic, is_continuous)
        self.x = x
        self.y = y
        self.point_range = point_range
    
    @classmethod
    def get_random_var(cls, state_vars, exclude_vars,
                       is_dynamic=None, is_continuous=None):
        return StateVar.get_random_var(cls, state_vars, exclude_vars,
                                       is_dynamic, is_continuous)

    def __str__(self):
        return "%2d-%2d" % (self.x, self.y)
    
class StateVarFlag(StateVar):
    
    def __init__(self, name, truth, is_dynamic):
        super(StateVarFlag, self).__init__(name, is_dynamic, is_continuous=False)
        self.truth = truth
    
    @classmethod
    def get_random_var(cls, state_vars, exclude_vars,
                       is_dynamic=None, is_continuous=None):
        return StateVar.get_random_var(cls, state_vars, exclude_vars,
                                       is_dynamic, is_continuous)

    def __str__(self):
        return "%s" % self.truth
        
class FeatureSet(object):
    
    def __init__(self, feature_list):
        self.feature_list = feature_list
        self.encoding_length = 0
        for feature in self.feature_list:
            self.encoding_length += feature.get_encoding_length()
    
    def add_feature(self, feature):
        self.feature_list.append(feature)
        self.encoding_length += feature.get_encoding_length()
    
    def __str__(self):
        names = ""
        is_first = True
        for feature in self.feature_list:
            if is_first:
                is_first = False
            else:
                names += " | "
            names += feature.get_name()
        return names
        
    def encode_state(self, state):
#        if USE_NUMPY:
#            encoding = numpy.array([])
#        else:
#            encoding = []
#        for feature in self.feature_list:
#            next_segment = feature.encode_state(state)
#            if USE_NUMPY:
#                encoding = numpy.concatenate((encoding, next_segment))
#            else:
#                encoding += next_segment
        encoding = []
        for feature in self.feature_list:
            next_segment = feature.encode_state(state)
            encoding += next_segment
        if DEBUG_VERBOSE:
            print str(self)
            print "encoded state %s as: %s" % (state, encoding)
        return encoding

    def get_encoding_length(self):
        return self.encoding_length
    
    def get_cost_factor(self):
        cost = 0
        for feature in self.feature_list:
            cost += feature.get_cost_factor()
        return cost

    def get_num_features(self):
        return len(self.feature_list)
    
    # creates a new list.  doesn't copy the actual features too.
    def get_feature_list_copy(self):
        return list(self.feature_list)
    
class Feature(object):
    
    def __init__(self):
        pass
        
    def __str__(self):
        return self.get_name()
    
    def __repr__(self):
        return self.get_name()
    
    def get_name(self):
        return NotImplemented
    
    def get_encoding_length(self):
        return NotImplemented
    
    def encode_state(self, state):
        return NotImplemented
    
    def get_underlying_features(self):
        return [self]
    
    def reconfigure(self):
        pass
    
    def get_cost_factor(self):
        return 1
    
class TiledFeature(Feature):

    DEFAULT_NUM_TILES = 10
    MIN_OFFSET = 0.0
    MAX_OFFSET = 1.0
    NEUTRAL_OFFSET = 0.5
    EVEN_OFFSETS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    
    def __init__(self, min_value, max_value, num_tiles = DEFAULT_NUM_TILES, 
                 offset = NEUTRAL_OFFSET):
        super(TiledFeature, self).__init__()
        self.num_tiles = num_tiles
        self.min_value = min_value
        self.max_value = max_value
        self.offset = offset

    def get_num_tiles(self):
        return self.num_tiles

    def get_encoding_length(self):
        return self.num_tiles

    def get_tile_index(self, value):
        offset_amount = ((self.offset - self.NEUTRAL_OFFSET) * 2 * 
                         (self.max_value - self.min_value)) / self.num_tiles
        value = value + offset_amount
        tile_index = int(float(value - self.min_value) * self.num_tiles / 
                         (self.max_value - self.min_value))
        tile_index = min(tile_index, self.num_tiles - 1)
        tile_index = max(tile_index, 0)
        return tile_index
    
    def reconfigure(self):
        self.offset = random.random()

class FeatureFlag(TiledFeature):
    
    VALUE_TRUE = 1
    VALUE_FALSE = 0
    
    def __init__(self, flag):
        num_tiles = 2
        self.flag_name = flag.name
        super(FeatureFlag, self).__init__(self.VALUE_FALSE, self.VALUE_TRUE,
                num_tiles, TiledFeature.NEUTRAL_OFFSET)
    
    def get_name(self):
        return "flag(%s)" % (self.flag_name)
    
    def encode_state(self, state):
#        if USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
        feature_encoding = [0] * self.num_tiles
            
#        if state.is_terminal():
#            return feature_encoding
        
        flag = state.index[self.flag_name]
        feature_index = self.VALUE_TRUE if flag.truth else self.VALUE_FALSE
        feature_encoding[feature_index] = 1
        return feature_encoding 
    
    # no random assignment of offset needed
    def reconfigure(self):
        pass

class FeatureAngle(TiledFeature):
    
    def __init__(self, point1, point2, point3,
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES,
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point1_name = point1.name
        self.point2_name = point2.name
        self.point3_name = point3.name
        min_angle = 0
        max_angle = math.pi
        super(FeatureAngle, self).__init__(min_angle, max_angle,
                num_tiles, offset)
    
    def get_name(self):
        return "angle(%s-%s-%s)[%.1f]" % (self.point1_name, self.point2_name,
                                          self.point3_name, self.offset)
            
    def encode_state(self, state):
#        if USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
        feature_encoding = [0] * self.num_tiles
            
#        if state.is_terminal():
#            return feature_encoding
        point1 = state.index[self.point1_name]
        point2 = state.index[self.point2_name]
        point3 = state.index[self.point3_name]
        angle = GeometryUtil.compute_angle(point1.x, point1.y,
                                           point2.x, point2.y,
                                           point3.x, point3.y)
        feature_index = self.get_tile_index(angle)
        feature_encoding[feature_index] = 1
        return feature_encoding 

class FeatureDist(TiledFeature):
    
    def __init__(self, point1, point2, 
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES,
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point1_name = point1.name
        self.point2_name = point2.name
        min_dist = 0
        point_range = point1.point_range
        max_dist = GeometryUtil.compute_dist(point_range[0][0], point_range[0][1], 
                point_range[1][0], point_range[1][1]) 
        super(FeatureDist, self).__init__(min_dist, max_dist,
                num_tiles, offset)

    def get_name(self):
        return "dist(%s-%s)[%.1f]" % (self.point1_name, self.point2_name,
                                      self.offset)
    
    def encode_state(self, state):
#        if USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
        feature_encoding = [0] * self.num_tiles
            
#        if state.is_terminal():
#            return feature_encoding
        point1 = state.index[self.point1_name]
        point2 = state.index[self.point2_name]
        dist = GeometryUtil.compute_dist(point1.x, point1.y,
                                         point2.x, point2.y)
        feature_index = self.get_tile_index(dist)
        feature_encoding[feature_index] = 1
        return feature_encoding 
        
class FeatureDistX(TiledFeature):
    
    def __init__(self, point1, point2, 
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES,
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point1_name = point1.name
        self.point2_name = point2.name
        point_range = point1.point_range
        max_dist = abs(point_range[1][0] - point_range[0][0])
        min_dist = -max_dist 
        super(FeatureDistX, self).__init__(min_dist, max_dist,
                num_tiles, offset)
        
    def get_name(self):
        return "dist-X(%s-%s)[%1.f]" % (self.point1_name, self.point2_name,
                                        self.offset)
    
    def encode_state(self, state):
#        if USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
        feature_encoding = [0] * self.num_tiles
            
#        if state.is_terminal():
#            return feature_encoding
        point1 = state.index[self.point1_name]
        point2 = state.index[self.point2_name]
        dist = point2.x - point1.x
        feature_index = self.get_tile_index(dist)
        feature_encoding[feature_index] = 1
        return feature_encoding 
        
class FeatureDistY(TiledFeature):
    
    def __init__(self, point1, point2, 
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES,
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point1_name = point1.name
        self.point2_name = point2.name
        point_range = point1.point_range
        max_dist = abs(point_range[1][1] - point_range[0][1])
        min_dist = -max_dist 
        super(FeatureDistY, self).__init__(min_dist, max_dist,
                num_tiles, offset)
    
    def get_name(self):
        return "dist-Y(%s-%s)[%1.f]" % (self.point1_name, self.point2_name,
                                        self.offset)

    def encode_state(self, state):
#        if USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
        feature_encoding = [0] * self.num_tiles
            
#        if state.is_terminal():
#            return feature_encoding
        point1 = state.index[self.point1_name]
        point2 = state.index[self.point2_name]
        dist = point2.y - point1.y
        feature_index = self.get_tile_index(dist)
        feature_encoding[feature_index] = 1
        return feature_encoding

#class FeaturePoint1D(TiledFeature):
#    
#    AXIS_X = "X"
#    AXIS_Y = "Y"
#    
#    def __init__(self, name, point, axis, 
#                 num_tiles = TiledFeature.DEFAULT_NUM_TILES, 
#                 offset = TiledFeature.NEUTRAL_OFFSET):
#        self.point_name = point.name
#        self.axis = axis
#        if self.axis == self.AXIS_X:
#            min_value = point.point_range[0][0]
#            max_value = point.point_range[1][0]
#        else:
#            min_value = point.point_range[0][1]
#            max_value = point.point_range[1][1]
#        super(FeaturePoint1D, self).__init__(name, min_value, max_value, 
#                                                  num_tiles, offset)
#            
#    def encode_state(self, state):
#        if USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
#        
#        point = state.index[self.point_name]
#            
##        if state.is_terminal():
##            return feature_encoding
#
#        if self.axis == self.AXIS_X:
#            value = point.x
#        else:
#            value = point.y 
#        feature_index = self.get_tile_index(value)
#        feature_encoding[feature_index] = 1
#        return feature_encoding 
#        

class FeaturePointX(TiledFeature):
    
    def __init__(self, point, 
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES, 
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point_name = point.name
        min_value = point.point_range[0][0]
        max_value = point.point_range[1][0]
        super(FeaturePointX, self).__init__(min_value, max_value, 
                                            num_tiles, offset)
    
    def get_name(self):
        return "pointX(%s)[%.1f]" % (self.point_name, self.offset)
    
    def encode_state(self, state):
#        if USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
        feature_encoding = [0] * self.num_tiles
        
        point = state.index[self.point_name]
            
#        if state.is_terminal():
#            return feature_encoding

        value = point.x
        feature_index = self.get_tile_index(value)
        feature_encoding[feature_index] = 1
        return feature_encoding 
        
class FeaturePointXY(Feature):
    
    def __init__(self, point,
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES * TiledFeature.DEFAULT_NUM_TILES, 
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point_name = point.name
        self.num_tiles = num_tiles
        self.num_tiles_1d = int(math.sqrt(num_tiles))
        self.offset = offset
        super(FeaturePointXY, self).__init__()
    
    def get_name(self):
        return "pointXY(%s)[%.1f]" % (self.point_name, self.offset)
    
    def get_num_tiles(self):
        return self.num_tiles

    def get_tile_index(self, value, min_value, max_value):
        offset_amount = ((self.offset - TiledFeature.NEUTRAL_OFFSET) * 2 * 
                         (max_value - min_value)) / self.num_tiles_1d
        value = value + offset_amount
        tile_index = int(float(value - min_value) * self.num_tiles_1d / 
                         (max_value - min_value))
        tile_index = min(tile_index, self.num_tiles_1d - 1)
        tile_index = max(tile_index, 0)
        return tile_index

    def get_encoding_length(self):
        return self.num_tiles
    
    def get_cost_factor(self):
        return 2

    def encode_state(self, state):
#        if USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
        feature_encoding = [0] * self.num_tiles
            
#        if state.is_terminal():
#            return feature_encoding
        
        point = state.index[self.point_name]
        
        x_tile_index = self.get_tile_index(point.x, 
                                           point.point_range[0][0],
                                           point.point_range[1][0])
        y_tile_index = self.get_tile_index(point.y, 
                                           point.point_range[0][1],
                                           point.point_range[1][1])

        feature_index = y_tile_index * self.num_tiles_1d + x_tile_index
        feature_encoding[feature_index] = 1
        return feature_encoding

class FeaturePointY(TiledFeature):
    
    def __init__(self, point, 
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES, 
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point_name = point.name
        min_value = point.point_range[0][1]
        max_value = point.point_range[1][1]
        super(FeaturePointX, self).__init__(min_value, max_value, 
                                            num_tiles, offset)
            
    def get_name(self):
        return "pointY(%s)[%.1f]" % (self.point_name, self.offset)
    
    def encode_state(self, state):
#        if USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
        feature_encoding = [0] * self.num_tiles
        
        point = state.index[self.point_name]
            
#        if state.is_terminal():
#            return feature_encoding

        value = point.y
        feature_index = self.get_tile_index(value)
        feature_encoding[feature_index] = 1
        return feature_encoding 
        
class FeatureInteraction(Feature):
    
    def __init__(self, base_feature_list):
        self.base_features = []
        for base_feature in base_feature_list:
            self.base_features += base_feature.get_underlying_features()
        
        num_tiles = 1
        name = "interaction("
        is_first = True
        for feature in self.base_features:
            num_tiles *= feature.get_num_tiles()
            if is_first:
                is_first = False
            else:
                name += " * "
            name += feature.get_name()
        name += ")"
        self.num_tiles = num_tiles
        self.name = name
        super(FeatureInteraction, self).__init__()
    
    def get_name(self):
        return self.name
    
    def get_underlying_features(self):
        return self.base_features
    
    def get_num_tiles(self):
        return self.num_tiles

    def get_encoding_length(self):
        return self.num_tiles
    
    def get_cost_factor(self):
        cost = 0
        for feature in self.base_features:
            cost += feature.get_cost_factor()
        return cost

    def encode_state(self, state):
#        if USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
        feature_encoding = [0] * self.num_tiles
        
        feature_index = 0
        multiplier = 1
        for feature in self.base_features:
            encoding = feature.encode_state(state)
            feature_index += multiplier * encoding.index(1)
            multiplier *= feature.get_encoding_length()    

        feature_encoding[feature_index] = 1
        return feature_encoding

class Featurizer(object):
    
    def __init__(self, state_vars):
        self.state_vars = state_vars
        
    def generate_feature(self):
        return NotImplemented
        
class FeaturizerFlag(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerFlag, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        flag_var = StateVarFlag.get_random_var(self.state_vars, [], is_dynamic=True)
        new_feature = FeatureFlag(flag_var)
        for feature in feature_list:
            if feature.get_name() == new_feature.get_name():
                return None
        return new_feature
        
class FeaturizerAngle(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerAngle, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        aux_point1 = StateVarPoint2D.get_random_var(self.state_vars, [main_point])
        aux_point2 = StateVarPoint2D.get_random_var(self.state_vars, [main_point, aux_point1])
        num_tiles = TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeatureAngle(main_point, aux_point1, aux_point2, num_tiles, offset)
        
class FeaturizerDist(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerDist, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        aux_point1 = StateVarPoint2D.get_random_var(self.state_vars, [main_point])
        num_tiles = TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeatureDist(main_point, aux_point1,
                           num_tiles, offset)
        
class FeaturizerDistX(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerDistX, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        aux_point1 = StateVarPoint2D.get_random_var(self.state_vars, [main_point])
        num_tiles = TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeatureDistX(main_point, aux_point1, 
                            num_tiles, offset)
        
class FeaturizerDistY(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerDistY, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        aux_point1 = StateVarPoint2D.get_random_var(self.state_vars, [main_point])
        num_tiles = TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeatureDistY(main_point, aux_point1, 
                            num_tiles, offset)
        
class FeaturizerPointXY(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerPointXY, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        num_tiles = TiledFeature.DEFAULT_NUM_TILES * TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeaturePointXY(main_point, num_tiles, offset)
    
class FeaturizerPointX(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerPointX, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        num_tiles = TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeaturePointX(main_point, num_tiles, offset)
        
class FeaturizerPointY(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerPointY, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        num_tiles = TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeaturePointX(main_point, num_tiles, offset)
        
class FeaturizerInteraction(Featurizer):
    
    MAX_NUM_FEATURES = 3
    MAX_DEGREE = 4
    
    def __init__(self, state_vars):
        super(FeaturizerInteraction, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        new_feature = None
        if len(feature_list) != 0:
            rand1_index = int(random.random() * len(feature_list))
            rand2_index = int(random.random() * len(feature_list))
            feature1 = feature_list[rand1_index]
            feature2 = feature_list[rand2_index]
            # don't create interactions of the same feature
            if feature1.get_name() != feature2.get_name():
                # don't create interactions of size more than MAX_NUM_FEATURES
                size1 = len(feature1.get_underlying_features())
                size2 = len(feature2.get_underlying_features())
                if size1 + size2 <= self.MAX_NUM_FEATURES:
                    # don't create interactions with more tiles than 10 ^ MAX_DEGREE
                    encoding_length1 = feature1.get_encoding_length()
                    encoding_length2 = feature2.get_encoding_length()
                    if (encoding_length1 * encoding_length2) < (TiledFeature.DEFAULT_NUM_TILES ** self.MAX_DEGREE): 
                        feature1_copy = copy.deepcopy(feature1)
                        feature2_copy = copy.deepcopy(feature2)
                        new_feature = FeatureInteraction([feature1_copy, feature2_copy])
        return new_feature

class FeaturizerRetile(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerRetile, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        if len(feature_list) == 0:
            return None
        rand_index = int(random.random() * len(feature_list))
        feature = feature_list[rand_index]
        new_feature = copy.deepcopy(feature)
        new_feature.reconfigure()
        return new_feature

class Mutator(object):
    
    def __init__(self, featurizers_map, state_vars):
        self.featurizers_map = featurizers_map
        self.state_vars = state_vars
        
    def select_featurizer(self):
        rand_number = random.random()
        selected_featurizer = None
        cumulative_prob = 0.0
        for (prob, featurizer) in self.featurizers_map:
            cumulative_prob += prob
            if selected_featurizer is None:
                if rand_number < cumulative_prob:
                    selected_featurizer = featurizer
        return selected_featurizer
    
#    def mutate_old(self, agent):
#        feature_list = agent.feature_set.get_feature_list_copy()
#        
#        new_feature = None
#        while new_feature is None:
#            featurizer = self.select_featurizer()
#            new_feature = featurizer.generate_feature(feature_list)
#            
#        feature_list.append(new_feature)
#        new_feature_set = FeatureSet(feature_list)
#        
#        agent_class = agent.__class__
#        new_agent = agent_class(new_feature_set)
#        
#        if MUTATE_NEW_FEATURE_WEIGHTS == WEIGHTS_OPTIMISTIC or len(feature_list) == 1:
#            new_segment_weights = ((agent.environment.get_max_episode_reward() * 
#                              INIT_Q_VALUE_MULTIPLIER) / 
#                              new_feature_set.get_num_features())
#        else:
#            new_segment_weights = 0
#        
#        for action in agent.all_actions():
#            new_agent.algorithm.w[action] = agent.algorithm.w[action] + \
#                    [new_segment_weights] * new_feature.get_encoding_length()
#        
#        return new_agent
        
    def mutate(self, agent):
        feature_list = agent.feature_set.feature_list

        new_feature = None
        while new_feature is None:
            featurizer = self.select_featurizer()
            new_feature = featurizer.generate_feature(feature_list)
            
        new_agent = agent.clone()
        new_agent.add_feature(new_feature)        

        return new_agent
        
#    def cross_over_old(self, agent1, agent2):
#        new_feature_list = agent1.feature_set.get_feature_list_copy()
#        feature_list_from_agent2 = agent2.feature_set.get_feature_list_copy()
#        last_feature_from_agent2 = feature_list_from_agent2[-1]
#        new_feature_list.append(last_feature_from_agent2)
#
#        new_feature_set = FeatureSet(new_feature_list)
#        
#        agent_class = agent1.__class__
#        new_agent = agent_class(new_feature_set)
#        
#        for action in agent1.actions.all_actions():
#            len_last_feature_from_agent2 = \
#                last_feature_from_agent2.get_encoding_length()
#            if MUTATE_CROSS_OVER_WEIGHTS == WEIGHTS_COPY:
#                feature_w_from_agent_2 = \
#                    agent2.algorithm.w[action][-len_last_feature_from_agent2:]
#                new_agent.algorithm.w[action] = \
#                    agent1.algorithm.w[action] + feature_w_from_agent_2
#            else:
#                new_agent.algorithm.w[action] = \
#                    agent1.algorithm.w[action] + [0] * len_last_feature_from_agent2
#        
#        return new_agent

    def cross_over(self, agent1, agent2):
        feature_list_from_agent2 = agent2.feature_set.feature_list
        last_feature_from_agent2 = feature_list_from_agent2[-1]

        new_agent = agent1.clone()
        new_agent.add_feature(last_feature_from_agent2)        

        return new_agent
        
class GeometryUtil(object):
    
    @classmethod
    def compute_dist(cls, x1, y1, x2, y2):
        diff_y = y2 - y1
        diff_x = x2 - x1
        dist = math.sqrt(diff_x * diff_x + diff_y * diff_y)
        return dist
    
    @classmethod
    def compute_angle(cls, x1, y1, x2, y2, x3, y3):
        diff_x1_x2 = x1 - x2
        diff_y1_y2 = y1 - y2
        diff_x1_x3 = x1 - x3
        diff_y1_y3 = y1 - y3
        diff_x2_x3 = x2 - x3
        diff_y2_y3 = y2 - y3
        
        a2 = diff_x1_x2 * diff_x1_x2 + diff_y1_y2 * diff_y1_y2
        b2 = diff_x1_x3 * diff_x1_x3 + diff_y1_y3 * diff_y1_y3
        c2 = diff_x2_x3 * diff_x2_x3 + diff_y2_y3 * diff_y2_y3
        
        b = math.sqrt(b2)
        a = math.sqrt(a2)
        
        if a2 == 0 or b2 == 0:
            angle = math.pi / 2
        else:
            cos = (a2 + b2 - c2) / (2 * a * b)
            if cos > 1.0:
                cos = 1.0
            elif cos < -1.0:
                cos = -1.0
            angle = math.acos(cos)
        
        return angle

    @classmethod
    def compute_dist_point_line(cls, x3, y3, x1, y1, x2, y2):
        # adapted code from stackoverflow.com
        t = x2 - x1, y2 - y1           # Vector ab
        dd = math.sqrt(t[0]**2 + t[1]**2)         # Length of ab
        t = t[0] / dd, t[1] / dd               # unit vector of ab
        n = -t[1], t[0]                    # normal unit vector to ab
        ac = x3 - x1, y3 - y1          # vector ac
        return math.fabs(ac[0] * n[0] + ac[1] * n[1]) # Projection of ac to n (the minimum distance)

class Algorithm(object):

    def __init__(self, actions, environment):
        self.actions = actions
        self.environment = environment
        if self.environment is not None:
            self.gamma = environment.gamma
        else:
            self.gamma = DEFAULT_GAMMA
        
        self.cached_action_valid = False
        self.cached_action = None
        self.cached_action_value = None
        
        self.is_learning = True
    
    def begin_episode(self, state):
        pass
    
    def select_action(self, state):
        if self.cached_action_valid:
            return (self.cached_action, self.cached_action_value)
        else:
            (a, v) = self.select_action_do(state)
            self.cached_action = a
            self.cached_action_value = v
            self.cached_action_valid = True
        return (a, v)
    
    def select_action_do(self, state):
        return NotImplemented
    
    def transition(self, state, action, reward, state_p, action_p):
#        self.cached_action = None
#        self.cached_action_value = None
#        self.cached_action_valid = False
        pass

    def pause_learning(self):
        self.is_learning = False
        
    def resume_learning(self):
        self.is_learning = True    

    def reset_learning(self):
        pass

#class RandomAlgorithm(Algorithm):
#    
#    def select_action(self):
#        return (self.agent.actions.random_action(), None)

class Sarsa(Algorithm):

    DEFAULT_ALPHA = 0.1
    DEFAULT_EPSILON = 0.05
    DEFAULT_LAMBDA = 0.9
    
    ACCUMULATING_TRACES = 1
    REPLACING_TRACES = 2
    TRACES = REPLACING_TRACES

    def __init__(self, actions, environment, alpha, epsilon, lamda):
        super(Sarsa, self).__init__(actions, environment)
        self.epsilon = epsilon
        self.lamda = lamda
        self.alpha = alpha
        
class SarsaLambda(Sarsa):

    def __init__(self, actions, environment,
                 alpha = Sarsa.DEFAULT_ALPHA, 
                 epsilon = Sarsa.DEFAULT_EPSILON,
                 lamda = Sarsa.DEFAULT_LAMBDA):
        super(SarsaLambda, self).__init__(actions, environment, alpha, epsilon, lamda)
        self.Q = {}
        self.Q_save = {}
        self.e = {}
        self.default_q = self.environment.get_max_episode_reward() * \
                INIT_Q_VALUE_MULTIPLIER
        
        # set values for terminal state
#        terminal_state_repr = str(self.environment.generate_terminal_state())
#        for action in self.agent.all_actions():
#            self.Q[(terminal_state_repr, action)] = 0
    
    def begin_episode(self, state):
        self.e = {}
    
    def select_action_do(self, state):
        state_s = str(state)
        
        if self.is_learning and (random.random() < self.epsilon):
            action = self.actions.random_action()
            value = self.Q.get((state_s, action), self.default_q) 
        else:
            action_values = []
            for action in self.actions.all_actions():
                # insert a random number to break the ties
                action_values.append(((self.Q.get((state_s, action), self.default_q), 
                                       random.random()), action))
                
            action_values_sorted = sorted(action_values, reverse=True)
            
            action = action_values_sorted[0][1]
            value = action_values_sorted[0][0][0]
        
        return (action, value)

    def update_values(self, delta):
        for (si, ai) in self.e.iterkeys():
            self.Q[(si, ai)] = self.Q.get((si, ai), self.default_q) + \
                    self.alpha * delta * self.e[(si, ai)]

    def transition(self, state, action, reward, state_p, action_p):
        super(SarsaLambda, self).transition(state, action, reward,
                                            state_p, action_p)
        
        if not self.is_learning:
            return
        
        s = str(state)
        a = action
        sp = str(state_p)
        ap = action_p
        
        if state_p.is_final():
            delta = reward - self.Q.get((s, a), self.default_q)
        else:
            delta = reward + self.gamma * self.Q.get((sp, ap), self.default_q) - \
                    self.Q.get((s, a), self.default_q)

        # update eligibility trace
        if self.TRACES == self.REPLACING_TRACES:
            for (si, ai) in self.e.iterkeys():
                if si == s and ai == a:
                    self.e[(s, a)] = (self.e.get((s, a), 0) * 
                                      self.gamma * self.lamda) + 1
                elif si == s and ai != a:
                    self.e[(si, ai)] = 0
                else:
                    self.e[(si, ai)] *= self.gamma * self.lamda
        else: # TRACES == ACCUMULATING_TRACES:
            self.Q[(s, a)] = self.Q.get((s, a), self.default_q) + \
                    self.alpha * delta * self.e[(s, a)]
            for (si, ai) in self.e.iterkeys():
                if si == s and ai == a:
                    self.e[(s, a)] = (self.e.get((s, a), 0) * 
                                      self.gamma * self.lamda) + 1
                else:
                    self.e[(si, ai)] *= self.gamma * self.lamda

        self.update_values(delta)

#    def end_episode(self, state, action, reward):
#        s = str(state)
#        a = action
#        delta = reward - self.Q.get((s, a), self.default_q)
#        self.update_values(delta)
    
#    def do_episode(self, start_state):
#        self.agent.episode_trace = ""
#        self.e = {}
#        episode_reward = 0
#
#        self.agent.begin_episode(start_state)
#        s_obj = self.agent.state
#        s = str(s_obj)
#        if DEBUG_EPISODE_TRACE:
#            self.agent.episode_trace += "starting at %s\n" % s
#        (a, v) = self.select_action()
#        while not self.agent.is_episode_ended():
#            r = self.agent.transition(a)
#            episode_reward += r
#            sp_obj = self.agent.state
#            sp = str(sp_obj)
#            (ap, vp) = self.select_action()
#            if DEBUG_EPISODE_TRACE:
#                self.agent.episode_trace += " (%s) -> %s\n" % (a, sp)
#            delta = r + self.gamma * self.Q.get((sp, ap), self.default_q) - \
#                    self.Q.get((s, a), self.default_q)
#
#            # update eligibility trace
#            if self.TRACES == self.REPLACING_TRACES:
#                self.e[(s, a)] = (self.e.get((s, a), 0) * 
#                                  self.gamma * self.lamda) + 1
#                self.Q[(s, a)] = self.Q.get((s, a), self.default_q) + \
#                        self.alpha * delta * self.e[(s, a)]
#                for (si, ai) in self.e.iterkeys():
#                    if si == s and ai == a:
#                        pass
#                    elif si == s and ai != a:
#                        self.e[(si, ai)] = 0
#                    else:
#                        self.e[(si, ai)] *= self.gamma * self.lamda
#                        self.Q[(si, ai)] = self.Q.get((si, ai), self.default_q) + \
#                                self.alpha * delta * self.e[(si, ai)]
#            else: # TRACES == ACCUMULATING_TRACES:
#                self.e[(s, a)] = (self.e.get((s, a), 0) * 
#                                  self.gamma * self.lamda) + 1
#                self.Q[(s, a)] = self.Q.get((s, a), self.default_q) + \
#                        self.alpha * delta * self.e[(s, a)]
#                for (si, ai) in self.e.iterkeys():
#                    if si == s and ai == a:
#                        pass
#                    else:
#                        self.e[(si, ai)] *= self.gamma * self.lamda
#                        self.Q[(si, ai)] = self.Q.get((si, ai), self.default_q) + \
#                                self.alpha * delta * self.e[(si, ai)]
#
#            s_obj = sp_obj
#            s = sp
#            a = ap
#
#        return episode_reward

    def save_learning_state(self):
        self.Q_save = copy.deepcopy(self.Q)
        
    def restore_learning_state(self):
        self.Q = copy.deepcopy(self.Q_save)

    def print_Q(self):
        Q_keys = self.Q.keys()
        Q_keys.sort()
        print "Q:"
        for key in Q_keys:
            print "Q%s -> %.2f" % (key, self.Q[key])

    def print_e(self):
        e_keys = self.e.keys()
        e_keys.sort()
        print "e:"
        for key in e_keys:
            print "e%s -> %.2f" % (key, self.e[key])

    def print_values(self):
        self.print_Q()
        self.print_e()

class SarsaLambdaFeaturized(Sarsa):

    def __init__(self, actions, environment, feature_set,
                 alpha = Sarsa.DEFAULT_ALPHA, 
                 epsilon = Sarsa.DEFAULT_EPSILON,
                 lamda = Sarsa.DEFAULT_LAMBDA):
        super(SarsaLambdaFeaturized, self).__init__(actions, environment, alpha,
                                                    epsilon, lamda)
        self.w = {}
        self.w_save = {}
        self.e = {}

        self.feature_set = feature_set
        self.init_weights()

    def init_weights(self):
        num_features = self.feature_set.get_num_features()
        if BASE_FEATURE_WEIGHTS == WEIGHTS_OPTIMISTIC and \
                (self.feature_set.get_num_features() > 0):
            optimistic_reward = (self.environment.get_max_episode_reward() * 
                                 INIT_Q_VALUE_MULTIPLIER)
            self.default_w = float(optimistic_reward) / num_features
#            self.default_w = optimistic_reward
        else:
            self.default_w = 0
            
        for action in self.actions.all_actions():
#            if USE_NUMPY:
#                self.w[action] = numpy.ones(
#                        self.feature_set.get_encoding_length()) * self.default_w
#            else:
#                self.w[action] = [self.default_w] * self.feature_set.get_encoding_length()
            self.w[action] = [self.default_w] * self.feature_set.get_encoding_length()
    
    def reset_learning(self):
        self.init_weights()
    
    def add_feature(self, feature):
        # feature_set has already received the new feature
        num_features = self.feature_set.get_num_features()
#        if MUTATE_NEW_FEATURE_WEIGHTS == WEIGHTS_OPTIMISTIC:
#        else:
#            new_segment_weights = 0
        optimistic_reward = (self.environment.get_max_episode_reward() *
                             MUTATE_NEW_WEIGHTS_MULT * 
                             INIT_Q_VALUE_MULTIPLIER)
        new_segment_weights = float(optimistic_reward) / num_features
        
        # adjust existing weights
        if MUTATE_NEW_WEIGHTS_MULT > 0.0:
#            multiplier = float(num_features - 1) / float(num_features)
            # now we have to factor in MUTATE_OPTIMISTIC_WEIGHT_MULTIPLIER
            # 1 2 ... n          n+1
            # w w w w w           W
            # 
            # * n/(n+1) * x     * 1/(n+1) * M
            # 
            # n/(n+1) + 1/(n+1) = 1
            #
            # n/(n+1) * x + 1/n+1 * M = 1
            # n/(n+1) * x = 1 - M/(n+1)
            #     1 - (M/n+1)
            # x = -----------
            #      n / (n+1)            
            multiplier = 1.0 - (float(MUTATE_NEW_WEIGHTS_MULT) / num_features)
            for action in self.actions.all_actions():
                self.w[action][:] = [w * multiplier for w in self.w[action]]
        
        # add new weights
        for action in self.actions.all_actions():
            self.w[action] += [new_segment_weights] * feature.get_encoding_length()

    def begin_episode(self, state):
#        if USE_NUMPY:
#            for action in self.agent.all_actions():
#                self.e[action] = numpy.zeros(self.feature_set.get_encoding_length())
#        else:
#            for action in self.agent.all_actions():
#                self.e[action] = [0] * self.feature_set.get_encoding_length()
        for action in self.actions.all_actions():
            self.e[action] = [0] * self.feature_set.get_encoding_length()
                
    def compute_Q(self, features_present, action):
        q_sum = 0
#        if USE_NUMPY:
#            sum = numpy.dot(self.w[action], features_present)
#        else:
#            for i in range(self.feature_set.get_encoding_length()):
#                sum += self.w[action][i] * features_present[i]
        for i in range(self.feature_set.get_encoding_length()):
            q_sum += self.w[action][i] * features_present[i]
        return q_sum
    
    def select_action_do(self, state):
        features = self.feature_set.encode_state(state)
        
        if self.is_learning and (random.random() < self.epsilon):
            action = self.actions.random_action()
            value = self.compute_Q(features, action) 
        else:
            action_values = []
            for action in self.actions.all_actions():
                # insert a random number to break the ties
                action_values.append(((self.compute_Q(features, action), 
                                       random.random()), action))
                
            action_values_sorted = sorted(action_values, reverse=True)
            
            action = action_values_sorted[0][1]
            value = action_values_sorted[0][0][0]
        
        return (action, value)

    def update_weights(self, delta):
        alpha = self.alpha / self.feature_set.get_num_features()
        for action in self.actions.all_actions():
            if USE_NUMPY:
                self.w[action] += (self.e[action] * alpha * delta)
            else:
#                self.w[action] += [alpha * delta * self.e[action][i]
#                                         for i in get_encoding_length(self.w[action])] 
                for i in range(len(self.w[action])):
                    self.w[action][i] += (alpha * delta * self.e[action][i])
    
    def transition(self, state, action, reward, state_p, action_p):
        super(SarsaLambdaFeaturized, self).transition(state, action, reward, 
                                                      state_p, action_p)

        if not self.is_learning:
            return
        
        s = str(state) #@UnusedVariable
        a = action
        sp = str(state_p) #@UnusedVariable
        ap = action_p #@UnusedVariable

        Fa = self.feature_set.encode_state(state)
        
        # update e
        for action in self.actions.all_actions():
        #                self.e[action] = [e * self.gamma * self.lamda 
        #                                  for e in self.e[action]]
            for i in range(len(self.e[action])):
                self.e[action][i] *= (self.gamma * self.lamda)
                
        for i in range(len(Fa)):
            if Fa[i] == 1:
                # replacing traces
                self.e[a][i] = 1
                # set the trace for the other actions to 0
                for action in self.actions.all_actions():
                    if action != a:
                        self.e[action][i] = 0

#        sigma_w_Fa = 0
#        if USE_NUMPY:
#            sigma_w_Fa = numpy.dot(Fa, self.w[a])
#        else:
#            sigma_w_Fa = self.compute_Q(Fa, a)
        sigma_w_Fa = self.compute_Q(Fa, a)


#            for i in range(get_encoding_length(Fa)):
#                if Fa[i] == 1:
#                    sigma_w_Fa += self.w[a][i]
        
        if state_p.is_final():
            delta = reward - sigma_w_Fa
        else:
            # select next action
#            Q_a = self.cached_action_value
            if self.cached_action_valid:
                Q_a = self.cached_action_value
            else:
                (ap, Q_a) = self.select_action(state_p) #@UnusedVariable
#            (ap, Q_a) = self.select_action()
            delta = reward + self.gamma * Q_a - sigma_w_Fa

        self.update_weights(delta)
        
#    def do_episode(self, start_state):
#        self.agent.episode_trace = ""
#        if USE_NUMPY:
#            for action in self.agent.all_actions():
#                self.e[action] = numpy.zeros(self.feature_set.get_encoding_length())
#        else:
#            for action in self.agent.all_actions():
#                self.e[action] = [0] * self.feature_set.get_encoding_length()
#        episode_reward = 0
#
#        self.agent.begin_episode(start_state)
#        s = self.agent.state
#        Fa = self.feature_set.encode_state(s)
#        if DEBUG_EPISODE_TRACE:
#            self.agent.episode_trace += "starting at %s\n" % s
#        (a, Q_a) = self.select_action()
#        while not self.agent.is_episode_ended():
#            # update e
#            for action in self.agent.all_actions():
##                self.e[action] = [e * self.gamma * self.lamda 
##                                  for e in self.e[action]]
#                for i in range(get_encoding_length(self.e[action])):
#                    self.e[action][i] *= (self.gamma * self.lamda)
#                    
#            for i in range(get_encoding_length(Fa)):
#                if Fa[i] == 1:
#                    # replacing traces
#                    self.e[a][i] = 1
#                    # set the trace for the other actions to 0
#                    for action in self.agent.all_actions():
#                        if action != a:
#                            self.e[action][i] = 0
# 
#            r = self.agent.transition(a)
#            episode_reward += r
#            sp = self.agent.state
#
#            if DEBUG_EPISODE_TRACE:
#                self.agent.episode_trace += " (%s) -> %s\n" % (a, sp)
#            
#            sigma_w_Fa = 0
#            if USE_NUMPY:
#                sigma_w_Fa = numpy.dot(Fa, self.w[a])
#            else:
#                sigma_w_Fa = self.compute_Q(Fa, a)
##                for i in range(get_encoding_length(Fa)):
##                    if Fa[i] == 1:
##                        sigma_w_Fa += self.w[a][i]
#
#            delta = r - sigma_w_Fa 
#            if sp.is_terminal():
#                self.update_weights(delta)
#                if DEBUG_ALG_VALUES:
#                    self.agent.print_values()
#                    self.print_episode_log()
#                break
#            
#            # select next action
#            (ap, Q_a) = self.select_action()
#            
#            delta = delta + self.gamma * Q_a
#
#            self.update_weights(delta)
#            
#            s = sp
#            a = ap
#            Fa = self.feature_set.encode_state(s)
#
#        return episode_reward
         
    def save_learning_state(self):
        self.w_save = copy.deepcopy(self.w)
        
    def restore_learning_state(self):
        self.w = copy.deepcopy(self.w_save)
        
    def print_w(self):
        w_keys = self.w.keys()
        w_keys.sort()
        for key in w_keys:
            print "w[%s]: " % (key),
            for i in range(len(self.w[key])):
                print "%.2f" % self.w[key][i],
            print

    def print_e(self):
        e_keys = self.e.keys()
        e_keys.sort()
        for key in e_keys:
            print "e[%s]: " % (key),
            for i in range(len(self.e[key])):
                print "%.2f" % self.e[key][i],
            print

    def print_values(self):
        self.print_w()
        self.print_e()

#class ExperimentParameters(object):
#    def __init__(self, num_generations=15, population_size=50,
#                 num_episodes=200,  num_champion_trials=20,
#                 optimistic_weights_multiplier=0.0, eta = 1.0):
#        self.num_generations = num_generations
#        self.population_size = population_size
#        self.num_episodes = num_episodes
#        self.num_champion_trials = num_champion_trials
#        self.optimistic_weights_multiplier = optimistic_weights_multiplier
#        self.eta = eta

# multiprocessing method
def arbitrator_do_episode((agent, start_state, max_steps)):
    agent.begin_episode(start_state)

    steps = 0
    while not agent.state.is_final():
        a = agent.select_action()
        r = agent.transition(a) #@UnusedVariable
        steps += 1
        if (max_steps != 0) and (steps >= max_steps):
            break
    
    episode_reward = agent.episode_reward
    return (episode_reward, steps)
    
# multiprocessing method
def arbitrator_test_agent((agent, start_states, start_seeds, max_steps,
                           num_episodes, num_trials, do_episode_func)):
    if DEBUG_PROGRESS and not USE_MULTIPROCESSING:
        print "testing agent: " + str(agent.feature_set)
    agent.reward_log = [0] * num_episodes
    begin_time = time.clock()
    agent.save_learning_state()
    for trial in range(num_trials): #@UnusedVariable
        trial_reward = 0
        agent.restore_learning_state()
        for episode in range(num_episodes):
            start_state_index = (episode + trial) % num_episodes 
            # get start state
            start_state = copy.deepcopy(start_states[start_state_index])
            # seed random number generator
            #random.seed(start_seeds[episode])
            random.seed(start_seeds[start_state_index])
            (episode_reward, steps) = do_episode_func((agent, start_state, max_steps))
            agent.reward_log[episode] += episode_reward
            trial_reward += episode_reward
    
            if DEBUG_EPISODE_REWARD:
                print "episode %i: reward %.2f, steps:%d" % (episode,
                                                episode_reward, steps)
            if DEBUG_ALG_VALUES:
                print "values:"
                agent.algorithm.print_values()
        if DEBUG_PROGRESS and (num_trials != 1):
            print "trial reward: %.2f" % (float(trial_reward) / num_episodes) 
    end_time = time.clock()
    agent.training_time = end_time - begin_time
    
    # average rewards over trials
    for episode in range(num_episodes):
        agent.reward_log[episode] = float(agent.reward_log[episode]) / num_trials
    
#    agent.average_reward = float(sum(agent.reward_log)) / num_episodes
    trailing_rewards = agent.reward_log[int(TRAINING_SLACK * num_episodes):]
    agent.average_reward = float(sum(trailing_rewards)) / len(trailing_rewards)

    if DEBUG_PROGRESS:
        if USE_MULTIPROCESSING:
            print "tested agent: " + str(agent.feature_set)
        print "average reward: %.2f, training time: %.1fs" % (
                agent.average_reward, agent.training_time)

#    gc.collect()
    return agent
        
class Arbitrator(object):

    def __init__(self):
        pass
    
    def run(self, max_steps = 0):
        return NotImplemented
    
    def test_agents_seedless(self, agents, max_steps, num_episodes,
                             use_common_states):
        if DEBUG_PROGRESS:
            print "generating start states"

        start_states = {}
        start_seeds = {}
        for trial in range(len(agents)):
            start_states[trial] = []
            start_seeds[trial] = []
                      
        for episode in range(num_episodes): #@UnusedVariable
            state = agents[0].environment.generate_start_state()
            seed = random.random()
            for trial in range(len(agents)):
                if not use_common_states:
                    state = agents[0].environment.generate_start_state()
                    seed = random.random()
                start_states[trial].append(state)
                start_seeds[trial].append(seed)
        
        return self.test_agents(agents, start_states, start_seeds, max_steps,
                                num_episodes)
    
    def test_agents(self, agents, start_states, start_seeds, max_steps,
                    num_episodes):

        if DEBUG_PROGRESS:
            print "testing the agents"

        # experiment with agents
        if USE_MULTIPROCESSING:
            pool = multiprocessing.Pool(processes=NUM_CORES)
            params = []
            for run_case in range(len(agents)):
                params.append((agents[run_case], start_states[run_case], 
                               start_seeds[run_case], max_steps,
                               num_episodes, 1, arbitrator_do_episode))
            updated_agents = pool.map(arbitrator_test_agent, params)
            agents = updated_agents
        else:
            for run_case in range(len(agents)):
                self.test_agent(agents[run_case], start_states[run_case],
                                start_seeds[run_case], max_steps,
                                num_episodes, 1) 
                
        return agents       

    def test_agent(self, agent, start_states, start_seeds, max_steps,
                   num_episodes, num_trials):
        return arbitrator_test_agent((agent, start_states, start_seeds, max_steps,
                                     num_episodes, num_trials,
                                     arbitrator_do_episode))
        
    def do_episode(self, agent, start_state, max_steps):
        return arbitrator_do_episode((agent, start_state, max_steps))

    def plot(self, folder_name, script_name, parameters):
        orig_wd = os.getcwd()
        os.chdir(folder_name)
        if DEBUG_PROGRESS:
            print "generating plot"
        subprocess.call('gnuplot ../../plot/%s.gp' % script_name, shell=True)
#        subprocess.call('../../plot/%s.sh %s' % (script_name, parameters), shell=True)
        os.chdir(orig_wd)

class ArbitratorStandard(Arbitrator):
    
    def __init__(self, agent, num_trials, num_episodes):
        super(ArbitratorStandard, self).__init__()
        self.agent = agent
        self.num_trials = num_trials
        self.num_episodes = num_episodes
        self.reward_log = []
        self.eval_reward_log = []

    def run(self, max_steps = 0):
        self.reward_log = [0] * self.num_episodes
        if DEBUG_PROGRESS:
            print "generating %d copies of the agent" % self.num_trials
            
        trial_agents = []
        for i in range(self.num_trials): #@UnusedVariable
            trial_agents.append(copy.deepcopy(self.agent))

        # run multiple trials with different start states
        trial_agents = self.test_agents_seedless(trial_agents, max_steps,
                                                 self.num_episodes, 
                                                 use_common_states=False)
        
        # average rewards
        for agent in trial_agents:
            for episode in range(self.num_episodes):
                self.reward_log[episode] += agent.reward_log[episode]
        
        for episode in range(self.num_episodes):
            self.reward_log[episode] /= float(self.num_trials)

        
        # find agent with best performance
        best_agent = None
        best_agent_reward = 0
        for agent in trial_agents:
            if best_agent is None or agent.average_reward > best_agent_reward:
                best_agent = agent
                best_agent_reward = agent.average_reward
        
#        trained = self.test_agents_seedless([self.agent], max_steps,
#                                             self.num_episodes, 
#                                             use_common_states=False)
# 
#        self.agent = trained[0]
#               
#        self.reward_log = self.agent.reward_log

        self.agent = best_agent
        
        if DEBUG_PROGRESS:
            print ""
            print "best eval agent: %s" % self.agent.feature_set
            print "with average reward: %.4f" % self.agent.average_reward
        if DEBUG_ALG_VALUES:
            print "values:"    
            self.agent.algorithm.print_w()

        # evaluate it
        self.agent.pause_learning()
        
        self.eval_reward_log = [0] * self.num_episodes
        if DEBUG_PROGRESS:
            print "generating %d copies of the agent" % self.num_trials
            
        trial_agents = []
        for i in range(self.num_trials): #@UnusedVariable
            trial_agents.append(copy.deepcopy(self.agent))

        # run multiple trials with different start states
        trial_agents = self.test_agents_seedless(trial_agents, max_steps,
                                                 self.num_episodes, 
                                                 use_common_states=False)
        
        # average rewards
        for agent in trial_agents:
            for episode in range(self.num_episodes):
                self.eval_reward_log[episode] += agent.reward_log[episode]
        
        for episode in range(self.num_episodes):
            self.eval_reward_log[episode] /= float(self.num_trials)

        if REPORT_RESULTS:
            self.report_results()
        
#    def run_old_non_parallel(self, max_steps = 0):
#        self.reward_log = [0] * self.num_episodes
#        self.agent.save_learning_state()
#        for trial in range(self.num_trials):
#            self.agent.restore_learning_state()
#            trial_reward = 0
#            for episode in range(self.num_episodes):
#                if DEBUG_PROGRESS and (episode % DEBUG_REPORT_ON_EPISODE == 0):
#                    print "trial %i episode %i" % (trial, episode)
##                    print agent.w
#                start_state = self.agent.environment.generate_start_state()
#                (episode_reward, steps) = self.do_episode(self.agent, start_state, max_steps)
#                trial_reward += episode_reward
#                self.reward_log[episode] += episode_reward
#                if DEBUG_EPISODE_REWARD:
#                    print "episode %i: reward %.2f, steps:%d" % (episode, 
#                                                    episode_reward, steps)
##                    print "trace:"
##                    print self.agent.get_episode_trace()
#                if DEBUG_ALG_VALUES:
#                    print "values:"
#                    self.agent.algorithm.print_values()
#        
#        if DEBUG_PROGRESS:
#            print "average reward: %.2f" % (float(trial_reward) / self.num_episodes) 
#        
#        if REPORT_RESULTS:
#            self.report_results()
        
    def report_results(self):
        program_name = sys.argv[0]
        program_name_parts = program_name.rsplit(".", 1)
        program_just_name = program_name_parts[0]        
        folder_name = 'results/%s-t%de%dw%d/' % (program_just_name, 
                                              self.num_trials, self.num_episodes,
                                              MUTATE_NEW_WEIGHTS_MULT * 100)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
        report_file = open(folder_name + 'results-standard-training.txt', 'w')
        for episode in range(self.num_episodes):
            report_file.write('%d %.2f\n' % 
                              (episode, self.reward_log[episode])) 
        report_file.close()
    
        report_file = open(folder_name + 'results-standard-training-interval.txt', 'w')
        episodes_per_interval = int (self.num_episodes / PLOT_INTERVALS) 
        for interval in range(PLOT_INTERVALS):
            sub_sum = sum(self.reward_log[interval * episodes_per_interval:
                                     (interval + 1) * episodes_per_interval])
            report_file.write('%d %.2f\n' % 
                              (interval * episodes_per_interval, 
                               float(sub_sum) / episodes_per_interval)) 
        report_file.close()
        
        report_file = open(folder_name + 'results-standard-eval.txt', 'w')
        for episode in range(self.num_episodes):
            report_file.write('%d %.2f\n' % 
                              (episode, self.eval_reward_log[episode])) 
        report_file.close()
    
        report_file = open(folder_name + 'results-standard-eval-interval.txt', 'w')
        episodes_per_interval = int (self.num_episodes / PLOT_INTERVALS) 
        for interval in range(PLOT_INTERVALS):
            sub_sum = sum(self.eval_reward_log[interval * episodes_per_interval:
                                     (interval + 1) * episodes_per_interval])
            report_file.write('%d %.2f\n' % 
                              (interval * episodes_per_interval, 
                               float(sub_sum) / episodes_per_interval)) 
        report_file.close()

        self.plot(folder_name, 'plot-standard', 
                  '%d %d' % (self.num_trials, self.num_episodes))

class ArbitratorEvolutionary(Arbitrator):
    
    def __init__(self, base_agent, featurizers_map, num_generations, 
                 population_size, num_generation_episodes, num_champion_trials, 
                 num_best_champion_episodes, num_best_champion_trials, eta):
        super(ArbitratorEvolutionary, self).__init__()
        self.base_agent = base_agent
        self.featurizers_map = featurizers_map
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_generation_episodes = num_generation_episodes
        self.num_champion_trials = num_champion_trials
        self.num_best_champion_episodes = num_best_champion_episodes
        self.num_best_champion_trials = num_best_champion_trials
        self.eta = eta
        
        self.champions = []
        self.champion_training_rewards = []
        self.champion_training_rewards_normalized = []
        self.champion_training_times = []
        self.champion_eval_rewards = []
        self.champions_training_reward_log = []
        self.champions_eval_reward_log = []
        self.population_reward_log = []
        self.best_champion_reward_log = []

        # check integrity of the featurizers map
        sum_prob = 0.0
        for (prob, featurizer) in featurizers_map: #@UnusedVariable
            sum_prob += prob
        print "Initialized %d featurizers" % len(featurizers_map)
        if round(sum_prob, 4) != 1.0:
            print "Aborting: sum of selection probabilities is %.2f" % sum_prob
            sys.exit(-1)
    
    def run(self, max_steps = 0):
        
        sample_state = self.base_agent.environment.generate_start_state()
        state_vars = sample_state.state_variables
        
        mutator = Mutator(self.featurizers_map, state_vars)
        
        for action in self.base_agent.all_actions():
            self.base_agent.algorithm.w[action] = []
            
        surviving_agents = []
        
        # initial population
        while len(surviving_agents) < self.population_size:
            new_agent = mutator.mutate(self.base_agent)
            surviving_agents.append(new_agent)
            
        self.champions_training_reward_log = []
        self.population_reward_log = [0] * (self.num_generations * self.num_generation_episodes)
        
        for generation in range(self.num_generations):
            if DEBUG_PROGRESS:
                print "generation %i" % (generation)
    
            # mutate agents
            highest_reward = surviving_agents[0].average_reward_normalized
            lowest_reward = surviving_agents[len(surviving_agents) - 1].average_reward_normalized
#            if highest_reward == 0:
#                highest_reward = 1
#            if lowest_reward == 0:
#                lowest_reward = 1
            reward_range = highest_reward - lowest_reward
            if reward_range == 0:
                reward_range = 1
            for agent in surviving_agents:
#                average_reward_normalized_in_generation = agent.average_reward_normalized / float(highest_reward)
#                agent.temperature = math.exp(average_reward_normalized_in_generation * TAU)
                reward_ratio = (agent.average_reward_normalized - lowest_reward) / float(reward_range)
                agent.temperature = math.exp(reward_ratio * TAU)
            
            sum_temperature = 0
            for agent in surviving_agents:
                sum_temperature += agent.temperature

            if DEBUG_PROGRESS:
                print "Selection probabilities:"
            cumulative_prob = 0.0
            for agent in surviving_agents:
                prob = agent.temperature / float(sum_temperature)
                cumulative_prob += prob
                agent.selection_probability_top = cumulative_prob
                if DEBUG_PROGRESS:
                    print "Agent with avg reward %.2f, prob: %.2f" % (
                        agent.average_reward_normalized, prob)
            
            agents = list(surviving_agents)
            while len(agents) < self.population_size:
#                index = int(random.random() * len(surviving_agents))
#                agent_to_mutate = surviving_agents[index]
                agent_to_mutate = None
                choice = random.random()
                for agent in surviving_agents:
                    if agent.selection_probability_top > choice:
                        agent_to_mutate = agent
                        break
                
                new_agent = mutator.mutate(agent_to_mutate)
                agents.append(new_agent)
            
#            # set up start states            
#            start_states = []
#            start_seeds = []
#            for i in range(self.num_generation_episodes): #@UnusedVariable
#                start_states.append(self.base_agent.environment.generate_start_state())
#                start_seeds.append(random.random())
#            
#            # experiment with agents
#            if USE_MULTIPROCESSING:
#                pool = multiprocessing.Pool(processes=NUM_CORES)
#                params = []
#                for agent in agents:
#                    params.append((agent, start_states, start_seeds, max_steps,
#                                   self.num_generation_episodes, 1, arbitrator_do_episode))
#                updated_champions = pool.map(arbitrator_test_agent, params)
#                agents = updated_champions
#            else:
#                for agent in agents:
#                    self.test_agent(agent, start_states, start_seeds, max_steps,
#                                    self.num_generation_episodes, 1)

            # test all the population on the same start states
            agents = self.test_agents_seedless(agents, max_steps, 
                                               self.num_generation_episodes,
                                               use_common_states = True)

            # update population averages
            for agent in agents:
                for episode in range(self.num_generation_episodes):
                    index = generation * self.num_generation_episodes + episode
                    self.population_reward_log[index] += agent.reward_log[episode]
            
            # find lowest training time
            base_time = None
            for agent in agents:
                if base_time is None:
                    base_time = agent.training_time
                else:
                    if agent.training_time < base_time:
                        base_time = agent.training_time

            # sort agents based on performance
            generation_perf = []
            for agent in agents:
#                computational_cost_multiplier = ETA ** (float(
#                        agent.feature_set.get_encoding_length()) / TiledFeature.DEFAULT_NUM_TILES)
                time_ratio = agent.training_time / base_time 
                computational_cost_multiplier = self.eta ** (time_ratio - 1)
                agent.average_reward_normalized = agent.average_reward * computational_cost_multiplier
                generation_perf.append((agent.average_reward_normalized, agent))
                
            # select generation champion
            generation_sorted = sorted(generation_perf, reverse=True)  
            surviving_agents = []
            
            champion = generation_sorted[0][1]
            champion_cloned = champion.clone()
            # disable learning on the champion
            champion_cloned.pause_learning()
            # add it to the list
            self.champions.append(champion_cloned)
            self.champion_training_rewards.append(champion.average_reward)
            self.champion_training_rewards_normalized.append(champion.average_reward_normalized)
            self.champion_training_times.append(champion.training_time)
            
            # add all episode rewards to the champion reward log
            self.champions_training_reward_log += champion.reward_log
                        
            if DEBUG_PROGRESS:
                print "generation champion: %s" % champion.feature_set
                print "with average reward: %.4f" % champion.average_reward
            if DEBUG_ALG_VALUES:
                print "values:"    
                champion.algorithm.print_w()
            if DEBUG_CHAMPION:
#                print "champion's algorithm values:"
#                champion.algorithm.print_w()
                print "last episode trace:"
                print champion.get_episode_trace()
                
#            # select runner up such that it does not have the same features
#            # as the champion
#            index = 1
#            while index < len(generation_sorted):
#                runnerup = generation_sorted[index][1]
#                if runnerup.get_name() != champion.get_name():
#                    break
#                index += 1
#                
#            surviving_agents.append(champion)
#            surviving_agents.append(runnerup)
#            cross_over = mutator.cross_over(champion, runnerup) 
#            surviving_agents.append(cross_over)

            # select all surviving agents such that they are different from
            # the already selected survivors
            num_survivors = int(self.population_size * SURVIVAL_RATE)
            index = 0
            while (len(surviving_agents) < num_survivors) and (index < len(generation_sorted)):
                agent = generation_sorted[index][1]
                has_new_signature = True 
                for existing_agent in surviving_agents:
                    if agent.get_name() == existing_agent.get_name():
                        has_new_signature = False
                if has_new_signature:
                    surviving_agents.append(agent)
                index += 1
        
#            gc.collect()
    
    
        # all generations finished
        last_champion = surviving_agents[0]
        if DEBUG_PROGRESS:
            print ""
            print "last champion: " + str(last_champion.feature_set)
            print "with average reward: %.4f" % last_champion.average_reward
        if DEBUG_ALG_VALUES:
            print "values:"
            last_champion.algorithm.print_w()
        
        # champion trials
        if self.num_champion_trials != 0:    
            if DEBUG_PROGRESS:
                print ""
                print "running champion trials"
            
#            # generate start states
#            start_states = []
#            start_seeds = []
#            for i in range(self.num_generation_episodes): #@UnusedVariable
#                start_states.append(self.base_agent.environment.generate_start_state())
#                start_seeds.append(random.random())
#
#            # extra trials with champions
#            if USE_MULTIPROCESSING:
#                pool = multiprocessing.Pool(processes=NUM_CORES)
#                params = []
#                for champion in reversed(self.champions):
#                    params.append((champion, start_states, start_seeds, max_steps,
#                                   self.num_generation_episodes, self.num_champion_trials,
#                                   arbitrator_do_episode))
#                updated_champions = pool.map(arbitrator_test_agent, params)
#                updated_champions.reverse()
#                self.champions = updated_champions
#                for champion in self.champions:
#                    champion_reward = champion.average_reward
#                    # repeating the average reward to replace all episode rewards
#                    champion.reward_log = [champion_reward] * self.num_generation_episodes
#                    self.champion_eval_rewards.append(champion_reward)
#                    self.champions_eval_reward_log += champion.reward_log
#            else:
#                for champion in self.champions:
#                    self.test_agent(champion, start_states, start_seeds, max_steps,
#                                    self.num_generation_episodes, self.num_champion_trials)
#                    champion_reward = champion.average_reward
#                    # repeating the average reward to replace all episode rewards
#                    champion.reward_log = [champion_reward] * self.num_generation_episodes
#                    self.champion_eval_rewards.append(champion_reward)
#                    self.champions_eval_reward_log += champion.reward_log
                    
            self.champions = self.test_agents_seedless(self.champions, max_steps,
                                                       self.num_generation_episodes,
                                                       use_common_states=True)
            
            for champion in self.champions:
                champion_reward = champion.average_reward
                # repeating the average reward to replace all episode rewards
                champion.reward_log = [champion_reward] * self.num_generation_episodes
                self.champion_eval_rewards.append(champion_reward)
                self.champions_eval_reward_log += champion.reward_log
            
        # final trial with best champion        
        best_champion = None
        best_champion_reward = 0
        for champion in self.champions:
            champion_reward = champion.average_reward
            if best_champion == None or champion_reward > best_champion_reward:
                best_champion_reward = champion_reward
                best_champion = champion
        
#        best_champion.reset_learning()
#        best_champion.resume_learning()
        
        if DEBUG_PROGRESS:
            print ""
            print "evaluating best champion: " + str(best_champion.feature_set)
            print "with average reward: %.2f" % best_champion.average_reward
        if DEBUG_ALG_VALUES:
            print "values:"
            last_champion.algorithm.print_w()
                            
#        # generate start states
#        start_states = []
#        start_seeds = []
#        for i in range(self.num_best_champion_episodes): #@UnusedVariable
#            start_states.append(self.base_agent.environment.generate_start_state())
#            start_seeds.append(random.random())
#
#        # test best champion
#        self.test_agent(best_champion, start_states, start_seeds, max_steps,
#                        self.num_best_champion_episodes, self.num_best_champion_trials)
#        self.best_champion_reward_log = best_champion.reward_log
            
        if DEBUG_PROGRESS:
            print "generating %d copies of the agent" % self.num_best_champion_trials

        best_champion_copies = []
        for i in range(self.num_best_champion_trials): #@UnusedVariable
            best_champion_copies.append(copy.deepcopy(best_champion))

#        if DEBUG_PROGRESS:
#            print "generating start states"
#
#        start_states_all_trials = {}
#        start_seeds_all_trials = {}
#        for trial in range(self.num_best_champion_trials):
#            start_states_all_trials[trial] = []
#            start_seeds_all_trials[trial] = []            
#            for i in range(self.num_best_champion_episodes): #@UnusedVariable
#                start_states_all_trials[trial].append(best_champion.environment.generate_start_state())
#                start_seeds_all_trials[trial].append(random.random())
#
#        if DEBUG_PROGRESS:
#            print "testing the agents"
#
#        # experiment with agents
#        if USE_MULTIPROCESSING:
#            pool = multiprocessing.Pool(processes=NUM_CORES)
#            params = []
#            for trial in range(self.num_best_champion_trials):
#                params.append((best_champion_copies[trial], start_states_all_trials[trial], start_seeds_all_trials[trial], max_steps,
#                               self.num_best_champion_episodes, 1, arbitrator_do_episode))
#            updated_agents = pool.map(arbitrator_test_agent, params)
#            best_champion_copies = updated_agents
#        else:
#            for trial in range(self.num_trials):
#                self.test_agent(best_champion_copies[trial], start_states_all_trials[trial], start_seeds_all_trials[trial], max_steps,
#                                self.num_episodes, 1)

        best_champion_copies = self.test_agents_seedless(best_champion_copies,
                                max_steps, self.num_best_champion_episodes,
                                use_common_states=False)
        
        self.best_champion_reward_log = [0] * self.num_best_champion_episodes
        for agent in best_champion_copies:
            for episode in range(self.num_best_champion_episodes):
                self.best_champion_reward_log[episode] += agent.reward_log[episode]
        
        for episode in range(self.num_best_champion_episodes):
            self.best_champion_reward_log[episode] /= float(self.num_best_champion_trials)

        if REPORT_RESULTS:
            self.report_results()

    def report_results(self):
        program_name = sys.argv[0]
        program_name_parts = program_name.rsplit(".", 1)
        program_just_name = program_name_parts[0]
        folder_name = 'results/%s-g%de%dt%de%dw%d/' % (program_just_name,
                        self.num_generations,
                        self.num_generation_episodes, self.num_champion_trials,
                        self.eta * 100, MUTATE_NEW_WEIGHTS_MULT * 100)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        report_file = open(folder_name + 'champions.txt', 'w')
        generation = 0
        for champion in self.champions:
            training_reward = self.champion_training_rewards[generation]
            training_reward_normalized = self.champion_training_rewards_normalized[generation]
            trial_reward = self.champion_eval_rewards[generation]
            training_time = self.champion_training_times[generation]
            report_file.write('Champion %d, average training reward: %.2f, normalized: %.2f, average eval reward: %.2f, training time: %.1fs\n' % 
                              (generation, training_reward, training_reward_normalized, trial_reward, training_time))
            report_file.write(champion.get_name())
            report_file.write('\n\n')
            generation += 1
        report_file.close()
        
        report_file = open(folder_name + 'results-champions-training.txt', 'w')
        for episode in range(self.num_generations * self.num_generation_episodes):
            report_file.write('%d %.2f\n' % 
                              (episode, self.champions_training_reward_log[episode]))
        report_file.close()
    
        report_file = open(folder_name + 'results-champions-training-interval.txt', 'w')
        episodes_per_interval = int(self.num_generations * 
                                    self.num_generation_episodes / PLOT_INTERVALS) 
        for interval in range(PLOT_INTERVALS):
            sub_sum = sum(self.champions_training_reward_log[interval * episodes_per_interval:
                                     (interval + 1) * episodes_per_interval])
            report_file.write('%d %.2f\n' % 
                              (interval * episodes_per_interval, 
                               float(sub_sum) / episodes_per_interval)) 
        report_file.close()
        
        report_file = open(folder_name + 'results-champions-eval.txt', 'w')
        for episode in range(self.num_generations * self.num_generation_episodes):
            report_file.write('%d %.2f\n' % 
                              (episode, self.champions_eval_reward_log[episode]))
        report_file.close()
    
        report_file = open(folder_name + 'results-champions-eval-interval.txt', 'w')
        episodes_per_interval = int(self.num_generations * 
                                    self.num_generation_episodes / PLOT_INTERVALS) 
        for interval in range(PLOT_INTERVALS):
            sub_sum = sum(self.champions_eval_reward_log[interval * episodes_per_interval:
                                     (interval + 1) * episodes_per_interval])
            report_file.write('%d %.2f\n' % 
                              (interval * episodes_per_interval, 
                               float(sub_sum) / episodes_per_interval)) 
        report_file.close()
        
        report_file = open(folder_name + 'results-best-champion.txt', 'w')
        for episode in range(self.num_generations * self.num_generation_episodes):
            report_file.write('%d %.2f\n' % 
                              (episode, self.best_champion_reward_log[episode]))
        report_file.close()
    
        report_file = open(folder_name + 'results-best-champion-interval.txt', 'w')
        episodes_per_interval = int(self.num_generations * 
                                    self.num_generation_episodes / PLOT_INTERVALS) 
        for interval in range(PLOT_INTERVALS):
            sub_sum = sum(self.best_champion_reward_log[interval * episodes_per_interval:
                                     (interval + 1) * episodes_per_interval])
            report_file.write('%d %.2f\n' % 
                              (interval * episodes_per_interval, 
                               float(sub_sum) / episodes_per_interval)) 
        report_file.close()

        report_file = open(folder_name + 'results-population.txt', 'w')
        for episode in range(self.num_generations * self.num_generation_episodes):
            report_file.write('%d %.2f\n' % 
                              (episode, float(self.population_reward_log[episode]) /
                              self.population_size))
        report_file.close()
    
        report_file = open(folder_name + 'results-population-interval.txt', 'w')
        episodes_per_interval = int(self.num_generations * 
                                    self.num_generation_episodes / PLOT_INTERVALS) 
        for interval in range(PLOT_INTERVALS):
            sub_sum = sum(self.population_reward_log[interval * episodes_per_interval:
                                     (interval + 1) * episodes_per_interval])
            report_file.write('%d %.2f\n' % 
                              (interval * episodes_per_interval, 
                               float(sub_sum) / 
                               (episodes_per_interval * self.population_size))) 
        report_file.close()
        print "episodes per plot interval: " + str(episodes_per_interval)

        self.plot(folder_name, 'plot-ev',
                  '%d %d' % (self.num_generations, self.num_generation_episodes))

