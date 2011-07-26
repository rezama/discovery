#/usr/bin/env python
'''
Created on Apr 28, 2011

@author: reza
'''
import random
import math
import numpy
import copy
import multiprocessing

USE_NUMPY = False
USE_MULTIPROCESSING = True
NUM_CORES = 8

DEFAULT_GAMMA = 1.0

# debug
DEBUG_PROGRESS = True
DEBUG_EPISODE_TRACE = False
DEBUG_EPISODE_REWARD = False
DEBUG_ALG_VALUES = False
DEBUG_VERBOSE = False
DEBUG_CHAMPION = True
DEBUG_REPORT_ON_EPISODE = 100

# reporting
PLOT_INTERVALS = 100

# default multiplier for initial Q values based on maximum possible reward
DEFAULT_Q_VALUE_MULT = 1.1

OPT_INIT_FEATURE_WEIGHTS = True

MUTATE_OPT_NEW_FEATURE_WEIGHTS = False
MUTATE_COPY_CROSS_OVER_WEIGHTS = True

FORCE_DYNAMIC_PROB = 0.75

class AgentStateBased(object):

    def __init__(self, actions, environment):
        self.state = None
        self.last_state = None
        self.cached_action_valid = False
        self.cached_action = None
        self.cached_action_value = None
        
        self.actions = actions
        self.environment = environment
        self.algorithm = None

        self.episode_steps = 0
        self.gamma_multiplier = 1.0
        self.episode_reward = 0
        self.episode_trace = ""
        self.reward_log = []
        
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
    
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
        if self.cached_action_valid:
            return self.cached_action
        else:
            (a, v) = self.algorithm.select_action()
            self.cached_action = a
            self.cached_action_value = v
            self.cached_action_valid = True
        return a
    
    def transition(self, action):

        #last_state = copy.deepcopy(self.state)
        reward = self.environment.respond(self.state, self.last_state, action)
        self.cached_action = None
        self.cached_action_value = None
        self.cached_action_valid = False
        
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

class AgentFeatureBased(AgentStateBased):

    def __init__(self, actions, environment, feature_set):
        super(AgentFeatureBased, self).__init__(actions, environment)
        self.feature_set = feature_set

    def get_name(self):
        return str(self.feature_set)
    
    def __str__(self):
        return self.get_name()
        
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
    
    def __str__(self):
        names = ""
        is_first = True
        for feature in self.feature_list:
            if is_first:
                is_first = False
            else:
                names += "-"
            names += feature.name
        return names
        
    def encode_state(self, state):
        if USE_NUMPY:
            encoding = numpy.array([])
        else:
            encoding = []
        for feature in self.feature_list:
            next_segment = feature.encode_state(state)
            if USE_NUMPY:
                encoding = numpy.concatenate((encoding, next_segment))
            else:
                encoding += next_segment
        if DEBUG_VERBOSE:
            print str(self)
            print "encoded state %s as: %s" % (state, encoding)
        return encoding

    def len(self):
        return self.encoding_length
    
    def get_num_feature_groups(self):
        return len(self.feature_list)
    
    def get_feature_list_copy(self):
        return list(self.feature_list)
    
class Feature(object):
    
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return NotImplemented
    
    def get_encoding_length(self):
        return NotImplemented
    
    def encode_state(self, state):
        return NotImplemented
    
    def get_underlying_features(self):
        return [self]

class TiledFeature(Feature):

    DEFAULT_NUM_TILES = 10
    MIN_OFFSET = 0.0
    MAX_OFFSET = 1.0
    NEUTRAL_OFFSET = 0.5
    EVEN_OFFSETS = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    
    def __init__(self, name, min_value, max_value, num_tiles = DEFAULT_NUM_TILES, 
                 offset = NEUTRAL_OFFSET):
        super(TiledFeature, self).__init__(name)
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

class FeatureDist(TiledFeature):
    
    def __init__(self, name, point1, point2, 
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES,
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point1_name = point1.name
        self.point2_name = point2.name
        min_dist = 0
        point_range = point1.point_range
        max_dist = GeometryUtil.compute_dist(point_range[0][0], point_range[0][1], 
                point_range[1][0], point_range[1][1]) 
        super(FeatureDist, self).__init__(name, min_dist, max_dist,
                num_tiles, offset)
    
    def encode_state(self, state):
        if USE_NUMPY:
            feature_encoding = numpy.zeros(self.num_tiles)
        else:
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
        
class FeatureAngle(TiledFeature):
    
    def __init__(self, name, point1, point2, point3,
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES,
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point1_name = point1.name
        self.point2_name = point2.name
        self.point3_name = point3.name
        min_angle = 0
        max_angle = math.pi
        super(FeatureAngle, self).__init__(name, min_angle, max_angle,
                num_tiles, offset)
    
    def encode_state(self, state):
        if USE_NUMPY:
            feature_encoding = numpy.zeros(self.num_tiles)
        else:
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

class FeaturePoint1D(TiledFeature):
    
    def __init__(self, name, point, use_x, 
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES, 
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point_name = point.name
        self.use_x = use_x
        if use_x:
            min_value = point.point_range[0][0]
            max_value = point.point_range[1][0]
        else:
            min_value = point.point_range[0][1]
            max_value = point.point_range[1][1]
        super(FeaturePoint1D, self).__init__(name, min_value, max_value, 
                                                  num_tiles, offset)
            
    def encode_state(self, state):
        if USE_NUMPY:
            feature_encoding = numpy.zeros(self.num_tiles)
        else:
            feature_encoding = [0] * self.num_tiles
        
        point = state.index[self.point_name]
            
#        if state.is_terminal():
#            return feature_encoding

        if self.use_x:
            value = point.x
        else:
            value = point.y 
        feature_index = self.get_tile_index(value)
        feature_encoding[feature_index] = 1
        return feature_encoding 
        
class FeaturePoint2D(Feature):
    
    def __init__(self, name, point,
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES * TiledFeature.DEFAULT_NUM_TILES, 
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point_name = point.name
        self.num_tiles = num_tiles
        self.num_tiles_1d = int(math.sqrt(num_tiles))
        self.offset = offset
        super(FeaturePoint2D, self).__init__(name)
            
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

    def encode_state(self, state):
        if USE_NUMPY:
            feature_encoding = numpy.zeros(self.num_tiles)
        else:
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

class FeatureFlag(TiledFeature):
    
    def __init__(self, name, flag):
        num_tiles = 2
        self.flag_name = flag.name
        super(FeatureFlag, self).__init__(name, 0, 1,
                num_tiles, TiledFeature.NEUTRAL_OFFSET)
    
    def encode_state(self, state):
        if USE_NUMPY:
            feature_encoding = numpy.zeros(self.num_tiles)
        else:
            feature_encoding = [0] * self.num_tiles
            
#        if state.is_terminal():
#            return feature_encoding
        
        flag = state.index[self.flag_name]
        feature_index = 1 if flag.truth else 0
        feature_encoding[feature_index] = 1
        return feature_encoding 

class FeatureDistX(TiledFeature):
    
    def __init__(self, name, point1, point2, 
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES,
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point1_name = point1.name
        self.point2_name = point2.name
        point_range = point1.point_range
        max_dist = abs(point_range[1][0] - point_range[0][0])
        min_dist = -max_dist 
        super(FeatureDistX, self).__init__(name, min_dist, max_dist,
                num_tiles, offset)
    
    def encode_state(self, state):
        if USE_NUMPY:
            feature_encoding = numpy.zeros(self.num_tiles)
        else:
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
    
    def __init__(self, name, point1, point2, 
                 num_tiles = TiledFeature.DEFAULT_NUM_TILES,
                 offset = TiledFeature.NEUTRAL_OFFSET):
        self.point1_name = point1.name
        self.point2_name = point2.name
        point_range = point1.point_range
        max_dist = abs(point_range[1][1] - point_range[0][1])
        min_dist = -max_dist 
        super(FeatureDistY, self).__init__(name, min_dist, max_dist,
                num_tiles, offset)
    
    def encode_state(self, state):
        if USE_NUMPY:
            feature_encoding = numpy.zeros(self.num_tiles)
        else:
            feature_encoding = [0] * self.num_tiles
            
#        if state.is_terminal():
#            return feature_encoding
        point1 = state.index[self.point1_name]
        point2 = state.index[self.point2_name]
        dist = point2.y - point1.y
        feature_index = self.get_tile_index(dist)
        feature_encoding[feature_index] = 1
        return feature_encoding

class FeatureMulti(Feature):
    
    def __init__(self, base_feature_list):
        self.all_features = []
        for base_feature in base_feature_list:
            self.all_features += base_feature.get_underlying_features()
        
        num_tiles = 1
        name = "multi("
        is_first = True
        for feature in self.all_features:
            num_tiles *= feature.get_num_tiles()
            if is_first:
                is_first = False
            else:
                name += ", "
            name += feature.name
        name += ")"
        self.num_tiles = num_tiles
        super(FeatureMulti, self).__init__(name)
    
    def get_underlying_features(self):
        return self.all_features
    
    def get_num_tiles(self):
        return self.num_tiles

    def get_encoding_length(self):
        return self.num_tiles

    def encode_state(self, state):
        if USE_NUMPY:
            feature_encoding = numpy.zeros(self.num_tiles)
        else:
            feature_encoding = [0] * self.num_tiles
        
        feature_index = 0
        multiplier = 1
        for feature in self.all_features:
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
        
class FeaturizerAngle(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerAngle, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        aux_point1 = StateVarPoint2D.get_random_var(self.state_vars, [main_point])
        aux_point2 = StateVarPoint2D.get_random_var(self.state_vars, [main_point, aux_point1])
        feature_name = "angle(%s-%s-%s)" % (main_point.name, aux_point1.name,
                                            aux_point2.name)
        num_tiles = TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeatureAngle(feature_name, main_point, aux_point1, 
                                 aux_point2, num_tiles, offset)
        
class FeaturizerDist(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerDist, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        aux_point1 = StateVarPoint2D.get_random_var(self.state_vars, [main_point])
        feature_name = "dist(%s-%s)" % (main_point.name, aux_point1.name)
        num_tiles = TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeatureDist(feature_name, main_point, aux_point1, 
                                 num_tiles, offset)
        
class FeaturizerFlag(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerFlag, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        flag_var = StateVarFlag.get_random_var(self.state_vars, [], is_dynamic=True)
        feature_name = "flag(%s)" % (flag_var.name)
        return FeatureFlag(feature_name, flag_var)
        
class FeaturizerDistX(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerDistX, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        aux_point1 = StateVarPoint2D.get_random_var(self.state_vars, [main_point])
        feature_name = "dist-X(%s-%s)" % (main_point.name, aux_point1.name)
        num_tiles = TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeatureDistX(feature_name, main_point, aux_point1, 
                                 num_tiles, offset)
        
class FeaturizerDistY(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerDistY, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        aux_point1 = StateVarPoint2D.get_random_var(self.state_vars, [main_point])
        feature_name = "dist-Y(%s-%s)" % (main_point.name, aux_point1.name)
        num_tiles = TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeatureDistY(feature_name, main_point, aux_point1, 
                                 num_tiles, offset)
        
class FeaturizerPoint2D(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerPoint2D, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        main_point = StateVarPoint2D.get_random_var(self.state_vars, [], is_dynamic=True)
        feature_name = "point2d(%s)" % (main_point.name)
        num_tiles = TiledFeature.DEFAULT_NUM_TILES * TiledFeature.DEFAULT_NUM_TILES
        offset = random.random()
        return FeaturePoint2D(feature_name, main_point, num_tiles, offset)
    
class FeaturizerMulti(Featurizer):
    
    def __init__(self, state_vars):
        super(FeaturizerMulti, self).__init__(state_vars)
        
    def generate_feature(self, feature_list):
        new_feature = None
        if len(feature_list) != 0:
            rand1_index = int(random.random() * len(feature_list))
            rand2_index = int(random.random() * len(feature_list))
            feature1 = feature_list[rand1_index]
            feature2 = feature_list[rand2_index]
            if feature1.name != feature2.name:
                feature1_copy = copy.deepcopy(feature1)
                feature2_copy = copy.deepcopy(feature2)
                new_feature = FeatureMulti([feature1_copy, feature2_copy])
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
        new_feature.offset = random.random()
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
        if cumulative_prob != 1.0:
            print "Error: Sum of featurizer selection probabilities is %.2f" % cumulative_prob
        return selected_featurizer
    
    def mutate(self, agent):
        feature_list = agent.feature_set.get_feature_list_copy()
        
        new_feature = None
        while new_feature is None:
            featurizer = self.select_featurizer()
            new_feature = featurizer.generate_feature(feature_list)
            
        feature_list.append(new_feature)
        new_feature_set = FeatureSet(feature_list)
        
        agent_class = agent.__class__
        new_agent = agent_class(new_feature_set)
        
        if MUTATE_OPT_NEW_FEATURE_WEIGHTS or len(feature_list) == 1:
            new_segment_weights = ((agent.environment.get_max_episode_reward() * 
                              DEFAULT_Q_VALUE_MULT) / 
                              new_feature_set.get_num_feature_groups())
        else:
            new_segment_weights = 0
        
        for action in agent.all_actions():
            new_agent.algorithm.w[action] = agent.algorithm.w[action] + \
                    [new_segment_weights] * new_feature.get_encoding_length()
        
        return new_agent
        
    def cross_over(self, agent1, agent2):
        new_feature_list = agent1.feature_set.get_feature_list_copy()
        feature_list_from_agent2 = agent2.feature_set.get_feature_list_copy()
        last_feature_from_agent2 = feature_list_from_agent2[-1]
        new_feature_list.append(last_feature_from_agent2)

        new_feature_set = FeatureSet(new_feature_list)
        
        agent_class = agent1.__class__
        new_agent = agent_class(new_feature_set)
        
        for action in agent1.actions.all_actions():
            len_last_feature_from_agent2 = \
                last_feature_from_agent2.get_encoding_length()
            if MUTATE_COPY_CROSS_OVER_WEIGHTS:
                feature_w_from_agent_2 = \
                    agent2.algorithm.w[action][-len_last_feature_from_agent2:]
                new_agent.algorithm.w[action] = \
                    agent1.algorithm.w[action] + feature_w_from_agent_2
            else:
                new_agent.algorithm.w[action] = \
                    agent1.algorithm.w[action] + [0] * len_last_feature_from_agent2
        
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

    def __init__(self, agent):
        self.agent = agent
        self.environment = agent.environment
        if self.environment is not None:
            self.gamma = agent.environment.gamma
        else:
            self.gamma = DEFAULT_GAMMA
    
    def begin_episode(self, state):
        pass
    
    def select_action(self):
        return NotImplemented
    
    def transition(self, state, action, reward, state_p, action_p):
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

    def __init__(self, agent, alpha, epsilon, lamda):
        super(Sarsa, self).__init__(agent)
        self.epsilon = epsilon
        self.lamda = lamda
        self.alpha = alpha
               
class SarsaLambda(Sarsa):

    def __init__(self, agent,
                 alpha = Sarsa.DEFAULT_ALPHA, 
                 epsilon = Sarsa.DEFAULT_EPSILON,
                 lamda = Sarsa.DEFAULT_LAMBDA):
        super(SarsaLambda, self).__init__(agent, alpha, epsilon, lamda)
        self.Q = {}
        self.e = {}
        self.default_q = self.environment.get_max_episode_reward() * \
                DEFAULT_Q_VALUE_MULT
        
        # set values for terminal state
#        terminal_state_repr = str(self.environment.generate_terminal_state())
#        for action in self.agent.all_actions():
#            self.Q[(terminal_state_repr, action)] = 0
    
    def begin_episode(self, state):
        self.e = {}
    
    def select_action(self):
        state = str(self.agent.state)
        
        if random.random() < self.epsilon:
            action = self.agent.actions.random_action()
            value = self.Q.get((state, action), self.default_q) 
        else:
            action_values = []
            for action in self.agent.all_actions():
                # insert a random number to break the ties
                action_values.append(((self.Q.get((state, action), self.default_q), 
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

    def __init__(self, agent,
                 alpha = Sarsa.DEFAULT_ALPHA, 
                 epsilon = Sarsa.DEFAULT_EPSILON,
                 lamda = Sarsa.DEFAULT_LAMBDA):
        super(SarsaLambdaFeaturized, self).__init__(agent, alpha, epsilon, lamda)
        self.w = {}
        self.e = {}

        self.feature_set = agent.feature_set

        if OPT_INIT_FEATURE_WEIGHTS and \
                (self.feature_set.get_num_feature_groups() > 0):
            self.default_w = ((self.environment.get_max_episode_reward() * 
                              DEFAULT_Q_VALUE_MULT) / 
                              self.feature_set.get_num_feature_groups())
    
        else:
            self.default_w = 0
            
        for action in self.agent.all_actions():
            if USE_NUMPY:
                self.w[action] = numpy.ones(self.feature_set.len()) * \
                        self.default_w
            else:
                self.w[action] = [self.default_w] * \
                        self.feature_set.len()

    def begin_episode(self, state):
        if USE_NUMPY:
            for action in self.agent.all_actions():
                self.e[action] = numpy.zeros(self.feature_set.len())
        else:
            for action in self.agent.all_actions():
                self.e[action] = [0] * self.feature_set.len()
                
    def compute_Q(self, features_present, action):
        sum = 0
        if USE_NUMPY:
            sum = numpy.dot(self.w[action], features_present)
        else:
            for i in range(self.feature_set.len()):
                sum += self.w[action][i] * features_present[i]
        return sum
    
    def select_action(self):
        features = self.feature_set.encode_state(self.agent.state)
        
        if random.random() < self.epsilon:
            action = self.agent.actions.random_action()
            value = self.compute_Q(features, action) 
        else:
            action_values = []
            for action in self.agent.all_actions():
                # insert a random number to break the ties
                action_values.append(((self.compute_Q(features, action), 
                                       random.random()), action))
                
            action_values_sorted = sorted(action_values, reverse=True)
            
            action = action_values_sorted[0][1]
            value = action_values_sorted[0][0][0]
        
        return (action, value)

    def update_weights(self, delta):
        alpha = self.alpha / self.feature_set.get_num_feature_groups()
        for action in self.agent.all_actions():
            if USE_NUMPY:
                self.w[action] += (self.e[action] * alpha * delta)
            else:
#                self.w[action] += [alpha * delta * self.e[action][i]
#                                         for i in len(self.w[action])] 
                for i in range(len(self.w[action])):
                    self.w[action][i] += (alpha * 
                                        delta * self.e[action][i])
    
    def transition(self, state, action, reward, state_p, action_p):
        s = str(state)
        a = action
        sp = str(state_p)
        ap = action_p

        Fa = self.feature_set.encode_state(state)
        
        # update e
        for action in self.agent.all_actions():
        #                self.e[action] = [e * self.gamma * self.lamda 
        #                                  for e in self.e[action]]
            for i in range(len(self.e[action])):
                self.e[action][i] *= (self.gamma * self.lamda)
                
        for i in range(len(Fa)):
            if Fa[i] == 1:
                # replacing traces
                self.e[a][i] = 1
                # set the trace for the other actions to 0
                for action in self.agent.all_actions():
                    if action != a:
                        self.e[action][i] = 0

        sigma_w_Fa = 0
        if USE_NUMPY:
            sigma_w_Fa = numpy.dot(Fa, self.w[a])
        else:
            sigma_w_Fa = self.compute_Q(Fa, a)
#            for i in range(len(Fa)):
#                if Fa[i] == 1:
#                    sigma_w_Fa += self.w[a][i]
        
        if state_p.is_final():
            delta = reward - sigma_w_Fa
        else:
            # select next action
#            (ap, Q_a) = self.select_action()
            Q_a = self.agent.cached_action_value
            delta = reward + self.gamma * Q_a - sigma_w_Fa

        self.update_weights(delta)
        
#    def do_episode(self, start_state):
#        self.agent.episode_trace = ""
#        if USE_NUMPY:
#            for action in self.agent.all_actions():
#                self.e[action] = numpy.zeros(self.feature_set.len())
#        else:
#            for action in self.agent.all_actions():
#                self.e[action] = [0] * self.feature_set.len()
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
#                for i in range(len(self.e[action])):
#                    self.e[action][i] *= (self.gamma * self.lamda)
#                    
#            for i in range(len(Fa)):
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
##                for i in range(len(Fa)):
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

def arbitrator_do_episode((agent, start_state, max_steps)):
    agent.begin_episode(start_state)

    steps = 0
    while not agent.state.is_final():
        a = agent.select_action()
        r = agent.transition(a)
        steps += 1
        if (max_steps != 0) and (steps >= max_steps):
            break
    
    episode_reward = agent.episode_reward
    return (episode_reward, steps)
    
def arbitrator_test_agent((agent, start_states, max_steps, generation_episodes,
                          do_episode_func)):
#    if DEBUG_PROGRESS:
#        print "* testing agent: " + str(agent.feature_set)
    agent.reward_log = [0] * generation_episodes
    for episode in range(generation_episodes):
        # get start state
        start_state = copy.deepcopy(start_states[episode])
        # seed random number generator
#                random.seed(start_seeds[episode])
        random.seed(start_states[episode])
        (episode_reward, steps) = do_episode_func((agent, start_state, max_steps))
        agent.reward_log[episode] += episode_reward

        if DEBUG_EPISODE_REWARD:
            print "episode %i: reward %.2f, steps:%d" % (episode,
                                            episode_reward, steps)
        if DEBUG_ALG_VALUES:
            print "values:"
            agent.algorithm.print_values()
            
    if DEBUG_PROGRESS:
        average_reward = float(sum(agent.reward_log)) / generation_episodes
        print "tested agent: " + str(agent.feature_set)
        print "average reward: %.2f" % average_reward            

    return agent
        
class Arbitrator(object):

    def __init__(self):
        pass
    
    def execute(self, max_steps = 0):
        return NotImplemented
    
    def do_episode(self, agent, start_state, max_steps):
        return arbitrator_do_episode((agent, start_state, max_steps))
        
    def report_results(self):
        return NotImplemented

class ArbitratorStandard(Arbitrator):
    
    def __init__(self, agent, num_trials, num_episodes):
        super(ArbitratorStandard, self).__init__()
        self.agent = agent
        self.num_trials = num_trials
        self.num_episodes = num_episodes

    def execute(self, max_steps = 0):
        reward_log = [0] * self.num_episodes
        for trial in range(self.num_trials):
            trial_reward = 0
            for episode in range(self.num_episodes):
                if DEBUG_PROGRESS and (episode % DEBUG_REPORT_ON_EPISODE == 0):
                    print "trial %i episode %i" % (trial, episode)
#                    print agent.w
                start_state = self.agent.environment.generate_start_state()
                (episode_reward, steps) = self.do_episode(self.agent, start_state, max_steps)
                trial_reward += episode_reward
                reward_log[episode] += episode_reward
                if DEBUG_EPISODE_REWARD:
                    print "episode %i: reward %.2f, steps:%d" % (episode, 
                                                    episode_reward, steps)
#                    print "trace:"
#                    print self.agent.get_episode_trace()
                if DEBUG_ALG_VALUES:
                    print "values:"
                    self.agent.algorithm.print_values()
        
        if DEBUG_PROGRESS:
            print "average reward: %.2f" % (float(trial_reward) / self.num_episodes) 
        
        self.report_results(reward_log)

    def report_results(self, reward_log):
        report_file = open('results/results-standard.txt', 'w')
        for episode in range(self.num_episodes):
            report_file.write('%d %.2f\n' % 
                              (episode, float(reward_log[episode]) / self.num_trials)) 
        report_file.close()
    
        report_file = open('results/results-standard-interval.txt', 'w')
        episodes_per_interval = int (self.num_episodes / PLOT_INTERVALS) 
        for interval in range(PLOT_INTERVALS):
            sub_sum = sum(reward_log[interval * episodes_per_interval:
                                     (interval + 1) * episodes_per_interval])
            report_file.write('%d %.2f\n' % 
                              (interval * episodes_per_interval, 
                               float(sub_sum) / (self.num_trials * episodes_per_interval))) 
        report_file.close()
    

class ArbitratorEvolutionary(Arbitrator):
    
    def __init__(self, base_agent, featurizers_map, num_generations, 
                 population_size, generation_episodes):
        super(ArbitratorEvolutionary, self).__init__()
        self.base_agent = base_agent
        self.featurizers_map = featurizers_map
        self.num_generations = num_generations
        self.population_size = population_size
        self.generation_episodes = generation_episodes

    def test_agent(self, agent, start_states, max_steps):
        return arbitrator_test_agent((agent, start_states, max_steps,
                                     self.generation_episodes, arbitrator_do_episode))
        
    def execute(self, max_steps = 0):
        
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
            
        champion_reward_log = []
        average_reward_log = [0] * (self.num_generations * self.generation_episodes)
        
        for generation in range(self.num_generations):
            start_states = []
            start_seeds = []
            for i in range(self.generation_episodes):
                start_states.append(self.base_agent.environment.generate_start_state())
                start_seeds.append(random.random())
            
            if DEBUG_PROGRESS:
                print "generation %i" % (generation)
    
            # mutate agents
            agents = list(surviving_agents)
            while len(agents) < self.population_size:
                index = int(random.random() * len(surviving_agents))
                agent_to_mutate = surviving_agents[index]
                
                new_agent = mutator.mutate(agent_to_mutate)
                agents.append(new_agent)
            
            generation_performance = []
            
            # experiment with agents
            if USE_MULTIPROCESSING:
                pool = multiprocessing.Pool(processes=NUM_CORES)
                params = []
                for agent in agents:
                    params.append((agent, start_states, max_steps,
                                   self.generation_episodes, arbitrator_do_episode))
                updated_agents = pool.map(arbitrator_test_agent, params)
                agents = updated_agents
            else:
                for agent in agents:
                    self.test_agent(agent, start_states, max_steps)
            
            # update average rewards
            for agent in agents:
                for episode in range(self.generation_episodes):
                    average_reward_log[generation * self.generation_episodes + episode] += \
                    agent.reward_log[episode]
            
                average_reward = float(sum(agent.reward_log)) / self.generation_episodes
                generation_performance.append((average_reward, agent))
            
            # select generation champion
            generation_sorted = sorted(generation_performance, reverse=True)  
            surviving_agents = []
            champion1 = generation_sorted[0][1]
            # select runner up such that it does not have the same features
            # as the champion
            index = 1
            while True:
                champion2 = generation_sorted[index][1]
                if champion2.get_name() != champion1.get_name():
                    break
                index += 1
                
            surviving_agents.append(champion1)
            surviving_agents.append(champion2)
            cross_over = mutator.cross_over(champion1, champion2) 
            surviving_agents.append(cross_over)
            
            champion_reward_log += champion1.reward_log
            
            if DEBUG_PROGRESS:
                print "generation champion: " + str(champion1.feature_set)
                print "with average reward: " + str(generation_sorted[0][0])
            if DEBUG_ALG_VALUES:
                print "values:"    
                champion1.algorithm.print_w()
            if DEBUG_CHAMPION:
#                print "champion's algorithm values:"
#                champion1.algorithm.print_w()
                print "last episode trace:"
                print champion1.get_episode_trace()
    
        winning_agent = surviving_agents[0]
        if DEBUG_PROGRESS:
            print "final champion: " + str(winning_agent.feature_set)
        if DEBUG_ALG_VALUES:
            print "values:"
            winning_agent.algorithm.print_w()
        
        self.report_results(champion_reward_log, average_reward_log)

    def report_results(self, champion_reward_log, average_reward_log):
        report_file = open('results/results-champion.txt', 'w')
        for episode in range(self.num_generations * self.generation_episodes):
            report_file.write('%d %.2f\n' % 
                              (episode, champion_reward_log[episode]))
        report_file.close()
    
        report_file = open('results/results-champion-interval.txt', 'w')
        episodes_per_interval = int(self.num_generations * 
                                    self.generation_episodes / PLOT_INTERVALS) 
        print "episodes per plot interval: " + str(episodes_per_interval)
        for interval in range(PLOT_INTERVALS):
            sub_sum = sum(champion_reward_log[interval * episodes_per_interval:
                                     (interval + 1) * episodes_per_interval])
            report_file.write('%d %.2f\n' % 
                              (interval * episodes_per_interval, 
                               float(sub_sum) / episodes_per_interval)) 
        report_file.close()
        
        report_file = open('results/results-population.txt', 'w')
        for episode in range(self.num_generations * self.generation_episodes):
            report_file.write('%d %.2f\n' % 
                              (episode, float(average_reward_log[episode]) /
                              self.population_size))
        report_file.close()
    
        report_file = open('results/results-population-interval.txt', 'w')
        episodes_per_interval = int(self.num_generations * 
                                    self.generation_episodes / PLOT_INTERVALS) 
        for interval in range(PLOT_INTERVALS):
            sub_sum = sum(average_reward_log[interval * episodes_per_interval:
                                     (interval + 1) * episodes_per_interval])
            report_file.write('%d %.2f\n' % 
                              (interval * episodes_per_interval, 
                               float(sub_sum) / 
                               (episodes_per_interval * self.population_size))) 
        report_file.close()

