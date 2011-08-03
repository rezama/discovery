#/usr/bin/env python

'''
Created on Apr 28, 2011

@author: reza
'''
import random

import rl
import sys

# start state configuration
RANDOMIZE_START_STATE = True

# trials
NUM_TRIALS = 1
NUM_EPISODES = 1000

# evolutionary settings
NUM_GENERATIONS = 15
POPULATION_SIZE = 30
GENERATION_EPISODES = 200
CHAMPION_TRIALS = 20

class KnightJoustStateBasedAgent(rl.AgentStateBased):
    
    def __init__(self):
        actions = KnightJoustActions()
        environment = KnightJoustEnvironment()
        algorithm = rl.SarsaLambda(environment, actions)
        super(KnightJoustStateBasedAgent, self).__init__(actions, environment, algorithm)
#        self.set_algorithm()

class KnightJoustFeatureBasedAgent(rl.AgentFeatureBased):
    
    def __init__(self, feature_set):
        actions = KnightJoustActions()
        environment = KnightJoustEnvironment()
        algorithm = rl.SarsaLambdaFeaturized(actions, environment, feature_set)
        super(KnightJoustFeatureBasedAgent, self).__init__(actions, environment,
                                                        feature_set, algorithm)
#        self.set_algorithm()
    
class KnightJoustEnvironment(rl.Environment):
    
    GRID_SIZE = 25
    
    REWARD_FORWARD = 20
    REWARD_FINAL = 20
    
    def __init__(self):
        super(KnightJoustEnvironment, self).__init__(KnightJoustState)

    def get_max_episode_reward(self):
        return (self.GRID_SIZE - 1) * self.REWARD_FORWARD + self.REWARD_FINAL

    @classmethod
    def get_environment_vars(cls):
        point_range = ((0, 0),
                       (KnightJoustEnvironment.GRID_SIZE - 1, 
                        KnightJoustEnvironment.GRID_SIZE - 1))
                
        lower_left_corner_x = 0
        lower_left_corner_y = 0
        lower_right_corner_x = KnightJoustEnvironment.GRID_SIZE - 1
        lower_right_corner_y = 0
        upper_left_corner_x = 0
        upper_left_corner_y = KnightJoustEnvironment.GRID_SIZE - 1
        upper_right_corner_x = KnightJoustEnvironment.GRID_SIZE - 1
        upper_right_corner_y = KnightJoustEnvironment.GRID_SIZE - 1
        
        lower_left_corner_state_var = rl.StateVarPoint2D("lowerleft",
                lower_left_corner_x, lower_left_corner_y, 
                point_range, is_dynamic=False, is_continuous=True)
        lower_right_corner_state_var = rl.StateVarPoint2D("lowerright",
                lower_right_corner_x, lower_right_corner_y, 
                point_range, is_dynamic=False, is_continuous=True)
        upper_left_corner_state_var = rl.StateVarPoint2D("upperleft",
                upper_left_corner_x, upper_left_corner_y, 
                point_range, is_dynamic=False, is_continuous=True)
        upper_right_corner_state_var = rl.StateVarPoint2D("upperright",
                upper_right_corner_x, upper_right_corner_y, 
                point_range, is_dynamic=False, is_continuous=True)
        
        return [lower_left_corner_state_var,
                lower_right_corner_state_var,
                upper_left_corner_state_var,
                upper_right_corner_state_var]
    
    def respond(self, state, last_state, action):
        reward = 0
#        if state.is_final():
#            state.make_terminal()
#        else:
        if not state.is_final():
            player = state.index["player"]
            opponent = state.index["opponent"]
            # save to old state
            player_p = last_state.index["player"]
            opponent_p = last_state.index["opponent"]
            player_p.x = player.x
            player_p.y = player.y
            opponent_p.x = opponent.x
            opponent_p.y = opponent.y
            # respond
            if action == KnightJoustActions.FORWARD:
                player.y += 1
                reward = self.REWARD_FORWARD 
            elif action == KnightJoustActions.JUMP_WEST:
                if player.x >= 2:
                    player.y += 1
                    player.x -= 2
            elif action == KnightJoustActions.JUMP_EAST:
                if self.GRID_SIZE - 1 - player.x >= 2:
                    player.y += 1
                    player.x += 2
            if player.y == self.GRID_SIZE - 1:
                reward += self.REWARD_FINAL
            
            # opponent's action
            if opponent.x > player.x:
                if random.random() < 0.9:
                    opponent.x -= 1
            elif opponent.x < player.x:
                if random.random() < 0.9:
                    opponent.x += 1
            if opponent.y > player.y:
                opponent.y -= 1
            elif opponent.y < player.y:
                if random.random() < 0.8:
                    opponent.y += 1
            
        return reward

class KnightJoustActions(rl.Actions):

    FORWARD = "f"
    JUMP_WEST = "w"
    JUMP_EAST = "e"

    def __init__(self):
        actions = [self.FORWARD, self.JUMP_WEST, self.JUMP_EAST]
        super(KnightJoustActions, self).__init__(actions)
    
#class KnightJoustState(rl.State):
#    
#    def __init__(self, player_row, player_col, opponent_row, opponent_col):
#        super(KnightJoustState, self).__init__()
#        self.player_row = player_row
#        self.player_col = player_col
#        self.opponent_row = opponent_row
#        self.opponent_col = opponent_col
#        
#    def __str__(self):
#        return "%2d-%2d-%2d-%2d" % (self.player_row, self.player_col, 
#                                    self.opponent_row, self.opponent_col)
#        
#    @classmethod
#    def generate_start_state(cls):
#        player_row = 0
##        player_col = self.grid_size / 2
#        player_col = int(random.random() * KnightJoustEnvironment.GRID_SIZE)
#        opponent_row = KnightJoustEnvironment.GRID_SIZE - 1
##        opponent_col = self.grid_size / 2
#        opponent_col = int(random.random() * KnightJoustEnvironment.GRID_SIZE)
#        return KnightJoustState(player_row, player_col,
#                                opponent_row, opponent_col)
#
#    @classmethod
#    def generate_terminal_state(cls):
#        return KnightJoustState(-1, -1, -1, -1)
#
#    def is_final(self):
#        if self.player_col == self.opponent_col and \
#                self.player_row == self.opponent_row:
#            return True
#        elif self.player_row == KnightJoustEnvironment.GRID_SIZE - 1:
#            return True
#        else:
#            return False
#        
#    def is_terminal(self):
#        return (self.player_row == -1)
#
#    def make_terminal(self):
#        self.player_row = -1
#        self.player_col = -1
#        self.opponent_row = -1
#        self.opponent_col = -1

class KnightJoustState(rl.ModularState):
    
    def __init__(self, state_variables):
        environment_vars = KnightJoustEnvironment.get_environment_vars()
        super(KnightJoustState, self).__init__(state_variables + environment_vars)
        
    @classmethod
    def generate_start_state(cls):
        point_range = ((0, 0),
                       (KnightJoustEnvironment.GRID_SIZE - 1, 
                        KnightJoustEnvironment.GRID_SIZE - 1))
        
        player_x = int(KnightJoustEnvironment.GRID_SIZE / 2)
        player_y = 0
        opponent_x = int(KnightJoustEnvironment.GRID_SIZE / 2)
        opponent_y = KnightJoustEnvironment.GRID_SIZE - 1
        if RANDOMIZE_START_STATE:
            player_x = int(random.random() * KnightJoustEnvironment.GRID_SIZE)
            opponent_x = int(random.random() * KnightJoustEnvironment.GRID_SIZE)

        player_state_var = rl.StateVarPoint2D("player", player_x, player_y, 
                point_range, is_dynamic=True, is_continuous=True)
        opponent_state_var = rl.StateVarPoint2D("opponent", opponent_x, opponent_y,
                point_range, is_dynamic=True, is_continuous=True)
        
        state = KnightJoustState([player_state_var, opponent_state_var])
        
        return state

    def is_final(self):
        player_var = self.index["player"]
        opponent_var = self.index["opponent"]
        if player_var.x == opponent_var.x and \
                player_var.y == opponent_var.y:
            return True
        elif player_var.y == KnightJoustEnvironment.GRID_SIZE - 1:
            return True
        else:
            return False

#class KnightJoustFeature(rl.TiledFeature):
#    
#    def __init__(self, name, min, max,
#                 num_tiles = rl.TiledFeature.DEFAULT_NUM_TILES, 
#                 offset = rl.TiledFeature.NEUTRAL_OFFSET):
#        super(KnightJoustFeature, self).__init__(name, min, max,
#                                                 num_tiles, offset)
#
#class KnightJoustFeatureDist(KnightJoustFeature):
#    
#    def __init__(self, offset = rl.TiledFeature.NEUTRAL_OFFSET):
#        max_dist = rl.GeometryUtil.compute_dist(0, 0, 
#                KnightJoustEnvironment.GRID_SIZE - 1, 
#                KnightJoustEnvironment.GRID_SIZE - 1)
#        super(KnightJoustFeatureDist, self).__init__("dist-po", 
#                0, max_dist, KnightJoustFeature.DEFAULT_NUM_TILES, offset)
#    
#    def encode_state(self, state):
#        if rl.USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
#            
##        if state.is_terminal():
##            return feature_encoding
#        player = state.index['player']
#        opponent = state.index['opponent']
#        dist = rl.GeometryUtil.compute_dist(opponent.x, opponent.y,
#                                         player.x, player.y)
#        feature_index = self.get_tile_index(dist)
#        feature_encoding[feature_index] = 1
#        return feature_encoding 
#        
#class KnightJoustFeatureAngleWest(KnightJoustFeature):
#    
#    def __init__(self, offset = 0):
#        super(KnightJoustFeatureAngleWest, self).__init__("angle-west", 
#                0, math.pi, KnightJoustFeature.DEFAULT_NUM_TILES, offset)
#    
#    def encode_state(self, state):
#        if rl.USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
#            
##        if state.is_terminal():
##            return feature_encoding
#        player = state.index['player']
#        opponent = state.index['opponent']
#        angle = rl.GeometryUtil.compute_angle(player.x, player.y,
#                opponent.x, opponent.y,
#                KnightJoustEnvironment.GRID_SIZE - 1, KnightJoustEnvironment.GRID_SIZE - 1)
#        feature_index = self.get_tile_index(angle)
#        feature_encoding[feature_index] = 1
#        return feature_encoding
#        
#class KnightJoustFeatureAngleEast(KnightJoustFeature):
#    
#    def __init__(self, offset = 0):
#        super(KnightJoustFeatureAngleEast, self).__init__("angle-east", 
#                0, math.pi, KnightJoustFeature.DEFAULT_NUM_TILES, offset)
#    
#    def encode_state(self, state):
#        if rl.USE_NUMPY:
#            feature_encoding = numpy.zeros(self.num_tiles)
#        else:
#            feature_encoding = [0] * self.num_tiles
#            
##        if state.is_terminal():
##            return feature_encoding
#        player = state.index['player']
#        opponent = state.index['opponent']
#        angle = rl.GeometryUtil.compute_angle(player.x, player.y,
#                opponent.x, opponent.y,
#                0, KnightJoustEnvironment.GRID_SIZE - 1)
#        feature_index = self.get_tile_index(angle)
#        feature_encoding[feature_index] = 1
#        return feature_encoding

def learn_w_raw_state():
    agent = KnightJoustStateBasedAgent()
    
    arbitrator = rl.ArbitratorStandard(agent, NUM_TRIALS, NUM_EPISODES)
    arbitrator.run()

def learn_w_features():
    sample_state = KnightJoustState.generate_start_state()

    player = sample_state.index['player']
    opponent = sample_state.index['opponent']
    upperright = sample_state.index['upperright']
    upperleft = sample_state.index['upperleft']

#    feature_dist_po = KnightJoustFeatureDist()
#    feature_angle_west = KnightJoustFeatureAngleWest()
#    feature_angle_east = KnightJoustFeatureAngleEast()
    feature_dist_po = rl.FeatureDist('dist-po', player, opponent)
    feature_angle_west = rl.FeatureAngle('angle-west', player, opponent, upperright)
    feature_angle_east = rl.FeatureAngle('angle-east', player, opponent, upperleft)    

    feature_list = [feature_dist_po, feature_angle_west, feature_angle_east]
    agent = KnightJoustFeatureBasedAgent(rl.FeatureSet(feature_list))
    
    arbitrator = rl.ArbitratorStandard(agent, NUM_TRIALS, NUM_EPISODES)
    arbitrator.run()    

def learn_w_multitile_features():
    sample_state = KnightJoustState.generate_start_state()

    player = sample_state.index['player']
    opponent = sample_state.index['opponent']
    upperright = sample_state.index['upperright']
    upperleft = sample_state.index['upperleft']
    
    offsets = rl.TiledFeature.EVEN_OFFSETS
    
    feature_list = []

    for offset in offsets:
#        feature_dist_po = KnightJoustFeatureDist(offset)
#        feature_angle_west = KnightJoustFeatureAngleWest(offset)
#        feature_angle_east = KnightJoustFeatureAngleEast(offset)
        feature_dist_po = rl.FeatureDist(player, opponent, 
                rl.TiledFeature.DEFAULT_NUM_TILES, offset)
        feature_angle_west = rl.FeatureAngle(player, opponent, upperright, 
                rl.TiledFeature.DEFAULT_NUM_TILES, offset)
        feature_angle_east = rl.FeatureAngle(player, opponent, upperleft,
                rl.TiledFeature.DEFAULT_NUM_TILES, offset)
        feature_list.append(feature_dist_po)
        feature_list.append(feature_angle_west)
        feature_list.append(feature_angle_east)
        
    agent = KnightJoustFeatureBasedAgent(rl.FeatureSet(feature_list))

    arbitrator = rl.ArbitratorStandard(agent, NUM_TRIALS, NUM_EPISODES)
    arbitrator.run()    

def learn_evolutionary():
    base_agent = KnightJoustFeatureBasedAgent(rl.FeatureSet([]))

    sample_state = base_agent.environment.generate_start_state()
    state_vars = sample_state.state_variables
    
    retile_featurizer = rl.FeaturizerRetile(state_vars)
    angle_featurizer = rl.FeaturizerAngle(state_vars)
    dist_featurizer = rl.FeaturizerDist(state_vars)
    
    featurizers_map = [(0.2, retile_featurizer),
                       (0.4, angle_featurizer),
                       (0.4, dist_featurizer)]

    arbitrator = rl.ArbitratorEvolutionary(base_agent, featurizers_map, 
                    NUM_GENERATIONS, POPULATION_SIZE, GENERATION_EPISODES,
                    CHAMPION_TRIALS, rl.DEFAULT_ETA)
    arbitrator.run()    

def external_config():
#    for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
#    for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
    w = float(sys.argv[1])
    print "Mutate weights multiplier is %.2f" % w
    rl.MUTATE_NEW_WEIGHTS_MULT = w
    learn_evolutionary()
        
def test_stuff():
#    feature_dist_po = KnightJoustFeatureDist()
#    feature_angle_west = KnightJoustFeatureAngleWest()
#    feature_angle_east = KnightJoustFeatureAngleEast()
#    feature_list = [feature_dist_po, feature_angle_west, feature_angle_east]
    player_x = int(random.random() * 25)
    player_y = int(random.random() * 25)
    opponent_x = int(random.random() * 25)
    opponent_y = int(random.random() * 25)

    point_range = ((0, 0), 
                   (KnightJoustEnvironment.GRID_SIZE - 1, 
                    KnightJoustEnvironment.GRID_SIZE - 1))

    player_var = rl.StateVarPoint2D("player", player_x, player_y,
            point_range, is_continuous=True, is_dynamic=True)
    opponent_var = rl.StateVarPoint2D("opponent", opponent_x, opponent_y, 
            point_range, is_continuous=True, is_dynamic=True)
    corner_var = rl.StateVarPoint2D("corner", #@UnusedVariable
            KnightJoustEnvironment.GRID_SIZE - 1, 
            KnightJoustEnvironment.GRID_SIZE - 1, point_range, 
            is_continuous=True, is_dynamic=False)
    state = KnightJoustState([player_var, opponent_var])
    
#    feature = rl.FeatureTiledPoint2D("player", player_var, 4)
#    feature = rl.FeaturePoint1DTiled("player", player_var, use_row=False)
#    feature = rl.FeatureAngle("angle", player_var, opponent_var, 
#                corner_var)
    feature = rl.FeatureDist("dist", player_var, opponent_var)
    
    feature_list = [feature]
    feature_set = rl.FeatureSet(feature_list)
    for i in range(10): #@UnusedVariable
        
        state.index['player'].x = int(random.random() * 25)
        state.index['player'].y = int(random.random() * 25)
        state.index['opponent'].x = int(random.random() * 25)
        state.index['opponent'].y = int(random.random() * 25)
#        state.index['player'].x = 20
#        state.index['player'].y = 11
#        state.index['opponent'].x = 9
#        state.index['opponent'].y = 0

        print state, 
        print feature_set.encode_state(state)

if __name__ == '__main__':

#    learn_w_raw_state()
#    learn_w_features()
#    learn_w_multitile_features()
#    learn_evolutionary()
    external_config()
#    test_stuff()
