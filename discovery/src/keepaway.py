#/usr/bin/env python
'''
Created on Apr 28, 2011

@author: reza
'''
import random
import math
import copy

import rl

# standard parameters
NUM_TRIALS = 1
NUM_EPISODES = 2000

# evolutionary parameters
NUM_GENERATIONS = 15
POPULATION_SIZE = 50
GENERATION_EPISODES = 200

# reporting
PLOT_INTERVALS = 100

class KeepAwayAgent(rl.AgentFeatureBased):
    
    def __init__(self, feature_set):
        super(KeepAwayAgent, self).__init__(KeepAwayActions(), 
                KeepAwayEnvironment(), feature_set)
        self.set_algorithm(rl.SarsaLambdaFeaturized(self))
    
class KeepAwayEnvironment(rl.Environment):
    
    FIELD_SIZE = 25.0
    
    REWARD_STEP = 1.0
    
    def __init__(self):
        super(KeepAwayEnvironment, self).__init__(KeepAwayState)

    def get_max_episode_reward(self):
        return 100

    @classmethod
    def get_environment_vars(cls):
        point_range = ((0, 0),
                       (KeepAwayEnvironment.FIELD_SIZE, 
                        KeepAwayEnvironment.FIELD_SIZE))
        
        lower_left_corner_x = 0
        lower_left_corner_y = 0
        lower_right_corner_x = KeepAwayEnvironment.FIELD_SIZE
        lower_right_corner_y = 0
        upper_left_corner_x = 0
        upper_left_corner_y = KeepAwayEnvironment.FIELD_SIZE
        upper_right_corner_x = KeepAwayEnvironment.FIELD_SIZE
        upper_right_corner_y = KeepAwayEnvironment.FIELD_SIZE
        center_x = KeepAwayEnvironment.FIELD_SIZE / 2
        center_y = KeepAwayEnvironment.FIELD_SIZE / 2
        
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
        center_state_var = rl.StateVarPoint2D("center",
                center_x, center_y, 
                point_range, is_dynamic=False, is_continuous=True)
        
        return [lower_left_corner_state_var, lower_right_corner_state_var,
                upper_left_corner_state_var, upper_right_corner_state_var,
                center_state_var]
                
    def move_keeper(self, point):
        angle = random.random() * math.pi * 2
        delta_x = math.cos(angle)
        delta_y = math.sin(angle)
        point.x += delta_x
        point.y += delta_y
        if point.x < 0:
            point.x = 0
        if point.y < 0:
            point.y = 0
        if point.x > self.FIELD_SIZE:
            point.x = self.FIELD_SIZE
        if point.y > self.FIELD_SIZE:
            point.y = self.FIELD_SIZE
        
    def move_taker(self, taker, ref, pos_neg):
        delta_x = ref.x - taker.x
        delta_y = ref.y - taker.y
        ratio = 1.0 / (rl.GeometryUtil.compute_dist(
                        taker.x, taker.y, ref.x, ref.y) + 0.1)
        taker.x += delta_x * ratio * pos_neg
        taker.y += delta_y * ratio * pos_neg
        if taker.x < 0:
            taker.x = 0
        if taker.y < 0:
            taker.y = 0
        if taker.x > self.FIELD_SIZE:
            taker.x = self.FIELD_SIZE
        if taker.y > self.FIELD_SIZE:
            taker.y = self.FIELD_SIZE
        
    def respond(self, state, last_state, action):
        reward = 0
        if not state.is_final():
            me = state.index["keeper1"]
            keeper2 = state.index["keeper2"]
            keeper3 = state.index["keeper3"]
            taker1 = state.index["taker1"]
            taker2 = state.index["taker2"]
            # save to old state
            me_p = last_state.index["keeper1"]
            keeper2_p = last_state.index["keeper2"]
            keeper3_p = last_state.index["keeper3"]
            taker1_p = last_state.index["taker1"]
            taker2_p = last_state.index["taker2"]
            me_p.x = me.x
            me_p.y = me.y
            keeper2_p.x = keeper2_p.x
            keeper2_p.y = keeper2_p.y
            keeper3_p.x = keeper3_p.x
            keeper3_p.y = keeper3_p.y
            taker1_p.x = taker1.x
            taker1_p.y = taker1.y
            taker2_p.x = taker2.x
            taker2_p.y = taker2.y
            
            # respond
            if action == KeepAwayActions.KEEP:
                dist_taker1 = rl.GeometryUtil.compute_dist(me.x, me.y,
                        taker1.x, taker1.y)
                dist_taker2 = rl.GeometryUtil.compute_dist(me.x, me.y,
                        taker2.x, taker2.y)                
                prob_capture1 = min(1.0 / (dist_taker1 + 0.1), 1.0) - 0.5
                prob_capture2 = min(1.0 / (dist_taker2 + 0.1), 1.0) - 0.5
                
                rand = random.random()
                if rand < prob_capture1:
                    state.make_final()
                    return reward  
                rand = random.random()
                if rand < prob_capture2:
                    state.make_final()
                    return reward  
                reward = self.REWARD_STEP
                
            elif action == KeepAwayActions.PASS2:
                dist_taker1 = rl.GeometryUtil.compute_dist_point_line(
                        taker1.x, taker1.y, me.x, me.y, keeper2.x, keeper2.y)
                dist_taker2 = rl.GeometryUtil.compute_dist_point_line(
                        taker2.x, taker2.y, me.x, me.y, keeper2.x, keeper2.y)                
                prob_capture1 = min(1.0 / (dist_taker1 + 4.0), 1.0) + 0.1
                prob_capture2 = min(1.0 / (dist_taker2 + 4.0), 1.0) + 0.1
                rand = random.random()
                if rand < prob_capture1:
                    state.make_final()
                    return reward  
                rand = random.random()
                if rand < prob_capture2:
                    state.make_final()
                    return reward  
                reward = self.REWARD_STEP

                state.index['keeper1'] = keeper2
                state.index['keeper2'] = me
                
            elif action == KeepAwayActions.PASS3:
                dist_taker1 = rl.GeometryUtil.compute_dist_point_line(
                        taker1.x, taker1.y, me.x, me.y, keeper3.x, keeper3.y)
                dist_taker2 = rl.GeometryUtil.compute_dist_point_line(
                        taker2.x, taker2.y, me.x, me.y, keeper3.x, keeper3.y)                 
                prob_capture1 = min(1.0 / (dist_taker1 + 4.0), 1.0) + 0.1
                prob_capture2 = min(1.0 / (dist_taker2 + 4.0), 1.0) + 0.1
                rand = random.random()
                if rand < prob_capture1:
                    state.make_final()
                    return reward  
                rand = random.random()
                if rand < prob_capture2:
                    state.make_final()
                    return reward  
                reward = self.REWARD_STEP

                state.index['keeper1'] = keeper3
                state.index['keeper2'] = me
                state.index['keeper3'] = keeper2
                            
            # keeper movement
#            self.move_keeper(me)
#            self.move_keeper(keeper2)
#            self.move_keeper(keeper3)

            # taker movement
            self.move_taker(taker1, me, 1)
            self.move_taker(taker2, me, 1)
            dist_takers = rl.GeometryUtil.compute_dist(
                        taker1.x, taker1.y, taker2.x, taker2.y)
            if dist_takers < 1:
                self.move_taker(taker2, taker1, -1)
            
        return reward

class KeepAwayActions(rl.Actions):

    KEEP = "k"
    PASS2 = "2"
    PASS3 = "3"

    def __init__(self):
        actions = [self.KEEP, self.PASS2, self.PASS3]
        super(KeepAwayActions, self).__init__(actions)

    
class KeepAwayState(rl.ModularState):
    
    def __init__(self, state_variables):
        environment_vars = KeepAwayEnvironment.get_environment_vars()
        super(KeepAwayState, self).__init__(state_variables + environment_vars)
        self.final = False
        
    def is_final(self):
        return self.final
    
    def make_final(self):
        self.final = True
        
    @classmethod
    def generate_start_state(cls):
        point_range = ((0, 0),
                       (KeepAwayEnvironment.FIELD_SIZE, 
                        KeepAwayEnvironment.FIELD_SIZE))
        
        keeper1_x = 2
        keeper1_y = 2
        keeper2_x = KeepAwayEnvironment.FIELD_SIZE - 2
        keeper2_y = 2
        keeper3_x = 2
        keeper3_y = KeepAwayEnvironment.FIELD_SIZE - 2
        taker1_x = KeepAwayEnvironment.FIELD_SIZE / 2 + 2
        taker1_y = KeepAwayEnvironment.FIELD_SIZE / 2 - 2
        taker2_x = KeepAwayEnvironment.FIELD_SIZE / 2 - 2
        taker2_y = KeepAwayEnvironment.FIELD_SIZE / 2 + 2

        keeper1_var = rl.StateVarPoint2D("keeper1", keeper1_x, keeper1_y,
                point_range, is_dynamic=True, is_continuous=True)
        keeper2_var = rl.StateVarPoint2D("keeper2", keeper2_x, keeper2_y,
                point_range, is_dynamic=True, is_continuous=True)
        keeper3_var = rl.StateVarPoint2D("keeper3", keeper3_x, keeper3_y,
                point_range, is_dynamic=True, is_continuous=True)
        taker1_var = rl.StateVarPoint2D("taker1", taker1_x, taker1_y,
                point_range, is_dynamic=True, is_continuous=True)
        taker2_var = rl.StateVarPoint2D("taker2", taker2_x, taker2_y,
                point_range, is_dynamic=True, is_continuous=True)
        
        state_vars = [keeper1_var, keeper2_var, keeper3_var,
                      taker1_var, taker2_var]
        
        state = KeepAwayState(state_vars)
        return state

def learn_w_multitile_features():
    sample_state = KeepAwayState.generate_start_state()

    keeper1 = sample_state.index['keeper1']
    keeper2 = sample_state.index['keeper2']
    keeper3 = sample_state.index['keeper3']
    
    taker1 = sample_state.index['taker1']
    taker2 = sample_state.index['taker2']
    
    center = sample_state.index['center']

    features = [
        rl.FeatureDist(keeper1, center),
        rl.FeatureDist(keeper2, center),
        rl.FeatureDist(keeper3, center),
        rl.FeatureDist(taker1, center),
        rl.FeatureDist(taker2, center),
        rl.FeatureDist(keeper1, taker1),
        rl.FeatureDist(keeper1, taker2),
        rl.FeatureDist(keeper2, taker1),
        rl.FeatureDist(keeper2, taker2),
        rl.FeatureDist(keeper3, taker1),
        rl.FeatureDist(keeper3, taker2),
        rl.FeatureAngle(keeper1, keeper2, taker1),
        rl.FeatureAngle(keeper1, keeper2, taker2),
        rl.FeatureAngle(keeper1, keeper3, taker1),
        rl.FeatureAngle(keeper1, keeper3, taker2)
        ]
    
    offsets = rl.TiledFeature.EVEN_OFFSETS
    
    feature_list = []
    for offset in offsets:
        for i in range(len(features)):
            the_feature = copy.deepcopy(features[i])
            the_feature.offset = offset
            feature_list.append(the_feature)

    agent = KeepAwayAgent(rl.FeatureSet(feature_list))
            
    arbitrator = rl.ArbitratorStandard(agent, NUM_TRIALS, NUM_EPISODES)
    arbitrator.execute()

def learn_evolutionary():
    base_agent = KeepAwayAgent(rl.FeatureSet([]))

    sample_state = base_agent.environment.generate_start_state()
    state_vars = sample_state.state_variables
    
    retile_featurizer = rl.FeaturizerRetile(state_vars)
    angle_featurizer = rl.FeaturizerAngle(state_vars)
    dist_featurizer = rl.FeaturizerDist(state_vars)
    
    featurizers_map = [(0.2, retile_featurizer),
                       (0.4, angle_featurizer),
                       (0.4, dist_featurizer)]
    
    arbitrator = rl.ArbitratorEvolutionary(base_agent, featurizers_map, 
                    NUM_GENERATIONS, POPULATION_SIZE, GENERATION_EPISODES)
    arbitrator.execute()
    
if __name__ == '__main__':

#    learn_w_multitile_features()
    learn_evolutionary()
