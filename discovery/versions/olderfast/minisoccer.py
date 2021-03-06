'''
Created on May 8, 2011

@author: reza
'''
import copy
import rl
import random

# Debug
DEBUG = False

ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.05
LAMBDA = 0.9
USE_VARIABLE_ALPHA = False

# hand coded agent's params
STAY_PROB = 0.2

# standard parameters
NUM_TRIALS = 1
NUM_EPISODES = 100
MAX_STEPS = 1000

# evolutionary parameters
NUM_GENERATIONS = 10
POPULATION_SIZE = 50
GENERATION_EPISODES = 500

class MiniSoccerAgent(rl.AgentFeatureBased):
    
    def __init__(self, feature_set):
        self.opponent_agent = MiniSoccerAgentHandCoded()
        super(MiniSoccerAgent, self).__init__(
                MiniSoccerActions(), MiniSoccerEnvironment(self.opponent_agent), 
                feature_set)
        self.set_algorithm(rl.SarsaLambdaFeaturized(self, ALPHA, EPSILON, LAMBDA))

    def begin_episode(self, state):
        super(MiniSoccerAgent, self).begin_episode(state)
        self.opponent_agent.begin_episode(state)
        

#class MiniSoccerAgentQ(rl.Agent):
#
#    INITIAL_Q_VALUE = 1
#
#    def __init__(self, name):
#        super(MiniSoccerAgentQ, self).__init__(name)
#        self.Qvals = {}
#        self.visit_count = {}
#        
#    def begin_episode(self, state):
#        super(MiniSoccerAgentQ, self).begin_episode(state)
#
#    def best_action(self, state):
#        if state.is_final():
#            return (None, 0)
#        
#        action_values = []
#        for action in self.all_actions():
#            # insert a random number to break the ties
#            action_values.append(((self.Q(state, action), random.random()), 
#                                  action))
#            
#        action_values_sorted = sorted(action_values, reverse=True)
#        
#        action = action_values_sorted[0][1]
#        value = action_values_sorted[0][0][0]
#        
#        return (action, value)
#    
#    def select_action(self):
#        
#        if random.random() < EPSILON:
#            return MiniSoccerActions.random_action()
#        else:
#            (action, value) = self.best_action(self.state)
#            return action
#    
#    def act(self, p_action, o_action):
#        before_state = copy.deepcopy(self.state)
#        # act and transition to new state
#        r = super(MiniSoccerAgentQ, self).act(p_action, o_action)
#        if self.is_learning:
#            # update Q
#            (ap, vp) = self.best_action(self.state)
#            delta = r + GAMMA * vp - self.Q(before_state, p_action)
#            
#            before_state_str = str(before_state)
#            if USE_VARIABLE_ALPHA:
#                self.visit_count[(before_state_str, p_action)] = \
#                    self.visit_count.get((before_state_str, p_action), 0) + 1
#                alpha = 1.0 / self.visit_count[(before_state_str, p_action)]
#            else:
#                alpha = ALPHA
#            self.Qvals[(str(before_state), p_action)] += alpha * delta
#        return r
#
#    def Q(self, state, action):
#        state_str = str(state)
#        if (state_str, action) not in self.Qvals:
#            if not state.is_final():
#                self.Qvals[(state_str, action)] = self.INITIAL_Q_VALUE
#                return self.INITIAL_Q_VALUE
#            else:
#                return 0
#        else:
#            return self.Qvals[(state_str, action)]
#        
#    def print_values(self):
#        Q_keys = self.Qvals.keys()
#        Q_keys.sort()
#        print "Q:"
#        for key in Q_keys:
#            print "Q(%s) = %.2f" % (key, self.Qvals[key])

class MiniSoccerAgentRandom(rl.AgentStateBased):

    def __init__(self):
        super(MiniSoccerAgentRandom, self).__init__(
                MiniSoccerActions(), None)
        self.set_algorithm(None)       

    def select_action(self):
        return self.actions.random_action()

class MiniSoccerAgentHandCoded(rl.AgentStateBased):

    def __init__(self):
        super(MiniSoccerAgentHandCoded, self).__init__(
                MiniSoccerActions(), None)
        self.set_algorithm(None)

    def select_action(self):
        state = self.state
        player = state.index['player']
        opponent = state.index['opponent']
#        player_on_left = state.index['player_on_left']
        player_has_ball = state.index['player_has_ball']
        left_goal_center = state.index['leftgoalcenter']
        right_goal_center = state.index['rightgoalcenter']
        
        if random.random() < STAY_PROB:
            action = MiniSoccerActions.D
        else:
            if not player_has_ball.truth:  # opponent has the ball
                diff_x = opponent.x - left_goal_center.x
                diff_y = opponent.y - left_goal_center.y
                if abs(diff_x) > abs(diff_y):
                    action = MiniSoccerActions.W
                elif diff_y > 0:
                    action = MiniSoccerActions.S
                else:
                    action = MiniSoccerActions.N
            else: # player has the ball
                diff_x = opponent.x - player.x
                diff_y = opponent.y - player.y
                if abs(diff_x) > 1.5 * abs(diff_y):
                    if diff_x > 0:
                        action = MiniSoccerActions.W
                    else:
                        action = MiniSoccerActions.E
                else:
                    if diff_y > 0:
                        action = MiniSoccerActions.S
                    else:
                        action = MiniSoccerActions.N
            
        return action

class MiniSoccerEnvironment(rl.Environment):
    
    FIELD_WIDTH = 20
    FIELD_HEIGHT = 10
    GOAL_HEIGHT = 4
    
    MIN_X = 0
    MAX_X = FIELD_WIDTH - 1
    MIN_Y = 0
    MAX_Y = FIELD_HEIGHT - 1
    
    MIN_GOAL_Y = round((FIELD_HEIGHT - 1) / 2.0 - GOAL_HEIGHT / 2.0)
    MAX_GOAL_Y = MIN_GOAL_Y + GOAL_HEIGHT - 1
    
    REWARD_WIN = 1
    REWARD_LOSE = -1
    REWARD_STEP = 0
    
    def __init__(self, opponent_agent):
        self.opponent_agent = opponent_agent
        super(MiniSoccerEnvironment, self).__init__(MiniSoccerState, GAMMA)

    def get_max_episode_reward(self):
        return self.REWARD_WIN

    @classmethod
    def get_environment_vars(cls):
        point_range = ((cls.MIN_X - 1, cls.MIN_Y),
                       (cls.MAX_X + 1, cls.MAX_Y))
        
        lower_left_corner_var = rl.StateVarPoint2D("lowerleft",
                cls.MIN_X, cls.MIN_Y, 
                point_range, is_dynamic=False, is_continuous=True)
        lower_right_corner_var = rl.StateVarPoint2D("lowerright",
                cls.MAX_X, cls.MIN_Y, 
                point_range, is_dynamic=False, is_continuous=True)
        upper_left_corner_var = rl.StateVarPoint2D("upperleft",
                cls.MIN_X, cls.MAX_Y, 
                point_range, is_dynamic=False, is_continuous=True)
        upper_right_corner_var = rl.StateVarPoint2D("upperright",
                cls.MAX_X, cls.MAX_Y, 
                point_range, is_dynamic=False, is_continuous=True)
        
        center_var = rl.StateVarPoint2D("center",
                cls.FIELD_WIDTH / 2, cls.FIELD_HEIGHT / 2, 
                point_range, is_dynamic=False, is_continuous=True)

        left_goal_bottom_var = rl.StateVarPoint2D("leftgoalbottom",
                cls.MIN_X, cls.MIN_GOAL_Y, 
                point_range, is_dynamic=False, is_continuous=True)
        left_goal_top_var = rl.StateVarPoint2D("leftgoaltop",
                cls.MIN_X, cls.MAX_GOAL_Y, 
                point_range, is_dynamic=False, is_continuous=True)
        left_goal_center_var = rl.StateVarPoint2D("leftgoalcenter",
                cls.MIN_X - 1, (cls.MIN_GOAL_Y + cls.MAX_GOAL_Y) / 2,
                point_range, is_dynamic=False, is_continuous=True)
        right_goal_bottom_var = rl.StateVarPoint2D("rightgoalbottom",
                cls.MAX_X, cls.MIN_GOAL_Y, 
                point_range, is_dynamic=False, is_continuous=True)
        right_goal_top_var = rl.StateVarPoint2D("rightgoaltop",
                cls.MAX_X, cls.MAX_GOAL_Y, 
                point_range, is_dynamic=False, is_continuous=True)
        right_goal_center_var = rl.StateVarPoint2D("rightgoalcenter",
                cls.MAX_X + 1, (cls.MIN_GOAL_Y + cls.MAX_GOAL_Y) / 2,
                point_range, is_dynamic=False, is_continuous=True)

        return [lower_left_corner_var, lower_right_corner_var,
                upper_left_corner_var, upper_right_corner_var,
                center_var,
                left_goal_bottom_var, left_goal_top_var, left_goal_center_var,
                right_goal_bottom_var, right_goal_top_var, right_goal_center_var]
                
    def respond(self, state, last_state, action):
        reward = 0
        if not state.is_final():
            player = state.index['player']
            opponent = state.index['opponent']
#            player_on_left = state.index['player_on_left']
            player_has_ball = state.index['player_has_ball']
            # save to old state
            player_p = last_state.index['player']
            opponent_p = last_state.index['opponent']
#            player_on_left = state.index['player_on_left']
            player_has_ball_p = last_state.index['player_has_ball']
            player_p.x = player.x
            player_p.y = player.y
            opponent_p.x = opponent.x
            opponent_p.y = opponent.y
            player_has_ball_p.truth = player_has_ball.truth

            # select opponent's action
            o_action = self.opponent_agent.select_action()
            
            # respond
            if action == MiniSoccerActions.N:
                if player.y < self.MAX_Y:
                    player.y += 1
            elif action == MiniSoccerActions.S:
                if player.y > self.MIN_Y:
                    player.y -= 1 
            elif action == MiniSoccerActions.E:
                if (player.x < self.MAX_X) or (player_has_ball.truth and
#                                              (player_on_left.truth) and
                                              (self.MIN_GOAL_Y <= player.y <= self.MAX_GOAL_Y)): 
                    player.x += 1
            elif action == MiniSoccerActions.W:
                if (player.x > self.MIN_X): 
                    player.x -= 1
        
            if o_action == MiniSoccerActions.N:
                if opponent.y < self.MAX_Y:
                    opponent.y += 1
            elif o_action == MiniSoccerActions.S:
                if opponent.y > self.MIN_Y:
                    opponent.y -= 1 
            elif o_action == MiniSoccerActions.E:
                if (opponent.x < self.MAX_X): 
                    opponent.x += 1
            elif o_action == MiniSoccerActions.W:
                if (opponent.x > self.MIN_X) or (not player_has_ball.truth and
#                                              (player_on_left.truth) and
                                              (self.MIN_GOAL_Y <= opponent.y <= self.MAX_GOAL_Y)): 
                    opponent.x -= 1
                    
            if (player.x == opponent.x) and (player.y == opponent.y):
                player.x = player_p.x
                player.y = player_p.y
                opponent.x = opponent_p.x
                opponent.y = opponent_p.y
                player_has_ball.truth = not player_has_ball_p.truth
    
            if state.is_final():
                if player_has_ball.truth:
                    reward = self.REWARD_WIN
                else:
                    reward = self.REWARD_LOSE
            else:
                reward = self.REWARD_STEP     
            
        return reward

class MiniSoccerActions(rl.Actions):

    N = "N"
    S = "S"
    W = "W"
    E = "E"
    D = "D"
    
    def __init__(self):
        actions = [self.N, self.S, self.W, self.E, self.D]
        super(MiniSoccerActions, self).__init__(actions)

class MiniSoccerState(rl.ModularState):
    
#    ROLE_ATTACK = "ROLE_LTR"
#    ROLE_DEFEND = "ROLE_RTL"

    def __init__(self, state_variables):
        environment_vars = MiniSoccerEnvironment.get_environment_vars()
        super(MiniSoccerState, self).__init__(state_variables + environment_vars)
        
    @classmethod
#    def generate_start_state(cls, role=ROLE_ATTACK):
    def generate_start_state(cls):
        point_range = ((MiniSoccerEnvironment.MIN_X - 1, MiniSoccerEnvironment.MIN_Y),
                       (MiniSoccerEnvironment.MAX_X + 1, MiniSoccerEnvironment.MAX_Y))
        
#        if role == cls.ROLE_ATTACK:
#            player_x = MiniSoccerEnvironment.FIELD_WIDTH / 4
#            player_y = MiniSoccerEnvironment.MIN_GOAL_Y
#            opponent_x = MiniSoccerEnvironment.FIELD_WIDTH * 3 / 4
#            opponent_y = MiniSoccerEnvironment.MAX_GOAL_Y
#            player_has_ball = True
#        else:
#            player_x = MiniSoccerEnvironment.FIELD_WIDTH * 3 / 4
#            player_y = MiniSoccerEnvironment.MAX_GOAL_Y
#            opponent_x = MiniSoccerEnvironment.FIELD_WIDTH / 4
#            opponent_y = MiniSoccerEnvironment.MIN_GOAL_Y
#            player_has_ball = False
        
        player_x = MiniSoccerEnvironment.FIELD_WIDTH / 4
        player_y = MiniSoccerEnvironment.MIN_GOAL_Y
        opponent_x = MiniSoccerEnvironment.FIELD_WIDTH * 3 / 4
        opponent_y = MiniSoccerEnvironment.MAX_GOAL_Y
#        ball_with_player = random.choice((True, False))
        ball_with_player = True

        player = rl.StateVarPoint2D("player", player_x, player_y,
                point_range, is_dynamic=True, is_continuous=True)
        opponent = rl.StateVarPoint2D("opponent", opponent_x, opponent_y,
                point_range, is_dynamic=True, is_continuous=True)
        player_has_ball = rl.StateVarFlag("player_has_ball", ball_with_player, 
                                          is_dynamic=True) 
#        role = rl.StateVarFlag("player_on_left", (role == cls.ROLE_ATTACK),
#                                is_dynamic=False)
        
#        state_vars = [player, opponent, player_has_ball, role]
        state_vars = [player, opponent, player_has_ball]
        
        state = MiniSoccerState(state_vars)
        return state

    def is_final(self):
        player = self.index['player']
        opponent = self.index['opponent']
#        player_on_left = self.index['player_on_left']
        player_has_ball = self.index['player_has_ball']
        
#        if player_on_left.truth:
#            if player_has_ball.truth:
#                if player.x == MiniSoccerEnvironment.MAX_X + 1:
#                    return True
#            else:
#                if opponent.x == MiniSoccerEnvironment.MIN_X - 1:
#                    return True
#        else:
#            if player_has_ball.truth:
#                if player.x == MiniSoccerEnvironment.MIN_X - 1:
#                    return True
#            else:
#                if opponent.x == MiniSoccerEnvironment.MAX_X + 1:
#                    return True

        if player_has_ball.truth:
            if player.x == MiniSoccerEnvironment.MAX_X + 1:
                return True
        else:
            if opponent.x == MiniSoccerEnvironment.MIN_X - 1:
                return True
        
        return False
            
def learn_w_multitile_features():
    sample_state = MiniSoccerState.generate_start_state()

    player = sample_state.index['player']
    opponent = sample_state.index['opponent']
#    player_on_left = sample_state.index['player_on_left']
    player_has_ball = sample_state.index['player_has_ball']
    right_goal_center = sample_state.index['rightgoalcenter']
    left_goal_center = sample_state.index['leftgoalcenter']
    
    features = [
        rl.FeatureDist('dist-player-opponent', player, opponent),
        rl.FeatureDist('dist-player-rightgoalcenter', player, right_goal_center),
        rl.FeatureDist('dist-opponent-rightgoalcenter', opponent, right_goal_center),
        rl.FeatureDist('dist-player-leftgoalcenter', player, left_goal_center),
        rl.FeatureDist('dist-opponent-leftgoalcenter', opponent, left_goal_center),
        rl.FeatureFlag('flag-has-ball', player_has_ball)]
    
    offsets = rl.TiledFeature.EVEN_OFFSETS
    
    feature_list = []
    for offset in offsets:
        for i in range(len(features)):
            the_feature = copy.deepcopy(features[i])
            the_feature.offset = offset
            feature_list.append(the_feature)

    agent = MiniSoccerAgent(rl.FeatureSet(feature_list))
            
    arbitrator = rl.ArbitratorStandard(agent, NUM_TRIALS, NUM_EPISODES)
    arbitrator.execute(MAX_STEPS)

def learn_evolutionary():
    base_agent = MiniSoccerAgent(rl.FeatureSet([]))

    sample_state = base_agent.environment.generate_start_state()
    state_vars = sample_state.state_variables
    
    featurizer_retile = rl.FeaturizerRetile(state_vars)
    featurizer_angle = rl.FeaturizerAngle(state_vars)
    featurizer_dist = rl.FeaturizerDist(state_vars)
    featurizer_dist_x = rl.FeaturizerDistX(state_vars)
    featurizer_dist_y = rl.FeaturizerDistY(state_vars)
    featurizer_flag = rl.FeaturizerFlag(state_vars)
    featurizer_point2d = rl.FeaturizerPoint2D(state_vars)
    
    featurizers_map = [(0.20, featurizer_retile),
                       (0.35, featurizer_dist),
                       (0.50, featurizer_dist_x),
                       (0.65, featurizer_dist_y),
                       (0.80, featurizer_angle),
                       (0.90, featurizer_point2d),
                       (1.00, featurizer_flag)]
    
    arbitrator = rl.ArbitratorEvolutionary(base_agent, featurizers_map, 
                    NUM_GENERATIONS, POPULATION_SIZE, GENERATION_EPISODES)
    arbitrator.execute(MAX_STEPS)
    
if __name__ == '__main__':
#    learn_w_multitile_features()
    learn_evolutionary()

