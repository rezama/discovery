#/usr/bin/env python
'''
Created on May 8, 2011

@author: reza
'''
import rl
import random
import time
import sys
import copy

# Debug
DEBUG = False

ALPHA = 0.1
GAMMA = 0.95
EPSILON = 0.05
LAMBDA = 0.9

# hand coded agent's params
STAY_PROB = 0.2

# roles
ROLE_PLAYER = 'player'
ROLE_OPPONENT = 'opponent'

MAX_STEPS = 400

# evolutionary parameters
NUM_GENERATIONS = rl.DEFAULT_NUM_GENERATIONS
POPULATION_SIZE = 100
NUM_GENERATION_EPISODES = rl.DEFAULT_NUM_GENERATION_EPISODES
NUM_CHAMPION_TRIALS = rl.DEFAULT_NUM_CHAMPION_TRIALS
NUM_BEST_CHAMPION_TRIALS = rl.DEFAULT_NUM_TRIALS
NUM_BEST_CHAMPION_EPISODES = rl.DEFAULT_NUM_EPISODES

# standard parameters
NUM_TRIALS = rl.DEFAULT_NUM_TRIALS
NUM_EPISODES = rl.DEFAULT_NUM_EPISODES

# always on the left
class MiniSoccerAgent(rl.AgentFeatureBased):
    
    def __init__(self, feature_set):
        self.opponent_agent = MiniSoccerAgentHandCoded(role=ROLE_OPPONENT)
        actions = MiniSoccerActions()
        environment = MiniSoccerEnvironment(self.opponent_agent)
        algorithm = rl.SarsaLambdaFeaturized(actions, environment, feature_set,
                                             ALPHA, EPSILON, LAMBDA)
        super(MiniSoccerAgent, self).__init__(actions, environment, feature_set,
                                              algorithm)
#        self.set_algorithm()

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

    def __init__(self, role=ROLE_OPPONENT, opponent_agent=None):
        self.role = role
        # only the player needs an environment as agent.transition() is called
        # by the arbitrator and it in turn calls environment.respond().
        # The agent on the right does not need to have the environment object
        self.opponent_agent = None
        environment = None
        if role == ROLE_PLAYER:
            self.opponent_agent = opponent_agent
            environment = MiniSoccerEnvironment(self.opponent_agent)
        super(MiniSoccerAgentRandom, self).__init__(MiniSoccerActions(), environment)
#        self.set_algorithm(None)

    def begin_episode(self, state):
        super(MiniSoccerAgentRandom, self).begin_episode(state)
        if self.opponent_agent is not None:
            self.opponent_agent.begin_episode(state)
        
    def select_action(self):
        return self.actions.random_action()

class MiniSoccerAgentHandCoded(rl.AgentStateBased):

    def __init__(self, role=ROLE_OPPONENT, opponent_agent=None):
        self.role = role
        # only the player needs an environment as agent.transition() is called
        # by the arbitrator and it in turn calls environment.respond().
        # The agent on the right does not need to have the environment object
        self.opponent_agent = None
        environment = None
        if role == ROLE_PLAYER:
            self.opponent_agent = opponent_agent
            environment = MiniSoccerEnvironment(self.opponent_agent)
        super(MiniSoccerAgentHandCoded, self).__init__(MiniSoccerActions(), 
                                                       environment)
#        self.set_algorithm(None)

    def begin_episode(self, state):
        super(MiniSoccerAgentHandCoded, self).begin_episode(state)
        if self.opponent_agent is not None:
            self.opponent_agent.begin_episode(state)
        
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
            if self.role == ROLE_OPPONENT:
                if not player_has_ball.truth: # opponent (we) has the ball
                    diff_x = opponent.x - left_goal_center.x
                    diff_y = opponent.y - left_goal_center.y
                    if abs(diff_x) > abs(diff_y):
                        action = MiniSoccerActions.W
                    elif diff_y > 0:
                        action = MiniSoccerActions.S
                    else:
                        action = MiniSoccerActions.N
                else: # player (them) has the ball
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
            else: # agent is playing as the player
                if not player_has_ball.truth: # opponent (them) has the ball
                    diff_x = opponent.x - player.x
                    diff_y = opponent.y - player.y
                    if abs(diff_x) > 1.5 * abs(diff_y):
                        if diff_x > 0:
                            action = MiniSoccerActions.E
                        else:
                            action = MiniSoccerActions.W
                    else:
                        if diff_y > 0:
                            action = MiniSoccerActions.N
                        else:
                            action = MiniSoccerActions.S
                else: # player (we) has the ball
                    diff_x = right_goal_center.x - player.x
                    diff_y = right_goal_center.y - player.y
                    if abs(diff_x) > abs(diff_y):
                        action = MiniSoccerActions.E
                    elif diff_y > 0:
                        action = MiniSoccerActions.N
                    else:
                        action = MiniSoccerActions.S
            
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
#        return self.REWARD_WIN
        distance_to_goal = self.MAX_X - MiniSoccerState.PLAYER_X_START
        return self.REWARD_WIN * (GAMMA ** distance_to_goal)

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

    PLAYER_X_START = MiniSoccerEnvironment.FIELD_WIDTH / 4
    PLAYER_Y_START = MiniSoccerEnvironment.MIN_GOAL_Y
    OPPONENT_X_START = MiniSoccerEnvironment.FIELD_WIDTH * 3 / 4
    OPPONENT_Y_START = MiniSoccerEnvironment.MAX_GOAL_Y

    def __init__(self, state_variables):
        environment_vars = MiniSoccerEnvironment.get_environment_vars()
        super(MiniSoccerState, self).__init__(state_variables + environment_vars)
        
    @classmethod
#    def generate_start_state(cls, role=ROLE_ATTACK):
    def generate_start_state(cls):
        point_range = ((MiniSoccerEnvironment.MIN_X - 1, MiniSoccerEnvironment.MIN_Y),
                       (MiniSoccerEnvironment.MAX_X + 1, MiniSoccerEnvironment.MAX_Y))
        
#        if role == cls.ROLE_ATTACK:
#            PLAYER_X_START = MiniSoccerEnvironment.FIELD_WIDTH / 4
#            PLAYER_Y_START = MiniSoccerEnvironment.MIN_GOAL_Y
#            OPPONENT_X_START = MiniSoccerEnvironment.FIELD_WIDTH * 3 / 4
#            OPPONENT_Y_START = MiniSoccerEnvironment.MAX_GOAL_Y
#            player_has_ball = True
#        else:
#            PLAYER_X_START = MiniSoccerEnvironment.FIELD_WIDTH * 3 / 4
#            PLAYER_Y_START = MiniSoccerEnvironment.MAX_GOAL_Y
#            OPPONENT_X_START = MiniSoccerEnvironment.FIELD_WIDTH / 4
#            OPPONENT_Y_START = MiniSoccerEnvironment.MIN_GOAL_Y
#            player_has_ball = False
        
#        ball_with_player = random.choice((True, False))
        ball_with_player = True
#        ball_with_player = False

        player = rl.StateVarPoint2D("player", cls.PLAYER_X_START, cls.PLAYER_Y_START,
                point_range, is_dynamic=True, is_continuous=True)
        opponent = rl.StateVarPoint2D("opponent", cls.OPPONENT_X_START, cls.OPPONENT_Y_START,
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

def try_hand_coded():
    opponent = MiniSoccerAgentHandCoded(role=ROLE_OPPONENT, opponent_agent=None)
    agent = MiniSoccerAgentHandCoded(role=ROLE_PLAYER, opponent_agent=opponent)
    
    arbitrator = rl.ArbitratorStandard(agent, NUM_TRIALS, NUM_EPISODES)
    arbitrator.run()

def cost_benchmark():
    sample_state = MiniSoccerState.generate_start_state()

    player = sample_state.index['player']
    opponent = sample_state.index['opponent']
#    player_on_left = sample_state.index['player_on_left']
    player_has_ball = sample_state.index['player_has_ball']
    right_goal_center = sample_state.index['rightgoalcenter']
    left_goal_center = sample_state.index['leftgoalcenter']
    upper_left = sample_state.index['upperleft']
    
    print "training the base agent..."
    base_features = [rl.FeatureFlag(player_has_ball),
                     rl.FeatureAngle(player, upper_left, left_goal_center),
                     rl.FeatureDistY(player, right_goal_center),
                     rl.FeaturePointXY(player)
                     ]
    base_agent = MiniSoccerAgent(rl.FeatureSet(base_features))
    a = time.clock()
    arbitrator = rl.ArbitratorStandard(base_agent, NUM_TRIALS, NUM_EPISODES)
    arbitrator.run(MAX_STEPS)
    b = time.clock()
    base_time = b - a
    print "Running time: %.1f" % base_time
    print "Do it again..."
    a = time.clock()
    arbitrator = rl.ArbitratorStandard(base_agent, NUM_TRIALS, NUM_EPISODES)
    arbitrator.run(MAX_STEPS)
    b = time.clock()
    base_time = b - a
    print "Running time: %.1f" % base_time
    
    feature_lists = [
        [rl.FeatureFlag(player_has_ball)], 
        [rl.FeatureAngle(opponent, left_goal_center, upper_left)],
        [rl.FeatureAngle(opponent, left_goal_center, upper_left, 20)],
        [rl.FeatureDist(opponent, player)],
        [rl.FeatureDist(opponent, player, 20)],
        [rl.FeatureDistX(opponent, player)],
        [rl.FeatureDistX(opponent, player, 20)],
        [rl.FeaturePointXY(opponent)],
        [rl.FeaturePointXY(opponent, 400)],
        [rl.FeatureInteraction([rl.FeatureDist(opponent, player), rl.FeatureAngle(opponent, left_goal_center, upper_left)])],
        [rl.FeatureInteraction([rl.FeatureDist(opponent, player, 20), rl.FeatureAngle(opponent, left_goal_center, upper_left)])],
        [rl.FeatureInteraction([rl.FeatureDist(opponent, player), rl.FeaturePointXY(opponent)])],
        [rl.FeatureInteraction([rl.FeatureDist(opponent, player, 20), rl.FeaturePointXY(opponent)])],
    ]
    
    for feature_list in feature_lists:
        agent = base_agent.clone()
        for feature in feature_list:
            agent.add_feature(feature) 
        arbitrator = rl.ArbitratorStandard(agent, NUM_TRIALS, NUM_EPISODES)
        print "testing %s..." % feature_list
        a = time.clock()
        arbitrator.run(MAX_STEPS)
        b = time.clock()
        print "Overhead time: %.1f" % (b - a - base_time)
        print

def learn_w_multitile_features():
    sample_state = MiniSoccerState.generate_start_state()

    player = sample_state.index['player']
    opponent = sample_state.index['opponent']
    player_has_ball = sample_state.index['player_has_ball']
    right_goal_center = sample_state.index['rightgoalcenter']
    left_goal_center = sample_state.index['leftgoalcenter']
    upper_right = sample_state.index['upperright']
    lower_right = sample_state.index['lowerleft']
        
    features = [
        rl.FeatureDist(player, opponent),
        rl.FeatureDist(player, right_goal_center),
        rl.FeatureDist(player, left_goal_center),
        rl.FeatureDist(opponent, right_goal_center),
        rl.FeatureDist(opponent, left_goal_center),
        rl.FeatureAngle(player, opponent, upper_right),
        rl.FeatureAngle(player, opponent, lower_right),
    ]
    
    offsets = rl.TiledFeature.EVEN_OFFSETS
    
    feature_list = []
    feature_list.append(rl.FeatureFlag(player_has_ball))
    for offset in offsets:
        for i in range(len(features)):
            the_feature = copy.deepcopy(features[i])
            the_feature.offset = offset
            feature_list.append(the_feature)

    agent = MiniSoccerAgent(rl.FeatureSet(feature_list))
            
    arbitrator = rl.ArbitratorStandard(agent, NUM_TRIALS, NUM_EPISODES)
    arbitrator.run(MAX_STEPS)

def learn_evolutionary():
    base_agent = MiniSoccerAgent(rl.FeatureSet([]))

    sample_state = base_agent.environment.generate_start_state()
    state_vars = sample_state.state_variables
    
    featurizer_retile = rl.FeaturizerRetile(state_vars)
    featurizer_interaction = rl.FeaturizerInteraction(state_vars)
    featurizer_angle = rl.FeaturizerAngle(state_vars)
    featurizer_dist = rl.FeaturizerDist(state_vars)
    featurizer_dist_x = rl.FeaturizerDistX(state_vars)
    featurizer_dist_y = rl.FeaturizerDistY(state_vars)
    featurizer_flag = rl.FeaturizerFlag(state_vars)
    featurizer_point_xy = rl.FeaturizerPointXY(state_vars)
    featurizer_point_x = rl.FeaturizerPointX(state_vars)
    featurizer_point_y = rl.FeaturizerPointY(state_vars)
    
    featurizers_map = [(0.12, featurizer_retile), #@UnusedVariable
                       (0.15, featurizer_interaction),
                       (0.10, featurizer_flag),
                       (0.16, featurizer_angle),
                       (0.12, featurizer_dist),
                       (0.09, featurizer_dist_x),
                       (0.09, featurizer_dist_y),
                       (0.07, featurizer_point_xy),
                       (0.05, featurizer_point_x),
                       (0.05, featurizer_point_y)
                       ]

#    featurizers_map = [(0.15, featurizer_retile),
#                       (0.10, featurizer_interaction),
#                       (0.10, featurizer_flag),
#                       (0.20, featurizer_angle),
#                       (0.15, featurizer_dist),
#                       (0.10, featurizer_dist_x),
#                       (0.10, featurizer_dist_y),
#                       (0.10, featurizer_point_xy),
#                       (0.0, featurizer_point_x),
#                       (0.0, featurizer_point_y)
#                       ]

    arbitrator = rl.ArbitratorEvolutionary(base_agent, featurizers_map, 
                    NUM_GENERATIONS, POPULATION_SIZE,
                    NUM_GENERATION_EPISODES, NUM_CHAMPION_TRIALS,
                    NUM_BEST_CHAMPION_EPISODES, NUM_BEST_CHAMPION_TRIALS,
                    rl.DEFAULT_ETA)
    arbitrator.run(MAX_STEPS)

def external_config_eta():
    eta = float(sys.argv[1])
    rl.DEFAULT_ETA = eta
    print "Eta is %.2f" % rl.DEFAULT_ETA
    
def external_config_w():
    w = float(sys.argv[1])
    rl.MUTATE_NEW_WEIGHTS_MULT = w
    print "Mutate weights multiplier is %.2f" % rl.MUTATE_NEW_WEIGHTS_MULT
    
if __name__ == '__main__':
#    try_hand_coded()
#    cost_benchmark()
    learn_w_multitile_features()
#    external_config_w()
#    print rl.MUTATE_NEW_WEIGHTS_MULT
#    external_config_eta()
#    print rl.DEFAULT_ETA
    learn_evolutionary()
