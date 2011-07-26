from minisoccer import MiniSoccerState
import rl

sample_state = MiniSoccerState.generate_start_state()

player_has_ball = sample_state.index['player_has_ball']

feature = rl.FeatureFlag('flag-has-ball', player_has_ball)

print feature.encode_state(sample_state)
sample_state.index['player_has_ball'].truth = False
print feature.encode_state(sample_state)
