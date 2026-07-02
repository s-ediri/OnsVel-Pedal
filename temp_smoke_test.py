import sys
print(sys.executable)
import torch
import pandas as pd
from ov_piano.inference import PedalDecoder
from ov_piano.eval import threshold_eval_pedals

probs = torch.tensor([[[0.10, 0.60, 0.55, 0.40, 0.35, 0.20]]], dtype=torch.float32)
decoder = PedalDecoder(num_pedals=1, threshold=0.5, hysteresis=0.1, min_hold_steps=2)
events_df, out_probs, states = decoder(probs)
print('events', events_df[['t_idx', 'event_type']].to_dict('records'))
print('states_shape', tuple(states.shape))

gt = {'pedal_idx': [0, 0], 'onset': [1.0, 3.0], 'event_type': ['onset', 'offset']}
res = threshold_eval_pedals(pd.DataFrame(gt), probs, secs_per_frame=0.016)
print('eval', res)
