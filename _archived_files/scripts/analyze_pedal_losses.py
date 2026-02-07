import re
import statistics
from pathlib import Path

LOG = Path('out/txt_logs/2026_01_29_19_35_50.787[1_train_onsets_velocities.py].log')
pattern = re.compile(r'"pedal":\s*([0-9]+\.?[0-9eE+-]*)')

pedals = []
with LOG.open('r', encoding='utf-8') as f:
    for line in f:
        if '"TRAIN"' in line and '"losses"' in line:
            m = pattern.search(line)
            if m:
                pedals.append(float(m.group(1)))

if not pedals:
    print('No pedal values found')
else:
    import numpy as np
    arr = np.array(pedals)
    print('count', len(pedals))
    print('min', arr.min())
    print('median', float(np.median(arr)))
    print('mean', float(arr.mean()))
    print('max', arr.max())
    print('std', float(arr.std()))
    # last values
    print('\nlast 10 values:')
    print('\n'.join(f'{x:.6f}' for x in pedals[-10:]))
    # simple trend: compare mean of first 10% vs last 10%
    n = len(pedals)
    first10 = arr[:max(1,int(0.1*n))].mean()
    last10 = arr[-max(1,int(0.1*n)):].mean()
    print('\nfirst10_mean', float(first10))
    print('last10_mean', float(last10))
    print('trend: decreasing' if last10 < first10 else 'trend: increasing' if last10>first10 else 'trend: flat')
