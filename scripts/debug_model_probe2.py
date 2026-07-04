import os
import torch
import torch.nn.functional as F
from ov_piano.data.maestro import MetaMAESTROv3, MelMaestro
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.utils import load_model

HDF5_MEL_PATH = os.path.join('datasets', 'MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5')
HDF5_ROLL_PATH = os.path.join('datasets', 'MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5')
MAESTRO_PATH = os.path.join('datasets', 'maestro', 'maestro-v3.0.0')
SNAPSHOT_INPATH = os.path.join('out', 'model_snapshots', 'OnsetsAndVelocities_2026_02_07_11_29_44.362.torch')

meta = MetaMAESTROv3(MAESTRO_PATH, splits=['validation'], years=MetaMAESTROv3.ALL_YEARS)
maestro = MelMaestro(HDF5_MEL_PATH, HDF5_ROLL_PATH, *(x[0] for x in meta.data), as_torch_tensors=False)
mel, roll, md = maestro[0]

key_beg, key_end = (21, 109)
num_piano_keys = key_end - key_beg
model = OnsetsAndVelocities(in_chans=2, in_height=mel.shape[0], out_height=num_piano_keys,
                            conv1x1head=(128,128), bn_momentum=0, leaky_relu_slope=0.1, dropout_drop_p=0)
load_model(model, SNAPSHOT_INPATH, eval_phase=True)
model = model.eval()
if torch.cuda.is_available():
    model = model.cuda()

# Run on a chunk
chunk = torch.from_numpy(mel).to(next(model.parameters()).device).unsqueeze(0)[..., :100]
print('Chunk shape:', chunk.shape)
out = model(chunk)
print('Raw out types:', [type(o) for o in out], 'len=', len(out))

# Simulate model_inference wrapper
try:
    if out is None:
        print('out is None')
    if isinstance(out, (list, tuple)) and len(out) >= 3:
        probs, vels, pedals = out
    else:
        print('Unexpected model output structure')
        raise RuntimeError('unexpected')

    print('probs type:', type(probs))
    if isinstance(probs, (list, tuple)):
        chosen = probs[-1]
        print('chosen probs shape:', chosen.shape)
        probs_tensor = F.pad(torch.sigmoid(chosen), (1,0))
    else:
        print('probs is tensor, shape:', probs.shape)
        probs_tensor = F.pad(torch.sigmoid(probs[-1]), (1,0))

    print('vels type/shape:', type(vels), getattr(vels, 'shape', None))
    vels_tensor = F.pad(torch.sigmoid(vels), (1,0))

    print('pedals type/shape:', type(pedals), getattr(pedals, 'shape', None))
    pedals_tensor = F.pad(torch.sigmoid(pedals), (1,0))

    print('Produced probs_tensor shape:', probs_tensor.shape)
    print('Produced vels_tensor shape:', vels_tensor.shape)
    print('Produced pedals_tensor shape:', pedals_tensor.shape)
except Exception as e:
    print('Wrapper simulation failed:', e)

print('Done')
