import os
import torch
import torch.nn.functional as F
from ov_piano.data.maestro import MetaMAESTROv3, MelMaestro
from ov_piano.models.ov import OnsetsAndVelocities
from ov_piano.utils import load_model

# Config (match script defaults)
HDF5_MEL_PATH = os.path.join('datasets', 'MAESTROv3_logmel_sr=16000_stft=2048w384h_mel=229(50-8000).h5')
HDF5_ROLL_PATH = os.path.join('datasets', 'MAESTROv3_roll_quant=0.024_midivals=128_extendsus=True.h5')
MAESTRO_PATH = os.path.join('datasets', 'maestro', 'maestro-v3.0.0')
SNAPSHOT_INPATH = os.path.join('out', 'model_snapshots', 'OnsetsAndVelocities_2026_02_07_11_29_44.362.torch')

print('Probe: loading metadata and first mel...')
meta = MetaMAESTROv3(MAESTRO_PATH, splits=['validation'], years=MetaMAESTROv3.ALL_YEARS)
maestro = MelMaestro(HDF5_MEL_PATH, HDF5_ROLL_PATH, *(x[0] for x in meta.data), as_torch_tensors=False)

print('Dataset length:', len(maestro))
mel, roll, md = maestro[0]
print('Mel shape:', mel.shape)

# Build model
key_beg, key_end = (21, 109)  # PIANO_MIDI_RANGE substitute
num_piano_keys = key_end - key_beg
model = OnsetsAndVelocities(in_chans=2, in_height=mel.shape[0], out_height=num_piano_keys,
                            conv1x1head=(128,128), bn_momentum=0, leaky_relu_slope=0.1, dropout_drop_p=0)

print('Loading snapshot...')
try:
    load_model(model, SNAPSHOT_INPATH, eval_phase=True)
    print('Snapshot loaded')
except Exception as e:
    print('Snapshot load failed:', e)

model = model.eval()
if torch.cuda.is_available():
    model = model.cuda()

# Prepare tensor and a small chunk
tmel = torch.from_numpy(mel).to(next(model.parameters()).device).unsqueeze(0)
print('tmel device/shape:', tmel.device, tmel.shape)
chunk = tmel[..., :100]
print('Running model on chunk shape:', chunk.shape)
try:
    out = model(chunk)
    print('Model returned type:', type(out))
    try:
        print('Len of out:', len(out))
    except Exception:
        pass
    if isinstance(out, (list, tuple)):
        for i, o in enumerate(out):
            print(f'out[{i}] type={type(o)} shape={(tuple(o.shape) if hasattr(o, "shape") else "N/A")}')
    else:
        print('Model returned single tensor with shape:', getattr(out, 'shape', 'N/A'))
except Exception as e:
    print('Model call exception:', e)

print('Done')
