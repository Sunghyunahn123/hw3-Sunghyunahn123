import numpy as np
from ligotools.utils import whiten, reqshift, write_wavfile
from scipy.io.wavfile import read as wavread 

def test_whiten_and_reqshift_minimal():
    fs = 4096; dt = 1.0/fs; n = 2048
    t = np.arange(n) / fs
    x = np.sin(2*np.pi*100*t)                 
    ones_psd = lambda f: np.ones_like(f)      

    y = whiten(x, ones_psd, dt)
    z = reqshift(x, fshift=200, sample_rate=fs)

    assert len(y) == len(x) and np.isfinite(y).all()
    assert len(z) == len(x) and np.isfinite(z).all()
    assert not np.allclose(z, x)             

def test_write_wavfile_minimal(tmp_path):
    fs = 8000
    x = np.zeros(1000)                       
    out = tmp_path / "test.wav"

    write_wavfile(str(out), fs, x)
    sr, data = wavread(str(out))

    assert out.exists() and sr == fs and len(data) == len(x)