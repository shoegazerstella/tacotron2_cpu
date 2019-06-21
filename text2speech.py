# -*- coding: utf-8 -*-
#

import sys, os
#sys.path.append('waveglow/')

import numpy as np
import time

import torch
import librosa

#from .model import Tacotron2
#from .layers import TacotronSTFT, STFT
#from .audio_processing import griffin_lim

from hparams_tts import create_hparams
from train_tts import load_model
from text import text_to_sequence
from waveglow.denoiser import Denoiser


def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

def load_tts_model(checkpoint_path=None, waveglow_path=None):

    # set-up params
    hparams = create_hparams()

    # load model from checkpoint
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
    _ = model.eval()

    # Load WaveGlow for mel2audio synthesis and denoiser
    waveglow = torch.load(waveglow_path, map_location='cpu')['model']
    waveglow.eval() 

    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    return model, denoiser, waveglow, hparams

def speechGeneration(model, denoiser, waveglow, hparams, text, outAudioPath, removeBias=False):

    # text pre-processing
    text = text.replace('\n\n', '')
    text = text.replace('\n', '')

    # Prepare text input
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long()
    
    # decode text input
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)

    # Synthesize audio from spectrogram using WaveGlow
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    
    # (Optional) Remove WaveGlow bias
    if removeBias:
        audio = denoiser(audio, strength=0.01)[:, 0]

    # save
    audio = audio.cpu().numpy()
    audio = audio.astype('float64')

    librosa.output.write_wav(outAudioPath, audio[0], hparams.sampling_rate, norm=False)

    return


if __name__ == "__main__":

    # load model
    start = time.time()
    model, denoiser, waveglow, hparams = load_tts_model(checkpoint_path="models/tacotron2_statedict.pt", waveglow_path="models/waveglow_old.pt")
    print('model loaded in: ', time.time() - start, 'seconds')

    # generate speech and save audio
    inputText = "Hello Musixmatch, how are you?"
    outAudioPath = "test.wav"

    start = time.time()
    speechGeneration(model, denoiser, waveglow, hparams, inputText, outAudioPath)
    print('inference done in: ', time.time() - start, 'seconds')
