import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.models import Conformer
from intervaltree import Interval, IntervalTree

    
def zcr_extractor(wav, win_length, hop_length):
    pad_length = win_length // 2
    wav = np.pad(wav, (pad_length, pad_length), 'constant')
    num_frames = 1 + (wav.shape[0] - win_length) // hop_length
    zcrs = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop_length
        end = start + win_length
        zcr = np.abs(np.sign(wav[start+1:end])-np.sign(wav[start:end-1]))
        zcr = np.sum(zcr) * 0.5 / win_length
        zcrs[i] = zcr
    return zcrs.astype(np.float32)


def feature_extractor(wav, sr=16000):
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=int(sr*0.025), 
                                         hop_length=int(sr*0.01), n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)
    zcr = zcr_extractor(wav, win_length=int(sr*0.025), hop_length=int(sr*0.01))
    vms = np.var(mel, axis=0)

    mel = torch.tensor(mel).unsqueeze(0)
    zcr = torch.tensor(zcr).unsqueeze(0)
    vms = torch.tensor(vms).unsqueeze(0)

    zcr = zcr.unsqueeze(1).expand(-1, 128, -1)
    vms = torch.var(mel, dim=1).unsqueeze(1).expand(-1, mel.shape[1], -1)

    feature = torch.stack((mel, vms, zcr), dim=1)
    length = torch.tensor([zcr.shape[-1]])
    return feature, length


def get_silence_ts(s_segs, total_len: int, sr=16000, seg_pad=0):
    no_speech = [((s_segs[i]['end'] + seg_pad)*sr,
                  (s_segs[i+1]['start'] + seg_pad)*sr) 
                    for i in range(len(s_segs) - 1)]
    if len(no_speech) > 0: 
      no_speech[0] = (max(no_speech[0][0], 0), no_speech[0][1])
      no_speech[-1] = (no_speech[-1][0], min(no_speech[-1][1], total_len))

      if s_segs[0]['start'] > 0:
        no_speech = [(0, s_segs[0]['start']*sr)] + no_speech 
      if s_segs[-1]['end']*sr < total_len:
        no_speech = no_speech + [(s_segs[0]['end']*sr, total_len)]

    return no_speech


def extract_non_speech(wav, speech_segs, sr=1600, seg_pad=0, min_length=20):
    no_speech = get_silence_ts(speech_segs, wav.shape[-1], sr, seg_pad)
    
    if len(no_speech) == 0:
        return []

    no_speech = [ns_ts for ns_ts in no_speech 
                 if (ns_ts[1] - ns_ts[0]) > min_length]

    feature_batch = [list(feature_extractor(wav[int(s):int(e)])) + [(s, e)] 
                    for (s, e) in no_speech]

    #features, lengths = zip(*feature_batch) 
    
    # features is passed as a list, since padding would be inneficient, 
    # due to the great differences in wav length
    return feature_batch


def preprocess_respiro(wav_path, speech_segs, sr=16000):
    wav, sr = librosa.load(wav_path, sr=sr)
    return extract_non_speech(wav, speech_segs, sr)


class DetectorWrapper:
    """
    """
    def __init__(self, model, device=None):
        super().__init__()
        self.model = model
        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def __call__(self, input, threshold=0.064, min_length=20, sr=16000):
        feature, length, no_speech = input
        feature, length = feature.to(self.device), length.to(self.device)

        output = self.model(feature, length)

        preds = self._process_output(output, threshold, min_length, 
                                     pred_start=no_speech[0]/sr)
        return preds
    
    def _process_output(self, output, threshold, min_length, pred_start=0):
        prediction = (output[0] > threshold).nonzero().squeeze().tolist()
        tree = IntervalTree()
        if isinstance(prediction, list) and len(prediction)>1:
            diffs = np.diff(prediction)
            splits = np.where(diffs != 1)[0] + 1
            splits = np.split(prediction, splits)
            splits = list(filter(lambda split: len(split)>min_length, splits))
            for split in splits:
                if split[-1]*0.01>split[0]*0.01:
                    tree.add(Interval(round(split[0]*0.01 + pred_start, 2), 
                                      round(split[-1]*0.01+ pred_start, 2)))
        return tree

    def postprocess(self, output: List[IntervalTree]):
        tree = IntervalTree()
        for other in output:
            tree = tree.union(other)
        return tree


class TwoStageBreathDetector:
    def __init__(self, stage_2nd, preprocess_2nd,
                 stage_1st=None, preprocess_1st=None, 
                 postprocess=None,
                 device=None):
        super().__init__()
        self.stage_1st = stage_1st
        self.stage_2nd = stage_2nd

        self.preprocess_1st = preprocess_1st
        self.preprocess_2nd = preprocess_2nd

        self.postprocess = postprocess

        if not device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def __call__(self, wav_path, threshold=0.064, min_length=20):
        if self.stage_1st is not None and self.preprocess_1st is not None:
            output_1 = self.preprocess_1st(wav_path)
            output_1 = self.stage_1st(output_1)
        else:
            output_1 = None

        output_2 = self.preprocess_2nd(wav_path, output_1)
        if isinstance(output_2, list): # batch wasn't padded
            
            output_2 = [self.stage_2nd(wav) for wav in output_2]
            
            if self.postprocess is not None:
                output_2 = self.postprocess(output_2)
        else:
            output_2 = self.stage_2nd(output_2)
          
        return output_2
