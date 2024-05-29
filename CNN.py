import numpy as np
import torch
import torchaudio
import torch.nn as nn
from torch import flatten
from torch.nn import functional as F

from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )


    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class MMSEncoder(nn.Module):
    def __init__(self, in_features, nclass: int = 2, **kwargs):
        super(MMSEncoder, self).__init__()
        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=512, win_length=400,
                hop_length=160, window_fn=torch.hamming_window, n_mels=40),
        )
       
        # Load the MMS model and processor
        model_id = "/home/notebook/code/personal/S9055428/2024_finvcup_baseline/facebook/mms-lid-4017"
        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.mms_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
       
        # Modify the final classification layer to match the number of classes
        self.mms_model.classifier = nn.Linear(self.mms_model.classifier.in_features, nclass)
       
    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            print(f"x1.shape:{x.shape}")
        if x.dim() == 3 and x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)
            print(f"x2.shape:{x.shape}")
        if x.size(-1) < 16:
            x = F.pad(x, (0, 16 - x.size(-1)))

        # Process inputs to match the MMS model's requirements
        inputs = self.processor(x.squeeze(1).cpu().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(self.mms_model.parameters()).device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.mms_model(**inputs).logits
       
        return outputs


if __name__ == "__main__":
    model = MMSEncoder(in_features=1, nclass=2)
    x = torch.Tensor(np.random.rand(32, 32000))
    y = model(x)
    print(y.shape)
