import torch
import torch.nn as nn
import numpy as np

class SimpleDenoiseNet(nn.Module):
    def __init__(self):
        super(SimpleDenoiseNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )

    def forward(self, x):
        return self.fc(x)

class MLNoiseSuppressor:
    def __init__(self, model_path=None):
        self.model = SimpleDenoiseNet()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def denoise(self, audio_frame):
        # Normalize and convert to tensor
        norm = np.max(np.abs(audio_frame)) + 1e-8
        audio_frame = audio_frame.astype(np.float32) / norm
        audio_tensor = torch.from_numpy(audio_frame).unsqueeze(0)

        with torch.no_grad():
            out = self.model(audio_tensor).squeeze(0).numpy()

        # De-normalize
        out = out * norm
        return out.astype(np.int16)
