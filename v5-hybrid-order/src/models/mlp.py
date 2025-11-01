import torch
import torch.nn as nn

class RCWA_MLP_Smatrix_correction(nn.Module):
    def __init__(self, in_ch=3, out_ch=13, dtype=torch.complex64):
        super(RCWA_MLP_Smatrix_correction, self).__init__()
        self.activation = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(1, 16, dtype=dtype),
            self.activation,
            nn.Linear(16, 16, dtype=dtype),
            self.activation,
            nn.Linear(16, 16, dtype=dtype),
        )
    
    def forward(self, rcwa_result_S):
        x = torch.stack(rcwa_result_S, dim=-1)
        print("", x.shape)
        output = self.net(x)
        return output

    def save_model(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.load_state_dict(torch.load(file_path))
        print(f"Model loaded from {file_path}")
        return self

# end of file
