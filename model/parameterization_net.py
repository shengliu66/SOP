import torch
import torch.nn as nn
import torch.nn.functional as F
class LabelParameterization(nn.Module):
    def __init__(self, n_samples, n_class, init='gaussian', mean=0., std=1e-4):
        super(LabelParameterization, self).__init__()
        self.n_samples = n_samples
        self.n_class = n_class
        self.init = init

        self.s = nn.Parameter(torch.empty(n_samples, n_class, dtype=torch.float32))
        self.t = nn.Parameter(torch.empty(n_samples, n_class, dtype=torch.float32))
        self.history = torch.zeros(n_samples, n_class, dtype=torch.float32).cuda()
        self.init_param(mean=mean, std=std)

    def init_param(self, mean=0., std=1e-4):
        if self.init == 'gaussian':
            torch.nn.init.normal_(self.s, mean=mean, std=std)
            torch.nn.init.normal_(self.s, mean=mean, std=std)
        elif self.init == 'zero':
            torch.nn.init.constant_(self.s, 0)
            torch.nn.init.constant_(self.t, 0)
        else:
            raise TypeError('Label not initialized.')

    def compute_loss(self):
        param_y = self.s * self.s - self.t * self.t
        return torch.linalg.norm(param_y, ord=2)**2
        
    def forward(self, feature, idx):
        y = feature #F.softmax(feature, dim=1)


        param_y = self.s[idx] * self.s[idx] - self.t[idx] * self.t[idx]

        history = 0.3 * param_y + 0.7 * self.history[idx]

        self.history[idx] = history.detach()

        #self.s_history[idx] * self.s_history[idx] - self.t_history[idx] * self.t_history[idx]

        # self.s_history[idx] = 0.3 * self.s[idx] + 0.7 * self.s_history[idx]
        # self.t_history[idx] = 0.3 * self.t[idx] + 0.7 * self.t_history[idx]

        assert param_y.shape == y.shape, 'Label and param shape do not match.'
        return y + history, y
        