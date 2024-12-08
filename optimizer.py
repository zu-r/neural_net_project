import torch

class SDLMOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.02):
        defaults = dict(lr=lr, alpha=alpha)
        super(SDLMOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # Estimate the diagonal Hessian (approximation)
                hessian_diag = torch.mean(grad ** 2, dim=0)
                step_size = 1 / (hessian_diag + alpha)
                p.data.add_(-step_size * grad)
        return loss
