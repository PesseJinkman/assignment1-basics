import math
import torch    

class AdamW(torch.optim.Optimizer):

    def __init__(
        self, 
        params,
        lr : float = 1e-3, 
        betas : tuple[float, float] = (0.9, 0.95),  
        eps : float = 1e-8, 
        weight_decay: float = 0.1
    ) -> None:

        defaults = {
            'lr':lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, defaults)

        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:

            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                m = state.get("m", 0)
                v = state.get("v", 0)
                t = state.get("t", 1)
                g = p.grad.data
                m = b1*m + (1-b1)*g
                v = b2*v + (1-b2)*(g**2)
                lr_t = lr*(math.sqrt(1-b2**t)/(1-b1**t))
                p.data -= lr_t*(m/(torch.sqrt(v)+eps))
                p.data -= lr*weight_decay*p.data
                state["m"] = m
                state["v"] = v
                state["t"] = t+1
        
        return loss
