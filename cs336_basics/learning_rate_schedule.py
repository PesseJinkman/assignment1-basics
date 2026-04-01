import math

def lr_cosine_schedule(t: int, alpha_min: float, alpha_max: float, t_w, t_c):
    if t<t_w:
        return (t/t_w)*alpha_max
    
    if t_w<=t<=t_c:
        return alpha_min + 0.5*(1+math.cos((t-t_w)/(t_c-t_w)*math.pi))*(alpha_max-alpha_min)

    return alpha_min