import torch

STOCH_PARAMS = {
                        0: [0.6, 0.4, 0],
                        1: [0.6, 0.2, 0.2],
                        2: [0.8, 0.2, 0.0],
                        3: [0.8, 0.,  0.2],
                        4: [0.4, 0.3, 0.3]}

# @profile(stream=open('logmem_svrp_sto_gc_tocpu4.log', 'w+'))
def get_stoch_var(inp, locs, w, alphas=None, A=0.6, B=0.2, G=0.2, **kw):
    '''
    locs: [batch, num_customers, 2]
    '''
    # h = hpy().heap()
    stoch_idx = STOCH_PARAMS[kw["stoch_idx"]]
    A, B, G = stoch_idx
    if inp.dim() <= 2:
        inp_ =  inp[..., None]
    else:
        inp_ = inp.clone()

    n_problems,n_nodes,shape = inp_.shape
    T = inp_/A
    
    var_noise = T*G

    noise = torch.sqrt(var_noise)*torch.randn(n_problems,n_nodes, shape).to(T.device)      #=np.rand.randn, normal dis(0, 1)
    noise = torch.clamp(noise, min=-var_noise, max=var_noise)

    var_w = T*B
    sum_alpha = var_w[:, :, None, :]*9
    
    if alphas is None:  
        alphas = torch.rand((n_problems, 1, 9, shape)).to(T.device)
    alphas_loc = locs.sum(-1)[..., None, None]/2 * alphas  # [batch, num_loc, 2]-> [batch, num_loc] -> [batch, num_loc, 1, 1], [batch, 1, 9,1]
    alphas_loc *= sum_alpha     # alpha value [4.5*var_w]
    signs = torch.rand((n_problems, n_nodes, 9, shape)).to(T.device)
    alphas_loc[torch.where(signs > 0.5)] *= -1     # half negative: 0 mean, [sqrt(-4.5*var_w) ,s sqrt(4.5*var_w)]
    
    w1 = w.repeat(1, 1, 3)[..., None]       # [batch, nodes, 3*repeat3=9, 1]
    # roll shift num in axis: [batch, nodes, 3] -> concat [batch, nodes, 9,1]
    w2 = torch.concatenate([w, torch.roll(w,shifts=1,dims=2), torch.roll(w,shifts=2,dims=2)], 2)[..., None]
    
    tot_w = (alphas_loc*w1*w2).sum(2)       # alpha_i * wm * wn, i[1-9], m,n[1-3], [batch, nodes, 9]->[batch, nodes,1]
    tot_w = torch.clamp(tot_w, min=-var_w, max=var_w)
    out = torch.clamp(inp_ + tot_w + noise, min=0.0)
    
    # del tot_w, noise
    del var_noise, sum_alpha, alphas_loc, signs, w1, w2, tot_w
    del T, noise, var_w
    del inp_
    # gc.collect()
    
    return out