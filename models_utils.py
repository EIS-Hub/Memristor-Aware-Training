import numpy as np
import math
import torch
from scipy.stats import norm
from collections import OrderedDict
from utils import Noisy_Inference

# defining the model: either a MLP, a SRNN or a CNN

class MLP( torch.nn.Module ):
    def __init__(self, hidden_size=[128,], input_size=784, output_size=10, noise_inference=False, noise_sd=0.05):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        layers = []
        for i in range( len(hidden_size)+1 ):
            if i==0:
                # if noise_inference: layers = layers + [( ('fc'+str(i)), Linear_noisy(input_size, hidden_size[i], noise_sd=noise_sd, with_bias=False) )]
                if noise_inference: layers = layers + [( ('fc'+str(i)), Linear(input_size, hidden_size[i], noise_inference=noise_inference, noise_sd=noise_sd, bias=False) )]
                else: layers = layers + [( ('fc'+str(i)), torch.nn.Linear(input_size, hidden_size[i], bias=False) )]
            elif i==len(hidden_size):
                # if noise_inference: layers = layers + [( ('fc'+str(i)), Linear_noisy(hidden_size[i-1], output_size, noise_sd=noise_sd, with_bias=False) )]
                if noise_inference: layers = layers + [( ('fc'+str(i)), Linear(hidden_size[i-1], output_size, noise_inference=noise_inference, noise_sd=noise_sd, bias=False) )]
                else: layers = layers + [( ('fc'+str(i)), torch.nn.Linear(hidden_size[i-1], output_size, bias=False) )]
            elif i!=0 and i!=len(hidden_size):
                # if noise_inference: layers = layers + [( ('fc'+str(i)), Linear_noisy(hidden_size[i-1], hidden_size[i], noise_sd=noise_sd, with_bias=False) )]
                if noise_inference: layers = layers + [( ('fc'+str(i)), Linear(hidden_size[i-1], hidden_size[i], noise_inference=noise_inference, noise_sd=noise_sd, bias=False) )]
                else: layers = layers + [( ('fc'+str(i)), torch.nn.Linear(hidden_size[i-1], hidden_size[i], bias=False) )]
        self.layers = torch.nn.ModuleDict(OrderedDict( layers ))
    
    def generate_hidden_weights( self ):
        for p in self.parameters():
            p.hid = p.data.clone()

    def forward(self, x):
        x = x.view( -1, self.input_size )
        for i in range( len(self.hidden_size)+1 ):
            x = self.layers['fc'+str(i)](x)
            if i != len(self.hidden_size):
                x = torch.nn.functional.relu( x )
        #x = torch.nn.functional.log_softmax( x, dim=1 ) #torch.nn.functional.softmax( x, dim=1 )
        return x
    

# Neurons MonteCarlo Simulation fit
def sample_tau_neuron( taus, popt_tau2sd=np.array([11.85344581,  0.4342391 ]) ):
    taus_flat = np.array(taus).flatten()
    taus_flat_log = np.log(taus_flat)
    sds = popt_tau2sd[-1] + taus*popt_tau2sd[-2]
    taus_samples = np.exp( norm.rvs(*np.array([taus_flat_log, sds]), size=taus_flat.shape[0] ) )
    return np.reshape(taus_samples, newshape=taus.shape)
    
def sample_tau_synapse(taus, sd_tau=0.32187836150454946):
    taus_flat = np.array(taus).flatten()
    taus_flat_log = np.log(taus_flat)
    sds = np.ones_like( taus_flat ) * sd_tau
    taus_samples = np.exp( norm.rvs(*np.array([taus_flat_log, sds]), size=taus_flat.shape[0] ) )
    return np.reshape(taus_samples, newshape=taus.shape)

params = {
    'nb_inputs'  : 4,
    'nb_channels' : 2,
    'nb_hidden'  : 100, #64
    'nb_outputs' : 2,
    'lr' : 1e-3,
    
    'tau_mem' : 20e-3,
    'tau_syn' : 5e-3,
    'tech_flag' : True,
    'init_type' : 'unif',
    'weight_scale' : 1,
    'surrogate_grad_scale' : 10,
    'L1_total_spikes' : 1e-7,
    'L2_per_neuron' : 1e-7,
    'activation' : 'spiking',

    'noise_inference' : False,
    'noise_sd' : 0.1,

    'batch_size' : 16,

    'device' : 'cpu',
    'dtype' : torch.float
}

class RSNN(torch.nn.Module):
    def __init__(self, params):
        super(RSNN, self).__init__()
        
        self.params = params
        self.tau_mem = params['tau_mem']
        self.tau_syn = params['tau_syn']

        self.alpha   = np.exp(-params['time_step']/self.tau_syn).astype(np.float32)
        self.beta    = np.exp(-params['time_step']/self.tau_mem).astype(np.float32)

        self.noise_inference = params['noise_inference']
        self.noise_sd = params['noise_sd']
        Noisy_Inference.noise_sd = params['noise_sd']
        self.noiser = Noisy_Inference.apply

        # time constant with technology plausibility
        if params['tech_flag']:
            alpha_h_np = np.exp( - params['time_step']/ \
                                sample_tau_neuron(np.array([self.tau_syn]*params['nb_hidden']) ) )
            beta_h_np = np.exp( - params['time_step']/ \
                                sample_tau_neuron(np.array([self.tau_mem]*params['nb_hidden']) )  )             
            self.alpha_h = torch.from_numpy( alpha_h_np ).type(params['dtype']).to(params['device'])
            self.beta_h = torch.from_numpy( beta_h_np ).type(params['dtype']).to(params['device'])
            alpha_o_np = np.exp( - params['time_step']/ \
                                sample_tau_neuron(np.array([self.tau_syn]*params['nb_outputs']) ) )
            beta_o_np = np.exp( - params['time_step']/ \
                                sample_tau_neuron(np.array([self.tau_mem]*params['nb_outputs']) )  )             
            self.alpha_o = torch.from_numpy( alpha_o_np ).type(params['dtype']).to(params['device'])
            self.beta_o = torch.from_numpy( beta_o_np ).type(params['dtype']).to(params['device'])
        else:
            self.alpha_h = self.alpha; self.alpha_o = self.alpha
            self.beta_h  = self.beta ; self.beta_o  = self.beta
            
        # weight initialization
        self.w1 = torch.nn.Parameter( torch.empty((params['nb_inputs'], params['nb_hidden']),  device=params['device'], 
                         dtype=params['dtype'], requires_grad=True) )
        self.w2 = torch.nn.Parameter( torch.empty((params['nb_hidden'], params['nb_outputs']), device=params['device'], 
                         dtype=params['dtype'], requires_grad=True) )
        self.v1 = torch.nn.Parameter( torch.empty((params['nb_hidden'], params['nb_hidden']), device=params['device'], 
                         dtype=params['dtype'], requires_grad=True) )
        
        weight_scale = params['weight_scale']
        if params['init_type'] == 'unif':
            w1_scale = weight_scale/np.sqrt(params['nb_inputs'])
            torch.nn.init.uniform_(self.w1, a=0.001-w1_scale, b=0.001+w1_scale)
            w2_scale = weight_scale/np.sqrt(params['nb_hidden'])
            torch.nn.init.uniform_(self.w2, a=0.001-w2_scale, b=0.001+w2_scale)
            v1_scale = weight_scale/np.sqrt(params['nb_hidden'])
            torch.nn.init.uniform_(self.v1, a=0.001-v1_scale, b=0.001+v1_scale)
        else:
            torch.nn.init.normal_(self.w1, mean=0.0, std=weight_scale/np.sqrt(params['nb_inputs']))
            torch.nn.init.normal_(self.w2, mean=0.0, std=weight_scale/np.sqrt(params['nb_hidden']))
            torch.nn.init.normal_(self.v1, mean=0.0, std=weight_scale/np.sqrt(params['nb_hidden']))
        
        self.v1_mask = torch.ones_like( self.v1, device=params['device'], dtype=params['dtype'] )
        self.v1_mask = self.v1_mask - torch.eye( params['nb_hidden'], device=params['device'], dtype=params['dtype'] )
    

    def forward(self, inputs):
        syn = torch.zeros((inputs.size(0), self.params['nb_hidden']), device=self.params['device'], dtype=self.params['dtype'])
        mem = torch.zeros((inputs.size(0), self.params['nb_hidden']), device=self.params['device'], dtype=self.params['dtype'])

        mem_rec = [mem]
        spk_rec = [mem]

        # eventually apply the noise to the weights
        if self.noise_inference:
            w1 = self.noiser( self.w1 )
            v1 = self.noiser( self.v1 )
            w2 = self.noiser( self.w2 )
        else:
            w1, v1, w2 = self.w1, self.v1, self.w2

        # Compute hidden layer activity
        h1 = torch.zeros((inputs.size(0), self.params['nb_hidden']), device=self.params['device'], dtype=self.params['dtype'])
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1))
        for t in range(inputs.size(1)):
            h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac", (h1, v1*self.v1_mask))
            mthr = mem-1.0
            if self.params['activation'] == 'spiking':
                out = spike_fn(mthr)
            else: 
                out = torch.nn.functional.tanh(mthr)
            rst = torch.zeros_like(mem)
            c   = (mthr > 0)
            rst[c] = torch.ones_like(mem)[c]

            new_syn = self.alpha_h*syn +h1
            new_mem = self.beta_h*mem +syn -rst

            mem = new_mem
            syn = new_syn

            mem_rec.append(mem)
            spk_rec.append(out)

        mem_rec = torch.stack(mem_rec,dim=1)
        spk_rec = torch.stack(spk_rec,dim=1)

        # Readout layer
        h2= torch.einsum("abc,cd->abd", (spk_rec, w2))
        flt = torch.zeros((inputs.size(0),self.params['nb_outputs']), device=self.params['device'], dtype=self.params['dtype'])
        out = torch.zeros((inputs.size(0),self.params['nb_outputs']), device=self.params['device'], dtype=self.params['dtype'])
        out_rec = [out]
        for t in range(inputs.size(1)):
            new_flt = self.alpha_o*flt +h2[:,t]
            new_out = self.beta_o*out +flt

            flt = new_flt
            out = new_out

            out_rec.append(out)

        out_rec = torch.stack(out_rec,dim=1)
        other_recs = [mem_rec, spk_rec]
        return out_rec, other_recs


class RSNN_2( torch.nn.Module ):
    def __init__( self, in_features=64, hidden_size=128, out_features=20, tau_n=40e-3, tau_s=10e-3, time_stamp=1e-3,
                  noise_forward=False, mixed_precision=False, noise_sd=0.05, num_levels=15, device='cpu' ):
        super(RSNN_2, self).__init__()
        self.in_feature = in_features; self.hidden_size=hidden_size; self.out_features=out_features
        self.tau_n = tau_n; self.tau_s = tau_s; self.time_stamp = time_stamp
        self.alpha_n = np.exp( -time_stamp/tau_n )
        self.alpha_s = np.exp( -time_stamp/tau_s )
        self.noise_forward = noise_forward; self.mixed_precision = mixed_precision
        self.noise_sd = noise_sd; self.num_levels = num_levels
        self.device = device

        # weight placeholders
        self.w_in  = torch.nn.Parameter( torch.zeros( (hidden_size, in_features), device=device ) )
        self.w_rec = torch.nn.Parameter( torch.zeros( (hidden_size, hidden_size), device=device ) )
        self.w_out = torch.nn.Parameter( torch.zeros( (out_features, hidden_size), device=device ) )
        # initialization of the weights
        torch.nn.init.kaiming_uniform_( self.w_in  )
        torch.nn.init.kaiming_uniform_( self.w_rec )
        torch.nn.init.kaiming_uniform_( self.w_out )

    def generate_hidden_weights( self ):
        for p in self.parameters():
            p.hid = p.data.clone()

    def forward( self, x ):
        x = x.to(self.device)
        # initialize neurons and synapses
        batch_size, t_steps = x.size(1), x.size(0)
        syn = torch.zeros( ( batch_size, self.hidden_size ), device=self.device )
        mem = torch.zeros( ( batch_size, self.hidden_size ), device=self.device )
        z   = torch.zeros( ( batch_size, self.hidden_size ), device=self.device )
        sut = torch.zeros( ( batch_size, self.out_features), device=self.device )
        out = torch.zeros( ( batch_size, self.out_features), device=self.device )
        # recordings
        spk_hist, out_hist = [], []
        for t in range(t_steps):
            syn = syn + torch.mm( x[t], self.w_in.T ) + torch.mm( z, self.w_rec.T )
            syn = syn * self.alpha_s
            z = spike_fn( mem-1.0 )
            rst = z.detach()
            mem = mem - rst*mem
            #mem_hist.append(mem)
            spk_hist.append( z )
            mem = mem + syn
            mem = mem * self.alpha_n
            out = out + torch.mm( z, self.w_out.T )
            out = out * self.alpha_s
            #out = out + sut
            #out = out * self.alpha_n
            out_hist.append(out)
        spk_hist = torch.stack(spk_hist, dim=1)
        out_hist = torch.stack(out_hist, dim=1)
        return out_hist, spk_hist
    
class SurrGradSpike(torch.autograd.Function):    
    scale = 10.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SurrGradSpike.apply


class Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor
    # noise_inference : bool
    # noise_sd : float

    def __init__(self, in_features: int, out_features: int, bias: bool = True, noise_inference : bool = False, noise_sd : float = 0.1,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.noise_inference = noise_inference
        self.set_noiser( noise_sd = noise_sd )

    def set_noiser( self, noise_sd = 0.05 ) -> None:
        self.noise_sd = noise_sd
        Noisy_Inference.noise_sd = noise_sd
        self.noiser = Noisy_Inference.apply

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.noise_inference:
            weight = self.noiser(self.weight)
            if self.bias is not False:
                bias = self.noiser(self.bias)
            else: bias = self.bias
        else: 
            weight = self.weight
            bias = self.bias
        return torch.nn.functional.linear(input, weight, bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, noise_inference={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.noise_inference
        )




class Linear_noisy(torch.nn.Module):
    """ Custom Linear layer that introduces noise in the forward pass """
    def __init__(self, in_features, out_features, noise_sd=0.01, with_bias:bool=False, remove_noise:bool=False):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        weights = torch.Tensor(out_features, in_features)
        self.weights = torch.nn.Parameter(weights)
        self.with_bias = with_bias
        self.remove_noise = remove_noise
        if with_bias:
            bias = torch.Tensor(out_features)
            self.bias = torch.nn.Parameter(bias)

        # initialize weights and biases
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
        torch.nn.init.kaiming_uniform_(self.weights) # weight init
        if with_bias:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)  # bias init
        Noisy_Inference.noise_sd = noise_sd
        self.noiser = Noisy_Inference.apply

    # def forward(self, x):
    #     if self.remove_noise: w_times_x= torch.mm(x, self.weights.t() )
    #     else: w_times_x= torch.mm(x, self.noiser( self.weights.t() ) )
    #     if self.with_bias:
    #         return torch.add(w_times_x, self.bias) 
    #     else: 
    #         return w_times_x
    
    def forward(self, x):
        if self.remove_noise:
            weights = self.weights
            if self.with_bias: biases = self.bias
        else: 
            weights = self.noiser( self.weights )
            if self.with_bias: biases = self.noiser( self.bias )
            
        if self.with_bias:
            return torch.nn.functional.linear(x, weight=weights, bias=biases )
        else: 
            return torch.nn.functional.linear(x, weight=weights )


# QAT optimizer
class Adam_QAT(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(Adam_QAT, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_QAT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
            
                    if len(p.size())!=1:
                        state['followed_weight'] = np.random.randint(p.size(0)),np.random.randint(p.size(1))
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])


                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                

                #binary_weight_before_update = torch.sign(p.data)
                #condition_consolidation = (torch.mul(binary_weight_before_update,exp_avg) > 0.0 )

                #decay = torch.max(0.3*torch.abs(p.data), torch.ones_like(p.data))
                #decayed_exp_avg = torch.mul(torch.ones_like(p.data)-torch.pow(torch.tanh(group['meta']*torch.abs(p.data)),2) ,exp_avg)
                #exp_avg_2 = torch.div(exp_avg, decay)
  
                if len(p.size())==1: # True if p is bias, false if p is weight
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p.data.addcdiv_(-step_size, exp_avg , denom)  #normal update
                    #p.data.addcdiv_(-step_size, torch.where(condition_consolidation, decayed_exp_avg, exp_avg) , denom)  #assymetric lr for metaplasticity
                    
        return loss