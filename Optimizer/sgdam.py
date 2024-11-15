import torch
from torch.optim.optimizer import Optimizer, required
import math
import copy

#from Utils.compressor import get_n_bits



class SGDAM(Optimizer):

    def __init__(self, params,num_domain, lr=required,
                 momentum_primal=required, momentum_dual=required,
                 rho_primal=required, rho_dual=required,
                 period=1, margin=1.0, clip_value=1.0, reg_coeff=500.0,
                 dampening=0, weight_decay=0, nesterov=False,
                 **kwargs):

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid primal learning rate eta: {}".format(lr))

        if momentum_primal is not required and momentum_primal < 0.0:
            raise ValueError("Invalid primal learning rate gamma: {}".format(momentum_primal))
        if momentum_dual is not required and momentum_dual < 0.0:
            raise ValueError("Invalid dual learning rate gamma: {}".format(momentum_dual))
        if rho_primal is not required and rho_primal < 0.0:
            raise ValueError("Invalid primal learning rate beta: {}".format(rho_primal))
        if rho_dual is not required and rho_dual < 0.0:
            raise ValueError("Invalid dual learning rate beta: {}".format(rho_dual))


        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if margin < 0.0:
            raise ValueError("Invalid margin value: {}".format(margin))

        defaults = dict(lr=lr,
                        momentum_primal=momentum_primal, momentum_dual=momentum_dual,
                        rho_primal=rho_primal, rho_dual=rho_dual,
                        margin=margin, clip_value=clip_value, reg_coeff=reg_coeff,
                        dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

        if nesterov and (momentum_dual <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(SGDAM, self).__init__(params, defaults)


        # define sorted param names.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )
        self.num_domain = num_domain
        self.T = 0
        self.cnt = 0
        self.period = period


    def __setstate__(self, state):
        super(SGDAM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('dscgdagp', False)

    def step(self):
        """Performs a single optimization step."""

        # 0. store the original model parameters
        param_groups_original = copy.deepcopy(self.param_groups)
        # 2. update the primal  model parameters with local gradients
        #for group in self.param_groups[:-1]:
        for group in self.param_groups:
            if "dual_alpha_" in group["name"]:
                continue
            weight_decay = group['weight_decay']
            momentum_primal = group['momentum_primal']
            rho_primal = group['rho_primal'] #local learning rate
            eta = group['lr'] #coefficient for convex combination

            clip_value = group['clip_value']
            reg_coeff = group['reg_coeff']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                # get param_state
                param_state = self.state[p]

                # State initialization
                if 'model_ref' not in param_state:
                    param_state['model_ref'] = torch.zeros_like(p.data)
                if 'model_acc' not in param_state:
                    param_state['model_acc'] = torch.zeros_like(p.data)

                # update momentum with local gradients
                if 'momentum_buffer' not in param_state:
                    momentum_buffer = param_state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    momentum_buffer = param_state['momentum_buffer']
                    momentum_buffer.mul_(1 - momentum_primal).add_(grad, alpha=momentum_primal)

                # update model parameters with gradients: \tilde{x}_{t+1} = wx_{t} - \gamma z_{t}
                model_ref = param_state['model_ref']
                model_acc = param_state['model_acc']

                p.data = p.data - rho_primal * eta * (torch.clamp(momentum_buffer, -clip_value, clip_value)
                                                        + 1.0 / reg_coeff * (p.data - model_ref)
                                                        + weight_decay * p.data)


                # === update accumulated model parameters =====
                model_acc.add_(p.data)

        # 3. update the dual  model parameters with local gradients
        for i in range(self.num_domain):
            group_a = param_groups_original[-3 - 3* (i)]
            group_b = param_groups_original[-2 - 3* (i)]
            group_alpha = self.param_groups[-1 - 3*(i)]
            #group_alpha =  param_groups_original[-1*(i+1)]
            
            # print(group_a['name'],-3 - 3* (i))
            # print(group_b['name'],-2 - 3* (i))
            # print(group_alpha['name'],-1 - 3*(i))
            
            if group_a["name"] != "primal_a_{}".format(self.num_domain - i -1 ):
                print("primal_a_{}".format(self.num_domain - i-1),group_a["name"])
                raise RuntimeError('Not group primal_a')
            if group_b["name"] != "primal_b_{}".format(self.num_domain - i -1):
                print("primal_b_{}".format(self.num_domain - i-1),group_b["name"])
                raise RuntimeError('Not group primal_b')
            if group_alpha["name"] != "dual_alpha_{}".format(self.num_domain - i-1):
                raise RuntimeError('Not group alpha')

            p_a = group_a["params"][0]
            p_b = group_b["params"][0]
            p_alpha = group_alpha["params"][0]

            # print(p_alpha.grad)
            # print(self.param_groups)
            # print(self.param_names)

            momentum_dual = group_alpha['momentum_dual']
            rho_dual = group_alpha['rho_dual']
            eta = group_alpha['lr']
            margin = group_alpha['margin']

            grad = p_alpha.grad.data

            param_state = self.state[p_alpha]
            if 'momentum_buffer' not in param_state:
                momentum_buffer = param_state['momentum_buffer'] = torch.clone(grad).detach()
            else:
                momentum_buffer = param_state['momentum_buffer']
                momentum_buffer.mul_(1 - momentum_dual).add_(2*(margin + p_b.data - p_a.data) - 2*p_alpha.data, alpha=momentum_dual)

            p_alpha.data.add_(momentum_buffer, alpha=rho_dual*eta)
            p_alpha.data = torch.clamp(p_alpha.data, 0, 999)


        self.T += 1
        self.cnt += 1


    def update_regularizer(self, decay_factor=None):

        if decay_factor != None:
            self.param_groups[-1]['lr'] = self.param_groups[-1]['lr'] / decay_factor
            print ('Reducing learning rate to %.5f @ T=%s!' % (self.param_groups[-1]['lr'], self.T))

        print ('Updating regularizer @ T=%s!' % (self.T))

        for group in self.param_groups[:-1]:

            if decay_factor != None:
                group['lr'] = group['lr'] / decay_factor

            for p in group['params']:
                param_state = self.state[p]

                param_state['model_ref'] = param_state['model_acc'] / self.T
                param_state['model_acc'] = torch.zeros_like(p.data)

        self.T = 0

