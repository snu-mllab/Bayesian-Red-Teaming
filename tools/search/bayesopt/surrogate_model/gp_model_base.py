from tools.search.bayesopt.fitting.fitting import fit_model
from tools.search.bayesopt.surrogate_model.model_wrapper import BaseModel
import torch
import numpy as np
import gc
from tools.search.bayesopt.acquisition.acquisition_function.acquisition_functions import expected_improvement
import gpytorch
import copy
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.constraints import GreaterThan

class MyGPModelBase(BaseModel):
    def __init__(self, fit_iter : int = 20):
        self.model = None 
        self.partition_size = 512
        self.fit_iter = fit_iter

    def reinit(self):
        del self.model
        self.model = None

    def fit(self, embs, scores):
        train_X = embs.to(dtype=torch.double).cuda()
        train_Y = torch.DoubleTensor(scores).view(-1,1).cuda()

        if type(self.model) == type(None):
            params = None
        else:
            params = copy.deepcopy(self.model.state_dict())
        del self.model
        self.model = SingleTaskGP(train_X=train_X, train_Y=train_Y).cuda()
        self.model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        self.model.mean_module.initialize(constant=-1.0)
        self.model.train()
        if params:
            self.model.load_state_dict(params)
        self.model = fit_model(surrogate_model=self.model, train_X=train_X, fit_iter=self.fit_iter)
    
    def predict(self, eval_X):
        self.model.eval()
        self.model.likelihood.eval()

        N, L = eval_X.shape
        N_pt = int(np.ceil(N/self.partition_size))
        
        means = []
        variances = []
        with gpytorch.settings.fast_pred_var():
            for i in range(N_pt):
                eval_X_pt = eval_X[self.partition_size*i : self.partition_size*(i+1)]
                pred_pt = self.model(eval_X_pt)
                pred_pt = self.model.likelihood(pred_pt)
                mean_pt, variance_pt = pred_pt.mean.detach(), pred_pt.variance.clamp_min(1e-9).detach()
                means.append(mean_pt)
                variances.append(variance_pt)
        mean = torch.cat(means, dim=0)
        variance = torch.cat(variances, dim=0)
        return mean, variance
    
    def acquisition(self, embs, bias=None, add_mean=0.0):
        eval_X = embs.to(dtype=torch.double).cuda()
        with torch.no_grad():
            mean, var = self.predict(eval_X)
            ei = expected_improvement(mean + add_mean, var, bias).cpu().detach()
        del mean, var, eval_X
        return ei

    def get_covar(self, embs):
        eval_X = embs.to(dtype=torch.double).cuda()
        with torch.no_grad():
            covar = self.model.posterior(eval_X).mvn.covariance_matrix
        return covar 