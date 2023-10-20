import sys
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from src.mlp import MLP
from src.utils import mmd_rbf, mmd_lin
import torch.optim as optim
import random
from collections import OrderedDict, Counter
from numbers import Number
import operator


def sample_domains(train_loader, N=1, stratified=True):
    """
    Sample N domains available in the train loader.
    """
    Ls = []
    for tl in train_loader.dataset.batches_left.values():
        Ls.append(max(tl, 0)) if stratified else Ls.append(min(tl, 1))

    positions = range(len(Ls))
    indices = []
    while True:
        needed = N - len(indices)
        if not needed:
            break
        for i in random.choices(positions, Ls, k=needed):
            if Ls[i]:
                Ls[i] = 0.0
                indices.append(i)
    return torch.tensor(indices)

class ParamDict(OrderedDict):
    """A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)


def fish_step(meta_weights, inner_weights, meta_lr):
    meta_weights, weights = ParamDict(meta_weights), ParamDict(inner_weights)
    meta_weights += meta_lr * sum([weights - meta_weights], 0 * meta_weights)
    return meta_weights

def get_score(model, x_test, y_test, t_test):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    N = len(x_test)

    # MSE
    _ypred = model.forward(x_test, t_test)
    mse = mean_squared_error(y_test, _ypred)

    # treatment index
    t_idx = np.where(t_test.to("cpu").detach().numpy().copy() == 1)[0]
    c_idx = np.where(t_test.to("cpu").detach().numpy().copy() == 0)[0]

    # ATE & ATT
    _t0 = torch.FloatTensor([0 for _ in range(N)]).reshape([-1, 1])
    _t1 = torch.FloatTensor([1 for _ in range(N)]).reshape([-1, 1])

    # _cate = model.forward(x_test, _t1) - model.forward(x_test, _t0)
    _cate_t = y_test - model.forward(x_test, _t0)
    _cate_c = model.forward(x_test, _t1) - y_test
    _cate = torch.cat([_cate_c[c_idx], _cate_t[t_idx]])
    
    _ate = np.mean(_cate.to("cpu").detach().numpy().copy())
    _att = np.mean(_cate_t[t_idx].to("cpu").detach().numpy().copy())

    return {"ATE": _ate, "ATT": _att, "RMSE": np.sqrt(mse)}


class Base(nn.Module):
    def __init__(self, cfg):
        super(Base, self).__init__()
        self.cfg = cfg
        self.criterion = nn.MSELoss(reduction='none')
        self.mse = mean_squared_error

    def fit(
        self,
        dataloader,
        x_train,
        y_train,
        t_train,
        x_test,
        y_test,
        t_test,
        logger,
    ):
        print("In fit!")
        losses = []
        ipm_result = []
        logger.debug("                          within sample,      out of sample")
        logger.debug("           [Train MSE, IPM], [RMSE, ATT, ATE], [RMSE, ATT, ATE]")
        for epoch in range(self.cfg["epochs"]):
            epoch_loss = 0
            epoch_ipm = []
            n = 0
            for (x, y, z) in dataloader:
                x = x.to(device=torch.device("cpu"))
                y = y.to(device=torch.device("cpu"))
                z = z.to(device=torch.device("cpu"))

                self.optimizer.zero_grad()

                x_rep = self.repnet(x)

                _t_id = np.where((z.cpu().detach().numpy() == 1).all(axis=1))[0]
                _c_id = np.where((z.cpu().detach().numpy() == 0).all(axis=1))[0]
                
                
                if self.cfg["split_outnet"]:
                    y_hat_treated = self.outnet_treated(x_rep[_t_id])
                    y_hat_control = self.outnet_control(x_rep[_c_id])

                    _index = np.argsort(np.concatenate([_t_id, _c_id], 0))

                    y_hat = torch.cat([y_hat_treated, y_hat_control])[_index]
                    
                else:
                    y_hat = self.outnet(torch.cat((x_rep, z), 1))
                    
                loss = self.criterion(y_hat, y.reshape([-1, 1]))
                # sample weight
                p_t = np.mean(z.cpu().detach().numpy())
                w_t = z/(2*p_t)
                w_c = (1-z)/(2*1-p_t)
                sample_weight = w_t + w_c
                if (p_t ==1) or (p_t ==0):
                    sample_weight = 1
                loss =torch.mean((loss * sample_weight))

                if self.cfg["alpha"] > 0.0:    
                    if self.cfg["ipm_type"] == "mmd_rbf":
                       ipm = mmd_rbf(
                            x_rep[_t_id],
                            x_rep[_c_id],
                            p=len(_t_id) / (len(_t_id) + len(_c_id)),
                            sig=self.cfg["sig"],
                        )
                    elif self.cfg["ipm_type"] == "mmd_lin":
                        ipm = mmd_lin(
                            x_rep[_t_id],
                            x_rep[_c_id],
                            p=len(_t_id) / (len(_t_id) + len(_c_id))
                        )
                    else:
                        logger.debug(f'{self.cfg["ipm_type"]} : TODO!!! Not implemented yet!')
                        sys.exit()

                    loss += ipm * self.cfg["alpha"]
                    epoch_ipm.append(ipm.cpu().detach().numpy())
                
                mse = self.mse(
                        y_hat.detach().cpu().numpy(),
                        y.reshape([-1, 1]).detach().cpu().numpy(),
                    )
                loss.backward()

                self.optimizer.step()
                epoch_loss += mse * y.shape[0]
                n += y.shape[0]
            
            self.scheduler.step()
            epoch_loss = epoch_loss / n
            losses.append(epoch_loss)
            if self.cfg["alpha"] > 0:
                ipm_result.append(np.mean(epoch_ipm))

            if epoch % 100 == 0:
                with torch.no_grad():
                    within_result = get_score(self, x_train, y_train, t_train)
                    outof_result = get_score(self, x_test, y_test, t_test)
                logger.debug(
                    "[Epoch: %d] [%.3f, %.3f], [%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f] "
                    % (
                        epoch,
                        epoch_loss,
                        ipm if self.cfg["alpha"] > 0 else -1,
                        within_result["RMSE"],
                        within_result["ATT"],
                        within_result["ATE"],
                        outof_result["RMSE"],
                        outof_result["ATT"],
                        outof_result["ATE"],
                    )
                )

        return within_result, outof_result, losses, ipm_result

    def train_fish(self,
        dataloader,
        x_train,
        y_train,
        t_train,
        x_test,
        y_test,
        t_test,
        logger,
        opt):
        
        print("In train fish!")
        losses = []
        ipm_result = []
        logger.debug("                                     FISH Training           ")
        logger.debug("                          within sample,      out of sample")
        logger.debug("           [Train MSE, IPM], [RMSE, ATT, ATE], [RMSE, ATT, ATE]")
        self.train()
        
        for epoch in range(self.cfg["epochs"]):
            epoch_loss = 0
            epoch_ipm = []
            n = 0
            dataloader.dataset.reset_batch()
            i = 0
            opt_inner_pre = None
            while sum([l > 1 for l in dataloader.dataset.batches_left.values()]) >= 3: #args.meta_step = 3
                i += 1
                
                # sample `meta_steps` number of domains to use for the inner loop
                domains = sample_domains(dataloader, 3).tolist()
                
                # prepare model for inner loop update
                model_inner = deepcopy(self)
                model_inner.train()

                opt_inner = opt(model_inner.parameters(), lr = 0.001)
                
                if opt_inner_pre is not None:
                    opt_inner.load_state_dict(opt_inner_pre)
                    
                # inner loop update
                for domain in domains:
                    data = dataloader.dataset.get_batch(domain)
                    # for (x, y, z, e) in data:
                    x = data[0].to(device=torch.device("cpu"))
                    y = data[1].to(device=torch.device("cpu"))
                    z = data[2].to(device=torch.device("cpu")) # Treatments
                    e = data[3].to(device=torch.device("cpu")).unsqueeze(1) # Domains
                    
                    # print(z.unique(return_counts = True))
                    # print(e.unique(return_counts = True))

                    opt_inner.zero_grad()
                    
                    x_rep = self.repnet(x)

                    _t_id = np.where((z.cpu().detach().numpy() == 1).all(axis=1))[0]
                    _c_id = np.where((z.cpu().detach().numpy() == 0).all(axis=1))[0]
                    # print(_t_id, _c_id)

                    y_hat_treated = self.outnet_treated(x_rep[_t_id])
                    y_hat_control = self.outnet_control(x_rep[_c_id])
                    # print(y_hat_control.size(), y_hat_treated.size())
                    
                    _index = np.argsort(np.concatenate([_t_id, _c_id], 0))

                    y_hat = torch.cat([y_hat_treated, y_hat_control])[_index]
                    criterion = nn.MSELoss(reduction='none')
                    
                    loss = criterion(y_hat, y.reshape([-1,1]))

                    p_t = np.mean(z.cpu().detach().numpy())
                    w_t = z/(2*p_t)
                    w_c = (1-z)/(2*1-p_t)
                    sample_weight = w_t + w_c
                    if (p_t ==1) or (p_t ==0):
                        sample_weight = 1
                    
                    loss =torch.mean((loss * sample_weight))
                    
                    loss.backward()
                    opt_inner.step()
                    
                    if self.cfg["alpha"] > 0.0:    
                        if self.cfg["ipm_type"] == "mmd_rbf":
                            ipm = mmd_rbf(
                                x_rep[_t_id],
                                x_rep[_c_id],
                                p=len(_t_id) / (len(_t_id) + len(_c_id)),
                                sig=self.cfg["sig"],
                            )
                        elif self.cfg["ipm_type"] == "mmd_lin":
                            ipm = mmd_lin(
                                x_rep[_t_id],
                                x_rep[_c_id],
                                p=len(_t_id) / (len(_t_id) + len(_c_id))
                            )
                        else:
                            logger.debug(f'{self.cfg["ipm_type"]} : TODO!!! Not implemented yet!')
                            sys.exit()

                        loss += ipm * self.cfg["alpha"]
                        epoch_ipm.append(ipm.cpu().detach().numpy())
                        
                    mse = mean_squared_error(
                            y_hat.detach().cpu().numpy(),
                            y.reshape([-1, 1]).detach().cpu().numpy(),
                        )
                    epoch_loss += mse * y.shape[0]
                    n += y.shape[0]

                StepLR(opt_inner, step_size=10, gamma=0.97)
                epoch_loss = epoch_loss / (n)
                
                if self.cfg["alpha"] > 0:
                    ipm_result.append(np.mean(epoch_ipm))
                losses.append(epoch_loss)
                opt_inner_pre = opt_inner.state_dict()
                # fish update
                meta_weights = fish_step(meta_weights=self.state_dict(),
                                        inner_weights=model_inner.state_dict(),
                                        meta_lr=0.015 / 3)     # args.meta_lr / args.meta_steps
                self.reset_weights(meta_weights)
                
                # log the number of batches left for each domain
                for domain in domains:
                    dataloader.dataset.batches_left[domain] = \
                        dataloader.dataset.batches_left[domain] - 1 \
                        if dataloader.dataset.batches_left[domain] > 1 else 1
            
            if epoch % 100 == 0:
                with torch.no_grad():
                    within_result = get_score(self, x_train, y_train, t_train)
                    outof_result = get_score(self, x_test, y_test, t_test)
                logger.debug(
                    "[Epoch: %d] [%.3f, %.3f], [%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f] "
                    % (
                        epoch,
                        epoch_loss,
                        ipm if self.cfg["alpha"] > 0 else -1,
                        within_result["RMSE"],
                        within_result["ATT"],
                        within_result["ATE"],
                        outof_result["RMSE"],
                        outof_result["ATT"],
                        outof_result["ATE"],
                    )
                )

        return within_result, outof_result, losses, ipm_result
                

class CFR(Base):
    def __init__(self, in_dim, out_dim, cfg={}):
        super().__init__(cfg)

        self.repnet = MLP(
            num_layers=cfg["repnet_num_layers"],
            in_dim=in_dim,
            hidden_dim=cfg["repnet_hidden_dim"],
            out_dim=cfg["repnet_out_dim"],
            activation=nn.ReLU(inplace=True),
            dropout=cfg["repnet_dropout"],
        )

        if cfg["split_outnet"]:

            self.outnet_treated = MLP(
                in_dim=cfg["repnet_out_dim"], out_dim=out_dim, num_layers=cfg["outnet_num_layers"], hidden_dim=cfg["outnet_hidden_dim"], dropout=cfg["outnet_dropout"]
            )
            self.outnet_control = MLP(
                in_dim=cfg["repnet_out_dim"], out_dim=out_dim, num_layers=cfg["outnet_num_layers"], hidden_dim=cfg["outnet_hidden_dim"], dropout=cfg["outnet_dropout"]
            )

            self.params = (
                list(self.repnet.parameters())
                + list(self.outnet_treated.parameters())
                + list(self.outnet_control.parameters())
            )
        else:
            self.outnet = MLP(
                in_dim=cfg["repnet_out_dim"] + 1, out_dim=out_dim, num_layers=cfg["outnet_num_layers"], hidden_dim=cfg["outnet_hidden_dim"], dropout=cfg["outnet_dropout"]
            )

            self.params = (
                list(self.repnet.parameters())
                + list(self.outnet.parameters())
            )

        self.optimizer = optim.Adam(
            params=self.params, lr=cfg["lr"], weight_decay=cfg["wd"]
        )
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=cfg["gamma"])

    def forward(self, x, z):
        with torch.no_grad():
            x_rep = self.repnet(x)

            if self.cfg["split_outnet"]:

                _t_id = np.where((z.cpu().detach().numpy() == 1).all(axis=1))[0]
                _c_id = np.where((z.cpu().detach().numpy() == 0).all(axis=1))[0]

                y_hat_treated = self.outnet_treated(x_rep[_t_id])
                y_hat_control = self.outnet_control(x_rep[_c_id])

                _index = np.argsort(np.concatenate([_t_id, _c_id], 0))
                y_hat = torch.cat([y_hat_treated, y_hat_control])[_index]
            else:
                y_hat = self.outnet(torch.cat((x_rep, z), 1))

        return y_hat

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))
        
    def get_repnet_outnet(self):
        return self.repnet, self.outnet_control, self.outnet_treated