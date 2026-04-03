import copy
import math
import os
from typing import Any, List, Optional, Union

import json
import pandas as pd
import numpy as np
import torch
import torchsde
from pytorch_lightning import LightningDataModule, LightningModule
from torch.distributions import MultivariateNormal
from torchdyn.core import NeuralODE
from torchvision import transforms

from .components.augmentation import (
    AugmentationModule,
    AugmentedVectorField,
    Sequential,
)
from .components.distribution_distances import compute_distribution_distances
from .components.optimal_transport import OTPlanSampler
from .components.plotting import (
    plot_paths,
    plot_samples,
    plot_trajectory,
    store_trajectories,
)
from .components.schedule import ConstantNoiseScheduler, NoiseScheduler
from .components.solver import FlowSolver
from .components.ltae import LTAE2d
from .utils import get_wandb_logger

from .SR_metrics_numpy import Measure, minmax_normalize
import lpips
from .components.losses import gradient_loss, VGGContentLoss, temp_weighted_MSE_Loss
from torch.nn.functional import softmax


class CFMLitModule(LightningModule):
    """Conditional Flow Matching Module for training generative models and models over time."""

    def __init__(
        self,
        net: Any,
        optimizer: Any,
        datamodule: LightningDataModule,
        augmentations: AugmentationModule,
        partial_solver: FlowSolver,
        scheduler: Optional[Any] = None,
        neural_ode: Optional[Any] = None,
        ot_sampler: Optional[Union[str, Any]] = None,
        sigma_min: float = 0.1,
        mode: str = "SISR",
        avg_size: int = -1,
        leaveout_timepoint: int = -1,
        test_nfe: int = 100,
        plot: bool = False,
        nice_name: str = "CFM",
    ) -> None:
        """Initialize a conditional flow matching network either as a generative model or for a
        sequence of timepoints.

        Note: DDP does not currently work with NeuralODE objects from torchdyn
        in the init so we initialize them every time we need to do a sampling
        step.

        Args:
            net: torch module representing dx/dt = f(t, x) for t in [1, T] missing dimension.
            optimizer: partial torch.optimizer missing parameters.
            datamodule: datamodule object needs to have "dim", "IS_TRAJECTORY" properties.
            ot_sampler: ot_sampler specified as an object or string. If none then no OT is used in minibatch.
            sigma_min: sigma_min determines the width of the Gaussian smoothing of the data and interpolations.
            leaveout_timepoint: which (if any) timepoint to leave out during the training phase
            plot: if true, log intermediate plots during validation
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "net",
                "optimizer",
                "scheduler",
                "datamodule",
                "augmentations",
                "partial_solver",
            ],
            logger=False,
        )
        self.datamodule = datamodule
        self.is_trajectory = False
        if hasattr(datamodule, "IS_TRAJECTORY"):
            self.is_trajectory = datamodule.IS_TRAJECTORY
        # dims is either an integer or a tuple. This helps us to decide whether to process things as
        # a vector or as an image.
        if hasattr(datamodule, "dim"):
            self.dim = datamodule.dim
            self.is_image = False
        elif hasattr(datamodule, "dims"):
            self.dim = datamodule.dims
            self.is_image = True
        else:
            raise NotImplementedError("Datamodule must have either dim or dims")
        self.net = net(dim=self.dim)
        self.augmentations = augmentations
        self.aug_net = AugmentedVectorField(self.net, self.augmentations.regs, self.dim)
        self.val_augmentations = AugmentationModule(
            # cnf_estimator=None,
            l1_reg=1,
            l2_reg=1,
            squared_l2_reg=1,
        )
        self.val_aug_net = AugmentedVectorField(self.net, self.val_augmentations.regs, self.dim)
        if neural_ode is not None:
            self.aug_node = Sequential(
                self.augmentations.augmenter,
                neural_ode(self.aug_net),
            )

        self.partial_solver = partial_solver
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ot_sampler = ot_sampler
        if ot_sampler == "None":
            self.ot_sampler = None
        if isinstance(self.ot_sampler, str):
            # regularization taken for optimal Schrodinger bridge relationship
            self.ot_sampler = OTPlanSampler(method=ot_sampler, reg=2 * sigma_min**2)
        self.criterion = torch.nn.MSELoss()
        self.measure = Measure()
        self.results = pd.DataFrame()
        #self.lpips = lpips.LPIPS(net='alex')
        #self.lpips.eval()
        #for param in self.lpips.parameters():
        #    param.requires_grad = False
        self.minmax_normalize = minmax_normalize
        #if self.hparams.mode == "MISR":
        #    self.ltae = LTAE2d()
        self.gradient_loss = gradient_loss
        self.vggcontent_loss = VGGContentLoss()
        self.t_1_shift = 0.5
        self.dossr = False
        self.coupling = True
        self.sigma = 0.2


        self.Temp = 30
        self.temporal_weights = softmax(-torch.tensor([i for i in range(105)], device="cuda")/self.Temp)
        #self.temporal_weights /= self.temporal_weights.max()
        self.temp_weighted_mse_loss = temp_weighted_MSE_Loss
        self.brownian_bridge = False


    def forward_integrate(self, batch: Any, t_span: torch.Tensor):
        """Forward pass with integration over t_span intervals.

        (t, x, t_span) -> [x_t_span].
        """
        X = self.unpack_batch(batch)
        X_start = X[:, t_span[0], :]
        traj = self.node.trajectory(X_start, t_span=t_span)
        return traj

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        """Forward pass (t, x) -> dx/dt."""
        return self.net(t, x)

    def unpack_batch(self, batch):
        """Unpacks a batch of data to a single tensor."""
        if self.is_trajectory:
            return torch.stack(batch, dim=1)
        if not isinstance(self.dim, int):
            # Assume this is an image classification dataset where we need to strip the targets
            if not self.hparams.mode == "MISR":
                return batch #batch[0]
            else:
                batch_size, seq_len, c_in, heigth, width = batch['img_lr'].shape
                batch['img_lr'] = batch['img_lr'].view(batch_size*seq_len, c_in, heigth, width)
                #batch['img_lr_up'] = batch['img_lr_up'].view(batch_size*seq_len, c_in, heigth, width)
        return batch

    def preprocess_batch(self, X, training=False):
        """Converts a batch of data into matched a random pair of (x0, x1)"""
        t_select = torch.zeros(1, device=X['img_hr'].device)
        if self.is_trajectory:
            batch_size, times, dim = X.shape
            if not hasattr(self.datamodule, "HAS_JOINT_PLANS"):
                # resample the OT plan
                # list of length t of tuples of length 2 of tensors of shape
                tmp_ot_list = []
                for t in range(times - 1):
                    if training and t + 1 == self.hparams.leaveout_timepoint:
                        tmp_ot = torch.stack((X[:, t], X[:, t + 2]))
                    else:
                        tmp_ot = torch.stack((X[:, t], X[:, t + 1]))
                    if (
                        training
                        and self.ot_sampler is not None
                        and t != self.hparams.leaveout_timepoint
                    ):
                        tmp_ot = torch.stack(self.ot_sampler.sample_plan(tmp_ot[0], tmp_ot[1]))

                    tmp_ot_list.append(tmp_ot)
                tmp_ot_list = torch.stack(tmp_ot_list)
                # randomly sample a batch

            if training and self.hparams.leaveout_timepoint > 0:
                # Select random except for the leftout timepoint
                t_select = torch.randint(times - 2, size=(batch_size,), device=X.device)
                t_select[t_select >= self.hparams.leaveout_timepoint] += 1
            else:
                t_select = torch.randint(times - 1, size=(batch_size,))
            x0 = []
            x1 = []
            for i in range(batch_size):
                ti = t_select[i]
                ti_next = ti + 1
                if training and ti_next == self.hparams.leaveout_timepoint:
                    ti_next += 1
                if hasattr(self.datamodule, "HAS_JOINT_PLANS"):
                    x0.append(torch.tensor(self.datamodule.timepoint_data[ti][X[i, ti]]))
                    pi = self.datamodule.pi[ti]
                    if training and ti + 1 == self.hparams.leaveout_timepoint:
                        pi = self.datamodule.pi_leaveout[ti]
                    index_batch = X[i][ti]
                    i_next = np.random.choice(
                        pi.shape[1], p=pi[index_batch] / pi[index_batch].sum()
                    )
                    x1.append(torch.tensor(self.datamodule.timepoint_data[ti_next][i_next]))
                else:
                    x0.append(tmp_ot_list[ti][0][i])
                    x1.append(tmp_ot_list[ti][1][i])
            x0, x1 = torch.stack(x0), torch.stack(x1)
        else:
            #import ipdb; ipdb.set_trace()
            if not self.hparams.mode == "MISR":
                batch_size = X['img_hr'].shape[0]
                # If no trajectory assume generate from standard normal
                x1 = X['img_hr']
                if not self.coupling:
                    x0 = torch.randn_like(x1)
                else:
                    if self.brownian_bridge:
                        x0 = X['img_lr_up']
                    else:
                        std_value = 0.2 #*torch.rand(x1.shape[0], 1, 1, 1).cuda() #*torch.rand(1).item()
                        x0 = torch.normal(mean=0, std=self.sigma, size=x1.shape).cuda()
                        x0 += X['img_lr_up']
                img_lr = X['img_lr']
                #import ipdb; ipdb.set_trace()
            else:
                batch_size = X['img_hr'].shape[0]
                seq_len = X['img_lr'].shape[0]//batch_size
                # If no trajectory assume generate from standard normal
                x1 = torch.repeat_interleave(X['img_hr'], seq_len, dim=0)
                x0 = torch.normal(mean=0, std=0.1, size=X['img_hr'].shape) #torch.randn_like(X['img_hr'])
                x0 += X['img_lr_up'] ##################################################### data dependent coupling
                x0 = torch.repeat_interleave(x0, seq_len, dim=0)
                img_lr = X['img_lr']   
        return x0, x1, img_lr, t_select

    def average_ut(self, x, t, mu_t, sigma_t, ut):
        pt = torch.exp(-0.5 * (torch.cdist(x, mu_t) ** 2) / (sigma_t**2))
        batch_size = x.shape[0]
        ind = torch.randint(
            batch_size, size=(batch_size, self.hparams.avg_size - 1)
        )  # randomly (non-repreat) sample m-many index
        # always include self
        ind = torch.cat([ind, torch.arange(batch_size)[:, None]], dim=1)
        pt_sub = torch.stack([pt[i, ind[i]] for i in range(batch_size)])
        ut_sub = torch.stack([ut[ind[i]] for i in range(batch_size)])
        p_sum = torch.sum(pt_sub, dim=1, keepdim=True)
        ut = torch.sum(pt_sub[:, :, None] * ut_sub, dim=1) / p_sum
        # Reduce batch size because they are all the same
        return x[:1], ut[:1], t[:1]

    def calc_mu_sigma(self, x0, x1, t):
        mu_t = t * x1 + (1 - t) * x0
        sigma_t = self.hparams.sigma_min
        return mu_t, sigma_t

    def calc_u(self, x0, x1, x, t, mu_t, sigma_t):
        del x, t, mu_t, sigma_t
        return x1 - x0
    

    def calc_u_brownian(self, x0, x1, x, t, mu_t, sigma_t, eps_t):
        del x, mu_t
        return x1 - x0 +  eps_t * (2*t-1) * sigma_t * 4


    def calc_loc_and_target(self, x0, x1, t, t_select, training, batch_size=64):
        """Computes the loss on a batch of data."""
        
        t_xshape = t.reshape(-1, *([1] * (x0.dim() - 1)))
        mu_t, sigma_t = self.calc_mu_sigma(x0, x1, t_xshape)
        eps_t = torch.randn_like(mu_t)
        if self.brownian_bridge:
            x = mu_t + 4*sigma_t * eps_t * t_xshape*(t_xshape-1)
            ut = self.calc_u_brownian(x0, x1, x, t_xshape, mu_t, sigma_t, eps_t)
        else:
            x = mu_t + sigma_t * eps_t
            ut = self.calc_u(x0, x1, x, t_xshape, mu_t, sigma_t)
        # if we are starting from right before the leaveout_timepoint then we
        # divide the target by 2
        if training and self.hparams.leaveout_timepoint > 0:
            ut[t_select + 1 == self.hparams.leaveout_timepoint] /= 2
            t[t_select + 1 == self.hparams.leaveout_timepoint] *= 2

        # p is the pair-wise conditional probability matrix. Note that this has to be torch.cdist(x, mu) in that order
        # t that network sees is incremented by first timepoint
        t = t + t_select.reshape(-1, *t.shape[1:])
        return x, ut, t, mu_t, sigma_t, eps_t
    
    def step(self, batch: Any, training: bool = False):
        """Computes the loss on a batch of data."""
        batch_size = batch['img_hr'].shape[0]
        X = self.unpack_batch(batch)
        
        x0, x1, img_lr, t_select = self.preprocess_batch(X, training)
        #import ipdb; ipdb.set_trace()
        # Either randomly sample a single T or sample a batch of T's
        if self.hparams.avg_size > 0:
            t = torch.rand(1).repeat(X['img_hr'].shape[0]).type_as(X['img_hr'])
        else:
            if not self.hparams.mode == "MISR":
                t = torch.rand(X['img_hr'].shape[0]).type_as(X['img_hr'])
            else:
                seq_len = batch['img_lr'].shape[0]//batch_size
                t = torch.rand(X['img_hr'].shape[0])
                t = torch.repeat_interleave(t, seq_len, dim=0).type_as(X['img_lr'])
        # Resample the plan if we are using optimal transport
        if self.ot_sampler is not None and not self.is_trajectory:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        if not self.dossr:
            x, ut, t, mu_t, sigma_t, eps_t = self.calc_loc_and_target(x0, x1, t, t_select, training, batch_size)
        else: 
            x, ut, t, mu_t, sigma_t, eps_t = self.calc_loc_and_target_domain_shift(x0, x1, t, t_select, training, batch_size, x_lr_up=X['img_lr_up'])
        if self.hparams.avg_size > 0:
            x, ut, t = self.average_ut(x, t, mu_t, sigma_t, ut)
        if not self.hparams.mode == "MISR":
            aug_x = self.aug_net(t, x, shape=X['img_hr'].shape[0], dates=None, low_res=img_lr, MISR=(self.hparams.mode=="MISR"), context=batch['xy'], augmented_input=False)
        else:    
            aug_x = self.aug_net(t, x, shape=X['img_hr'].shape[0], dates=batch['dates_encoding'], low_res=img_lr, MISR=(self.hparams.mode=="MISR"), augmented_input=False)
        reg, vt = self.augmentations(aug_x)
        if not self.hparams.mode == "MISR":
            if self.hparams.sigma_min == 0:

                tstamp=t.view(-1, 1, 1, 1)
                #pred = (1-tstamp)*vt+(1-tstamp)*x0+tstamp*x1 # x1=vt(1-t)+xt
                #import ipdb; ipdb.set_trace()
                #weights = self.temporal_weights[batch['delta_t']]
                """
                weights = softmax(-batch['delta_t'].float()/50)
                #print(weights/weights.sum()
                reg =  (weights*self.vggcontent_loss(pred[:,:3,...], x1[:,:3,...])).sum()
                loss = self.temp_weighted_mse_loss(vt, ut, weights) + reg # weighted loss"""

                #reg =  self.vggcontent_loss(pred[:,:3,...], x1[:,:3,...]).mean()
                reg=torch.tensor(0.0).cuda()
                loss = self.criterion(vt, ut) + reg
                #loss = self.criterion(vt, ut) + reg.mean() #+self.lpips(pred, x1).mean()#+self.gradient_loss(pred, x1) 
                #loss = self.criterion(vt, ut)#+lpips_loss(pred, x1).mean()# + self.vggcontent_loss(pred, x1)
                #print(self.lpips(pred, x1).mean())
                #print(loss)
                return torch.mean(reg), loss
            return torch.mean(reg), self.criterion(vt, ut) #+(1-self.hparams.sigma_min)*self.lpips(pred, x1)
        else:
            # vt: B*TxCxHxW to BxTxCxHxW
            #vt = vt.view((batch_size, seq_len, vt.shape[-3], vt.shape[-2], vt.shape[-1]))
            # apply ltae
            #fused_vt = self.ltae(vt, batch['dates_encoding'])
            #import ipdb; ipdb.set_trace()
            ut = ut[[seq_len*i for i in range(batch_size)],...]
            return  torch.mean(reg), self.criterion(vt, ut)

            
    def training_step(self, batch: Any, batch_idx: int):
        #import ipdb; ipdb.set_trace()
        #print(torch.backends.cudnn.version())
        torch.cuda.empty_cache()
        reg, mse = self.step(batch, training=True)
        loss = mse + reg
        prefix = "train"
        self.log_dict(
            {f"{prefix}/loss": loss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

    def image_eval_step(self, batch: Any, batch_idx: int, prefix: str):
        import os
        from math import prod

        from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
        from torchvision.utils import save_image

        solver = self.partial_solver(self.net, self.dim)
        if isinstance(self.hparams.test_nfe, int):
            t_span = torch.linspace(0, 1, int(self.hparams.test_nfe) + 1)
        elif isinstance(self.hparams.test_nfe, str):
            solver.ode_solver = "tsit5"
            t_span = torch.linspace(0, 1, 100)
        else:
            raise NotImplementedError(f"Unknown test procedure {self.hparams.test_nfe}")
        #t_span=torch.linspace(0, 1, 2)
        #import ipdb; ipdb.set_trace()
        if not self.hparams.mode == "MISR":
            if not self.coupling:
                traj = solver.odeint(torch.randn(batch["img_hr"].shape[0], *self.dim).type_as(batch["img_hr"]), t_span, shape=batch['img_hr'].shape[0], dates=None, low_res=batch['img_lr'], MISR=(self.hparams.mode=="MISR"), context=batch['xy'])
            else:
                if self.coupling:
                    if self.brownian_bridge:
                        traj = solver.odeint(batch["img_lr_up"].type_as(batch["img_hr"]), t_span, shape=batch['img_hr'].shape[0], dates=None, low_res=batch['img_lr'], MISR=(self.hparams.mode=="MISR"), context=batch['xy'])
                    else:
                        std_value = 0.2#*torch.rand(batch["img_hr"].shape[0], 1, 1, 1).cuda() #*torch.rand(1).item()
                        rand =  torch.normal(mean=0, std=self.sigma, size=batch["img_hr"].shape).cuda() 
                        traj = solver.odeint((batch["img_lr_up"]+rand).type_as(batch["img_hr"]), t_span, shape=batch['img_hr'].shape[0], dates=None, low_res=batch['img_lr'], MISR=(self.hparams.mode=="MISR"), context=batch['xy'])
                else:
                    traj = solver.odeint((batch["img_lr_up"]).type_as(batch["img_hr"]), t_span, shape=batch['img_hr'].shape[0], dates=None, low_res=batch['img_lr'], MISR=(self.hparams.mode=="MISR"))

        else:
            traj = solver.odeint(torch.randn(batch["img_hr"].shape[0], *self.dim).type_as(batch["img_hr"]), t_span, shape=batch['img_hr'].shape[0], dates=batch['dates_encoding'], low_res=batch['img_lr'], MISR=(self.hparams.mode=="MISR"))
        traj = traj[-1]
        
        os.makedirs("images", exist_ok=True)

        traj = torch.clip(traj, min=-1.0, max=1.0) 
        
        if prefix == "lol":
            sr_metrics = self.measure.measure(traj, batch['img_hr'], batch['img_lr'], 4)
            sr_metrics['indexes'] = batch['indexes'].cpu().numpy()
            df_sr_metrics = pd.DataFrame(sr_metrics)
        df_sr_metrics = pd.DataFrame()

        traj = (traj+1)/2
        """
        for i, image in enumerate(traj):
            save_image(image, fp=f"images/{batch_idx}_{i}.png")
        for i, image in enumerate(batch['img_hr']):
            save_image(torch.clip((batch['img_hr'][i]+1)/2, min=0, max=1.0), fp=f"images/{batch_idx}_{i}_img_hr.png")
            save_image(torch.clip((batch['img_lr'][i]+1)/2, min=0, max=1.0), fp=f"images/{batch_idx}_{i}_img_lr.png")"""
        
        id_patch = batch['ID_PATCH'][0].item()
        dates_s2 = json.loads(batch['dates-S2'][0])
        keys = list(dates_s2.keys())
        for i, image in enumerate(traj):
            save_image(image, fp=f"images/{id_patch}_{batch['item_name'][0]}_{dates_s2[keys[i]]}.png")
        return {"x": batch['img_hr']}, df_sr_metrics

    def eval_step(self, batch: Any, batch_idx: int, prefix: str):
        if prefix == "test" and self.is_image:

            _ , df_sr_metrics = self.image_eval_step(batch, batch_idx, prefix)
            self.results = pd.concat([self.results, df_sr_metrics], axis=0)

        shapes = [b.shape[0] for b in batch['img_hr']]

        if not self.is_image and prefix == "val" and shapes.count(shapes[0]) == len(shapes):
            reg, mse = self.step(batch, training=False)
            loss = mse + reg
            self.log_dict(
                {f"{prefix}/loss": loss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            return {"loss": loss, "mse": mse, "reg": reg, "x": self.unpack_batch(batch)}
        if self.is_image and prefix == "val" and shapes.count(shapes[0]) == len(shapes):
            reg, mse = self.step(batch, training=False)
            loss = mse + reg
            self.log_dict(
                {f"{prefix}/loss": loss, f"{prefix}/mse": mse, f"{prefix}/reg": reg},
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            if self.hparams.mode=="MISR":
                batch_size = batch['img_hr'].shape[0]
                seq_len = batch['img_lr'].shape[0]//batch_size
                batch['img_lr'] = batch['img_lr'].view((batch_size, seq_len) + batch['img_lr'].shape[1:])
            return {"loss": loss, "mse": mse, "reg": reg, "x": self.unpack_batch(batch)}

        return {"x": batch}

    def preprocess_epoch_end(self, outputs: List[Any], prefix: str):
        """Preprocess the outputs of the epoch end function."""
        if self.is_trajectory and prefix == "test" and isinstance(outputs[0]["x"], list):
            # x is jagged if doing a trajectory
            x = outputs[0]["x"]
            ts = len(x)
            x0 = x[0]
            x_rest = x[1:]
        elif self.is_trajectory:
            if hasattr(self.datamodule, "HAS_JOINT_PLANS"):
                x = [torch.tensor(dd) for dd in self.datamodule.timepoint_data]
                x0 = x[0]
                x_rest = x[1:]
                ts = len(x)
            else:
                v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
                x = v["x"]
                ts = x.shape[1]
                x0 = x[:, 0, :]
                x_rest = x[:, 1:]
        else:
            if isinstance(self.dim, int):
                v = {k: torch.cat([d[k] for d in outputs]) for k in ["x"]}
                x = v["x"]
            else:
                #import ipdb; ipdb.set_trace()
                x = [d["x"]['img_hr'] for d in outputs][0]#[0][:100]
            # Sample some random points for the plotting function
            #import ipdb; ipdb.set_trace()
            if self.hparams.mode =="MISR":
                rand = torch.randn_like(x)
                # rand = torch.randn_like(x, generator=torch.Generator(device=x.device).manual_seed(42))
                x = torch.stack([rand, x], dim=1)
                ts = x.shape[1]
                x0 = x[:, 0]
                seq_len= outputs[0]['x']['img_lr'].shape[0]//x.shape[0]
                #c_in, height, width = x.shape[-3], x.shape[-2], x.shape[-1]
                #x = x.view((x.shape[0]*seq_len, c_in, height, width))
                x0 = torch.repeat_interleave(x0, seq_len, dim=0)

                x_rest = x[:, 1:]
                
                return ts, x, x0, x_rest, outputs[0]['x']['dates_encoding']
            if not self.coupling:
                rand = torch.randn_like(x) 
            else:
                if self.brownian_bridge:
                    rand = outputs[0]['x']['img_lr_up']
                else:
                    std_value = 0.2#*torch.rand(x.shape[0], 1, 1, 1).cuda() #*torch.rand(1).item()
                    rand = torch.normal(mean=0, std=self.sigma, size=x.shape).cuda() 
                    rand = outputs[0]['x']['img_lr_up'] +rand ################## data dependent coupling

            # rand = torch.randn_like(x, generator=torch.Generator(device=x.device).manual_seed(42))
            x = torch.stack([rand, x], dim=1)
            ts = x.shape[1]
            x0 = x[:, 0]
            x_rest = x[:, 1:]
        return ts, x, x0, x_rest
        #x==outputs[0]['x']['img_hr'][0]

    def forward_eval_integrate(self, ts, x0, x_rest, outputs, prefix, dates=None):
        # Build a trajectory
        t_span = torch.linspace(0, 1, self.hparams.test_nfe)
        aug_dims = self.val_augmentations.aug_dims
        regs = []
        trajs = []
        full_trajs = []
        solver = self.partial_solver(self.net, self.dim)
        nfe = 0
        x0_tmp = x0.clone()
        low_res = outputs[0]["x"]['img_lr']

        if self.is_image:
            traj = solver.odeint(x0, t_span, x0.shape[0], dates=dates, low_res=low_res, MISR=(self.hparams.mode=="MISR"), context=outputs[0]['x']['xy']) 
            full_trajs.append(traj)
            trajs.append(traj[0])
            trajs.append(traj[-1])
            nfe += solver.nfe

        if not self.is_image:
            solver.augmentations = self.val_augmentations
            for i in range(ts - 1):
                traj, aug = solver.odeint(x0_tmp, t_span + i)
                full_trajs.append(traj)
                traj, aug = traj[-1], aug[-1]
                x0_tmp = traj
                regs.append(torch.mean(aug, dim=0).detach().cpu().numpy())
                trajs.append(traj)
                nfe += solver.nfe

        full_trajs = torch.cat(full_trajs)

        if not self.is_image:
            regs = np.stack(regs).mean(axis=0)
            names = [f"{prefix}/{name}" for name in self.val_augmentations.names]
            self.log_dict(dict(zip(names, regs)), sync_dist=True)

            # Evaluate the fit
            if (
                self.is_trajectory
                and prefix == "test"
                and isinstance(outputs[0]["x"], list)
                and not hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM")
            ):
                # Redo the solver for each timepoint
                trajs = []
                full_trajs = []
                nfe = 0
                x0_tmp = x0
                for i in range(ts - 1):
                    traj, _ = solver.odeint(x0_tmp, t_span + i)
                    traj = traj[-1]
                    x0_tmp = x_rest[i]
                    trajs.append(traj)
                    nfe += solver.nfe
                names, dists = compute_distribution_distances(trajs[:-1], x_rest[:-1])
            else:
                names, dists = compute_distribution_distances(trajs, x_rest)
            names = [f"{prefix}/{name}" for name in names]
            d = dict(zip(names, dists))
            if self.hparams.leaveout_timepoint >= 0:
                to_add = {
                    f"{prefix}/t_out/{key.split('/')[-1]}": val
                    for key, val in d.items()
                    if key.startswith(f"{prefix}/t{self.hparams.leaveout_timepoint}")
                }
                d.update(to_add)
            d[f"{prefix}/nfe"] = nfe

            self.log_dict(d, sync_dist=True)

        if hasattr(self.datamodule, "GAUSSIAN_CLOSED_FORM"):
            solver.augmentations = None
            # t_span = torch.linspace(0, 1, 101)
            # traj = solver.odeint(x0, t_span)
            # t_span = t_span[::5]
            # traj = traj[::5]
            t_span = torch.linspace(0, 1, 21)
            traj = solver.odeint(x0, t_span)
            assert traj.shape[0] == t_span.shape[0]
            kls = [
                self.datamodule.KL(xt, self.hparams.sigma_min, t) for t, xt in zip(t_span, traj)
            ]
            self.log_dict({f"{prefix}/kl/mean": torch.stack(kls).mean().item()}, sync_dist=True)
            self.log_dict({f"{prefix}/kl/tp_{i}": kls[i] for i in range(21)}, sync_dist=True)

        return trajs, full_trajs

    def eval_epoch_end(self, outputs: List[Any], prefix: str):
        wandb_logger = get_wandb_logger(self.loggers)
        """
        if prefix == "test" and self.is_image:
            os.makedirs("images", exist_ok=True)
            if len(os.listdir("images")) > 0:
                path = "/home/okabayas/Documents/breizhsr/BreizhSR/dataset_train/SISR" #"/home/mila/a/alexander.tong/scratch/trajectory-inference/data/fid_stats_cifar10_train.npz"
                from pytorch_fid import fid_score

                fid = fid_score.calculate_fid_given_paths(["images", path], 256, "cuda", 2048, 0)
                self.log(f"{prefix}/fid", fid)"""

        if not self.hparams.mode=="MISR":
            ts, x, x0, x_rest = self.preprocess_epoch_end(outputs, prefix)
            trajs, full_trajs = self.forward_eval_integrate(ts, x0, x_rest, outputs, prefix)
        else:
            ts, x, x0, x_rest, dates = self.preprocess_epoch_end(outputs, prefix)
            trajs, full_trajs = self.forward_eval_integrate(ts, x0, x_rest, outputs, prefix, dates)
        

        if self.hparams.plot:
            if isinstance(self.dim, int):
                plot_trajectory(
                    x,
                    full_trajs,
                    title=f"{self.current_epoch}_ode",
                    key="ode_path",
                    wandb_logger=wandb_logger,
                )
            else:
                plot_samples(
                    trajs[-1],
                    title=f"{self.current_epoch}_samples",
                    wandb_logger=wandb_logger,
                )
                #import ipdb; ipdb.set_trace()
                sr_metrics = self.measure.measure(trajs[-1], x_rest.squeeze(), x0, 4)
                df_sr_metrics = pd.DataFrame(sr_metrics)
                df_sr_metrics.to_pickle(f"figs/{self.current_epoch}_metrics.pkl")

        if prefix == "test" and not self.is_image:
            store_trajectories(x, self.net)
        if prefix == "test":
            self.results.to_pickle(f"test_results.pkl")

    def validation_step(self, batch: Any, batch_idx: int):
        torch.cuda.empty_cache() 
        #import ipdb; ipdb.set_trace()
        return self.eval_step(batch, batch_idx, "val")

    def validation_epoch_end(self, outputs: List[Any]):
        #import ipdb; ipdb.set_trace()
        self.eval_epoch_end(outputs, "val")

    def test_step(self, batch: Any, batch_idx: int):
        B, T, C, H, W = batch['img_lr'].shape
        batch['img_lr_up'] = batch['img_lr_up'].view(B*T, C, H*4, W*4)
        batch['img_lr'] = batch['img_lr'].view(B*T, C, H, W)
        return self.eval_step(batch, batch_idx, "test")

    def test_epoch_end(self, outputs: List[Any]):
        self.eval_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """Pass model parameters to optimizer."""
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is None:
            return optimizer

        scheduler = self.scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(epoch=self.current_epoch)


