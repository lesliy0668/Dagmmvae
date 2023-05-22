import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
from torch.autograd import grad
from torch.autograd import Variable
from model import *
import matplotlib.pyplot as plt
from utils import *
from data_loader import *
import IPython
from tqdm import tqdm
from scipy.spatial import distance_matrix
from mpl_toolkits.mplot3d import Axes3D

class Solver(object):
    DEFAULTS = {}   
    def __init__(self, data_loader, config,input_dim=118):
        # Data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.data_loader = data_loader
        self.input_dim = input_dim
        # Build tensorboard if use
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()
    
    """ 自适应确定K """
    def autoK(self):
        data,labels = load_data(self.data_path)
        features = data
        N, D = features.shape
        features = Normalize(features)
        normal_data = features[labels==0]
        t0 = time.time()
        #dises = distance_matrix(normal_data,normal_data)
        
        print("mean shift to get K....")

        K = mean_shift(normal_data)
        print("done!,total time:{},K:{}".format(time.time()-t0,K))
        return

    def build_model(self):
        # Define model
        if self.backbone == "DAE":
            self.dagmm = DaGMM(self.gmm_k,input_dim=self.input_dim)
            name = "DaGMM"
        elif self.backbone == "VAE":
            self.dagmm = DaGMMVAE(self.gmm_k,input_dim=self.input_dim)
            name = "DaGMMVAE"
            self.AWL = AutomaticWeightedLoss(2)
            self.para = (self.AWL.params.cpu().detach().numpy()).tolist()
            self.optimizerAWL = torch.optim.Adam(self.AWL.parameters(), lr=self.lr)
            
        # Optimizers
        self.optimizer = torch.optim.Adam(self.dagmm.parameters(), lr=self.lr)
        
        # Print networks
        self.print_network(self.dagmm, name)

        if torch.cuda.is_available():
            self.dagmm.cuda()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        self.dagmm.load_state_dict(torch.load(self.pretrained_model))

        print("phi", self.dagmm.phi,"mu",self.dagmm.mu, "cov",self.dagmm.cov)

        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def reset_grad(self):
        self.dagmm.zero_grad()

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def train(self,print_interval=20,save_interval=10):
        fbest = 0
        iters_per_epoch = len(self.data_loader)

        # Start with trained model if exists
        # if self.pretrained_model:
        #     start = int(self.pretrained_model.split('_')[0])
        # else:
        #     start = 0
        start = 0
        # Start training
        iter_ctr = 0
        start_time = time.time()
        self.test()

        #print("len:",len(self.data_loader))
        self.ap_global_train = np.array([0,0,0])
        for e in range(start, self.num_epochs):
            self.data_loader.dataset.mode="train"
            for i, (input_data, labels) in enumerate(tqdm(self.data_loader)):
                #print(input_data.shape)
                #exit()
                iter_ctr += 1
                start = time.time()

                input_data = self.to_var(input_data)

                total_loss,sample_energy, recon_error, cov_diag = self.dagmm_step(input_data)
                # Logging
                loss = {}
                loss['total_loss'] = total_loss.data.item()
                loss['sample_energy'] = sample_energy.item()
                loss['recon_error'] = recon_error.item()
                loss['cov_diag'] = cov_diag.item()



                # Print out log info
                if (i+1) % self.log_step == 0:
                    elapsed = time.time() - start_time
                    total_time = ((self.num_epochs*iters_per_epoch)-(e*iters_per_epoch+i)) * elapsed/(e*iters_per_epoch+i+1)
                    epoch_time = (iters_per_epoch-i)* elapsed/(e*iters_per_epoch+i+1)
                    
                    epoch_time = str(datetime.timedelta(seconds=epoch_time))
                    total_time = str(datetime.timedelta(seconds=total_time))
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    lr_tmp = []
                    for param_group in self.optimizer.param_groups:
                        lr_tmp.append(param_group['lr'])
                    tmplr = np.squeeze(np.array(lr_tmp))

                    log = "Elapsed {}/{} -- {} , Epoch [{}/{}], Iter [{}/{}], lr {}".format(
                        elapsed,epoch_time,total_time, e+1, self.num_epochs, i+1, iters_per_epoch, tmplr)

                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    if self.backbone == "VAE":
                        log+=" -- {}:{}".format("AWL",self.para)
                    IPython.display.clear_output()
                    
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.logger.scalar_summary(tag, value, e * iters_per_epoch + i + 1)
                    else:
                        plt_ctr = 1
                        if not hasattr(self,"loss_logs"):
                            self.loss_logs = {}
                            for loss_key in loss:
                                self.loss_logs[loss_key] = [loss[loss_key]]
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1
                        else:
                            for loss_key in loss:
                                self.loss_logs[loss_key].append(loss[loss_key])
                                plt.subplot(2,2,plt_ctr)
                                plt.plot(np.array(self.loss_logs[loss_key]), label=loss_key)
                                plt.legend()
                                plt_ctr += 1

                        plt.show()

                    #print("phi", self.dagmm.phi,"mu",self.dagmm.mu, "cov",self.dagmm.cov)
                # Save model checkpoints
                # if (i+1) % self.model_save_step == 0:
                #     torch.save(self.dagmm.state_dict(),
                #         os.path.join(self.model_save_path, '{}_{}_dagmm.pth'.format(e+1, i+1)))
            if e%save_interval==0 and e>0:
                torch.save(self.dagmm.state_dict(), os.path.join(self.model_save_path, '{}_{}_dagmm.pth'.format(e+1, i+1)))
            
            accuracy, precision, recall, f_score = self.test()
            self.dagmm.train()
            if f_score>fbest:
                torch.save(self.dagmm.state_dict(), os.path.join(self.model_save_path, '{}_best_dagmm.pth'.format(f_score)))
                fbest = f_score

    def dagmm_step(self, input_data):
        self.dagmm.train()
        if self.backbone=="VAE":
            enc, dec, z, gamma,mean,std = self.dagmm(input_data)
            total_loss, sample_energy, recon_error, cov_diag,klloss = self.dagmm.loss_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag,mean,std)
            total_loss = recon_error+self.lambda_energy * sample_energy+1e-3*self.AWL(cov_diag,klloss)
        else:
            enc, dec, z, gamma = self.dagmm(input_data)
            total_loss, sample_energy, recon_error, cov_diag = self.dagmm.loss_function(input_data, dec, z, gamma, self.lambda_energy, self.lambda_cov_diag)

        self.reset_grad()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.dagmm.parameters(), 5)
        self.optimizer.step()
        if self.backbone=="VAE":
            torch.nn.utils.clip_grad_norm_(self.AWL.parameters(), 1)
            self.optimizerAWL.step()
            
            for p in self.AWL.parameters():
                p.data.clamp_(0, 1)
            self.para = (self.AWL.params.cpu().detach().numpy()).tolist()
            return total_loss,sample_energy, recon_error, cov_diag
        return total_loss,sample_energy, recon_error, cov_diag

    def test(self,isplot=0):
        print("======================TEST MODE======================")
        self.dagmm.eval()
        self.data_loader.dataset.mode="train"

        N = 0
        mu_sum = 0
        cov_sum = 0
        gamma_sum = 0
        # if test_loader is None:
        #     test_loader = self.data_loader
        bar = tqdm(self.data_loader,"test",len(self.data_loader))
        with torch.no_grad():
            for it, (input_data, labels) in enumerate(bar):
                input_data = self.to_var(input_data)
                if self.backbone=="VAE":
                    enc, dec, z, gamma,mean,std = self.dagmm(input_data)
                else:
                    enc, dec, z, gamma = self.dagmm(input_data)
                phi, mu, cov = self.dagmm.compute_gmm_params(z, gamma)
                
                batch_gamma_sum = torch.sum(gamma, dim=0)
                
                gamma_sum += batch_gamma_sum
                mu_sum += mu * batch_gamma_sum.unsqueeze(-1) # keep sums of the numerator only
                cov_sum += cov * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1) # keep sums of the numerator only
                
                N += input_data.size(0)
                
            train_phi = gamma_sum / N
            train_mu = mu_sum / gamma_sum.unsqueeze(-1)
            train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

            # print("N:",N)
            # print("phi :\n",train_phi)
            # print("mu :\n",train_mu)
            # print("cov :\n",train_cov)

            train_energy = []
            train_labels = []
            train_z = []
            for it, (input_data, labels) in enumerate(self.data_loader):
                input_data = self.to_var(input_data)
                if self.backbone=="VAE":
                    enc, dec, z, gamma,mean,std = self.dagmm(input_data)
                else:
                    enc, dec, z, gamma = self.dagmm(input_data)
                sample_energy, cov_diag = self.dagmm.compute_energy(z, phi=train_phi, mu=train_mu, cov=train_cov, size_average=False)
                
                train_energy.append(sample_energy.data.cpu().numpy())
                train_z.append(z.data.cpu().numpy())
                train_labels.append(labels.numpy())


            train_energy = np.concatenate(train_energy,axis=0)
            train_z = np.concatenate(train_z,axis=0)
            train_labels = np.concatenate(train_labels,axis=0)


            self.data_loader.dataset.mode="test"
            test_energy = []
            test_labels = []
            test_z = []
            recloss = []
            cosines = []
            bar = tqdm(self.data_loader,"test",len(self.data_loader))
            for it, (input_data, labels) in enumerate(bar):
                input_data = self.to_var(input_data)
                if self.backbone=="VAE":
                    enc, dec, z, gamma,mean,std = self.dagmm(input_data)
                else:
                    enc, dec, z, gamma = self.dagmm(input_data)
                sample_energy, cov_diag = self.dagmm.compute_energy(z, size_average=False)
                test_energy.append(sample_energy.data.cpu().numpy())
                test_z.append(z.data.cpu().numpy())
                test_labels.append(labels.numpy())

        test_energy = np.concatenate(test_energy,axis=0)
        test_z = np.concatenate(test_z,axis=0)
        test_labels = np.concatenate(test_labels,axis=0)
        print("shape:",test_z.shape,test_labels.shape)
        
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels], axis=0)

        thresh = np.percentile(combined_energy, 100 - 20)
        print("Threshold :", thresh)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score

        accuracy = accuracy_score(gt,pred)
        precision, recall, f_score, support = prf(gt, pred, average='binary')

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall, f_score))
        if isplot:
            x = test_z[:,0]
            y = test_z[:,1]
            z = test_z[:,2]
            label = test_labels[:]
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(x[label==0],y[label==0],z[label==0],c="b",label="Normal")
            ax.scatter(x[label==1],y[label==1],z[label==1],c="r",label="Abnormal")
            ax.set_zlabel('Encoded', fontdict={'size': 15, 'color': 'red'})
            ax.set_ylabel('Euclidean', fontdict={'size': 15, 'color': 'red'})
            ax.set_xlabel('Cosine', fontdict={'size': 15, 'color': 'red'})
            plt.legend()
            #plt.show()
            plt.savefig(self.log_path+"/pic1.jpg")
            fig = plt.figure()
            ids = np.argsort(test_labels)
            test_energy = test_energy[ids]
            x = np.arange(0,test_energy.shape[0])
            plt.plot(x,test_energy,"o-")
            plt.xlabel("samples")
            plt.ylabel("energy")
            plt.savefig(self.log_path+"/pic2.jpg")

        return accuracy, precision, recall, f_score