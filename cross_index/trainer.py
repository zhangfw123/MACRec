import logging

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from collections import defaultdict
from utils import ensure_dir,set_color,get_local_time
import os

class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        # saved_model_dir = "{}".format(get_local_time())
        # self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_rq_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, data in enumerate(iter_data):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, indices = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_rq_loss += rq_loss

        return total_loss, total_recon_loss, total_rq_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        # indices_set = set()
        indices_list = []
        num_sample = 0
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices, distances = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                # indices_set.add(code)
                indices_list.append(code)
        
        freq_count = defaultdict(int)      
        for c in indices_list:
            freq_count[c] += 1
        max_value = max(list(freq_count.values()))
        min_value = min(list(freq_count.values()))

        indices_set = set(indices_list)
        collision_rate = (num_sample - len(indices_set))/num_sample

        return max_value, min_value, collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"


    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss, total_rq_loss = self._train_epoch(data, epoch_idx)
            print(f'epoch {epoch_idx}, total loss: {train_loss}, recon loss: {train_recon_loss}, rq loss: {total_rq_loss}')
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                # self._save_checkpoint(epoch=epoch_idx,ckpt_file=self.best_loss_ckpt)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                max_value, min_value, collision_rate = self._valid_epoch(data)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1

                print(f'MAX: {max_value}, Min: {min_value}, collision_rate: {collision_rate}, best_collision_rate: {self.best_collision_rate}')

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                
                # if epoch_idx > 1000:
                # if epoch_idx % 10 == 0:
                #     self._save_checkpoint(epoch_idx, collision_rate=collision_rate)


        return self.best_loss, self.best_collision_rate



class CrossTrainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_text_collision_rate = np.inf
        self.best_image_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_text_collision_ckpt = "best_text_collision_model.pth"
        self.best_image_collision_ckpt = "best_image_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_rq_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        for batch_idx, (text_data, image_data, index) in enumerate(iter_data):
            text_data = text_data.to(self.device)
            image_data = image_data.to(self.device)
            self.optimizer.zero_grad()
            text_out, image_out, text_rq_loss, image_rq_loss, text_indices, image_indices, text_distances, image_distances, share_out = self.model(text_data, image_data, item_index=index)
            loss, loss_recon = self.model.compute_loss(text_out, image_out, text_rq_loss, image_rq_loss, text_indices, image_indices, text_distances, image_distances, text_data, image_data, share_out)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_rq_loss += (text_rq_loss + image_rq_loss).item()

        return total_loss, total_recon_loss, total_rq_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        # indices_set = set()
        text_indices_list = []
        image_indices_list = []
        num_sample = 0
        for batch_idx, (text_data, image_data, index) in enumerate(iter_data):
            num_sample += len(text_data)
            text_data = text_data.to(self.device)
            image_data = image_data.to(self.device)
            text_indices, image_indices, text_distances, image_distances = self.model.get_indices(text_data, image_data)
            text_indices = text_indices.view(-1,text_indices.shape[-1]).cpu().numpy()
            image_indices = image_indices.view(-1,image_indices.shape[-1]).cpu().numpy()
            for text_index, image_index in zip(text_indices, image_indices):
                text_code = "-".join([str(int(_)) for _ in text_index])
                image_code = "-".join([str(int(_)) for _ in image_index])
                # indices_set.add(code)
                text_indices_list.append(text_code)
                image_indices_list.append(image_code)
        
        freq_count = defaultdict(int)      
        for c in text_indices_list:
            freq_count[c] += 1
        max_value = max(list(freq_count.values()))
        min_value = min(list(freq_count.values()))

        text_indices_set = set(text_indices_list)
        image_indices_set = set(image_indices_list)
        text_collision_rate = (num_sample - len(text_indices_set))/num_sample
        image_collision_rate = (num_sample - len(image_indices_set))/num_sample

        return max_value, min_value, text_collision_rate, image_collision_rate

    def _save_checkpoint(self, epoch, text_collision_rate=1, image_collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_text_collision_%.4f_image_collision_%.4f_model.pth' % (epoch, text_collision_rate, image_collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_text_collision_rate": self.best_text_collision_rate,
            "best_image_collision_rate": self.best_image_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"


    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss, total_rq_loss = self._train_epoch(data, epoch_idx)
            print(f'epoch {epoch_idx}, total loss: {train_loss}, recon loss: {train_recon_loss}, rq loss: {total_rq_loss}')
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                # self._save_checkpoint(epoch=epoch_idx,ckpt_file=self.best_loss_ckpt)

            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                max_value, min_value, text_collision_rate, image_collision_rate = self._valid_epoch(data)

                if text_collision_rate < self.best_text_collision_rate:
                    self.best_text_collision_rate = text_collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, text_collision_rate=text_collision_rate,
                                          image_collision_rate=image_collision_rate,
                                          ckpt_file=self.best_text_collision_ckpt)
                if image_collision_rate < self.best_image_collision_rate:
                    self.best_image_collision_rate = image_collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, text_collision_rate=text_collision_rate,
                                          image_collision_rate=image_collision_rate,
                                          ckpt_file=self.best_image_collision_ckpt)
                else:
                    cur_eval_step += 1

                print(f'MAX: {max_value}, Min: {min_value}, text_collision_rate: {text_collision_rate}, image_collision_rate: {image_collision_rate}, best_text_collision_rate: {self.best_text_collision_rate}, best_image_collision_rate: {self.best_image_collision_rate}')

                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("text_collision_rate", "blue")
                    + ": %f, "
                    + set_color("image_collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, text_collision_rate, image_collision_rate)

                self.logger.info(valid_score_output)
                
                # if epoch_idx > 1000:
                # if epoch_idx % 10 == 0:
                #     self._save_checkpoint(epoch_idx, collision_rate=collision_rate)


        return self.best_loss, self.best_text_collision_rate, self.best_image_collision_rate

