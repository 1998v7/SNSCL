import torch
import torch.nn as nn
import numpy as np

class Aug_Module(nn.Module):
    def __init__(self, input_size):
        super(Aug_Module, self).__init__()
        self.fc_mu = nn.Linear(input_size, input_size)
        self.fc_var = nn.Linear(input_size, input_size)

    def sampler(self, feature):
        mu = self.fc_mu(feature)
        log_var = self.fc_var(feature)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        new_feat = eps + std * mu

        kld_loss = - torch.mean(0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return new_feat, kld_loss


class SNSCL(nn.Module):
    def __init__(self, network, backbone, queue_size=32, projector_dim=256, feature_dim=256,
                 class_num=200, momentum=0.999, temp=0.07, pretrained=True, pretrained_path=None):

        super(SNSCL, self).__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.class_num = class_num
        self.backbone = backbone
        self.pretrained = pretrained
        self.temp = temp
        self.pretrained_path = pretrained_path

        # create the encoders
        self.encoder_q = network(projector_dim=projector_dim)
        self.encoder_k = network(projector_dim=projector_dim)
        self.AugModule = Aug_Module(input_size=self.encoder_q.feature_len())

        self.load_pretrained(network)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # don't be updated by gradient

        # create the queue
        self.register_buffer("queue_list", torch.randn(projector_dim, queue_size * self.class_num))
        self.queue_list = nn.functional.normalize(self.queue_list, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(self.class_num, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, key_c, c, w_u=None):
        # gather keys before updating queue
        batch_size = key_c.shape[0]
        ptr = int(self.queue_ptr[c])
        real_ptr = ptr + c * self.queue_size
        # replace the keys at ptr (dequeue and enqueue)
        self.queue_list[:, real_ptr:real_ptr + batch_size] = key_c.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        # selectively update the queue
        if w_u is None:
            self.queue_ptr[c] = ptr
        else:
            if np.random.uniform(0,1) < w_u:
                self.queue_ptr[c] = ptr
            else:
                return

    def load_state_from_q(self):
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())

    def forward(self, im_q, im_k, labels, w_u=None):
        q_c, q_f = self.encoder_q(im_q)
        q_c = nn.functional.normalize(q_c, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k_c, k_f = self.encoder_k(im_k)  # keys: k_c (N x projector_dim)
            k_c = nn.functional.normalize(k_c, dim=1)

        # feature transformation =====================
        q_f_aug, kl_loss1 = self.AugModule.sampler(q_f)
        q_c_aug = self.encoder_q.make_projector(q_f_aug)
        q_c_aug = nn.functional.normalize(q_c_aug, dim=1)

        k_f_aug, kl_loss2 = self.AugModule.sampler(k_f)
        with torch.no_grad():  # no gradient to keys
            k_c_aug = self.encoder_k.make_projector(k_f_aug)
            k_c_aug = nn.functional.normalize(k_c_aug, dim=1)

        q_c = torch.cat([q_c, q_c_aug], dim=0)
        k_c = torch.cat([k_c, k_c_aug], dim=0)
        labels = torch.cat([labels, labels], dim=0)
        if w_u is not None:
            w_u = torch.cat([w_u, w_u], dim=0)

        batch_size = q_c.size()[0]

        l_pos = torch.einsum('nl,nl->n', q_c, k_c).unsqueeze(-1)  # Einstein sum is more intuitive

        cur_queue_list = self.queue_list.clone().detach()
        l_neg_list = torch.Tensor([]).cuda()
        l_pos_list = torch.Tensor([]).cuda()

        for i in range(batch_size):
            neg_sample = torch.cat([cur_queue_list[:, 0:labels[i]*self.queue_size], cur_queue_list[:, (labels[i]+1)*self.queue_size:]], dim=1)
            pos_sample = cur_queue_list[:, labels[i]*self.queue_size: (labels[i]+1) * self.queue_size]
            ith_neg = torch.einsum('nl,lk->nk', q_c[i:i+1], neg_sample)
            ith_pos = torch.einsum('nl,lk->nk', q_c[i:i+1], pos_sample)
            l_neg_list = torch.cat((l_neg_list, ith_neg), dim = 0)
            l_pos_list = torch.cat((l_pos_list, ith_pos), dim = 0)
            if w_u is None:
                self._dequeue_and_enqueue(k_c[i:i + 1], labels[i])
            else:
                self._dequeue_and_enqueue(k_c[i:i + 1], labels[i], w_u[i])

        SCL_logits = torch.cat([l_pos, l_pos_list, l_neg_list], dim=1)
        SCL_logits = nn.LogSoftmax(dim=1)(SCL_logits / self.temp)

        SCL_labels = torch.zeros([batch_size, 1 + self.queue_size*self.class_num]).cuda()
        SCL_labels[:, 0:self.queue_size+1].fill_(1.0/(self.queue_size+1))
        return SCL_logits, SCL_labels, q_f, (kl_loss1 + kl_loss2) * 0.0001

    def load_pretrained(self, network):
        q = network(projector_dim=1000, pretrained=self.pretrained)
        q.fc = self.encoder_q.fc
        self.encoder_q = q

    def inference(self, img):
        y, feat = self.encoder_q(img)
        return feat


