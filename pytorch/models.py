import torch
import math
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from extra import SiameseTensorset

batch_size = 128
epochs = 10000

seed = 1  # ='random seed (default: 1)')
log_interval = 100
cuda = torch.cuda.is_available()

torch.manual_seed(seed)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


device = torch.device("cuda" if cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=batch_size, shuffle=True, **kwargs)
# cust_dset = torch.utils.data.TensorDataset(torch.Tensor(Coil_x[:,0,:,:]), torch.Tensor(X_dists[:, -1]))
# cust_dset = SiameseTensorset(cust_dset, X_inds)
# # sampler = NeighborBatchSampler(cust_dset, X_inds, batch_size)
# train_loader = torch.utils.data.DataLoader(
#     cust_dset, shuffle = True, batch_size=batch_size, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     cust_dset,
#     batch_size=batch_size, shuffle=True, **kwargs)


# cust_dset
def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


def binomial_kl(p, q):
    p = p.clamp(1e-13, 1 - 1e-7)
    q = q.clamp(1e-13, 1 - 1e-7)
    kl = p * (torch.log(p) - torch.log(q)) + (1 - p) * (torch.log(1 - p) - torch.log(1 - q))
    return kl


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(16384, 1600)
        self.fc2 = nn.Linear(1600, 1600)
        self.fc3 = nn.Linear(1600, 1600)
        self.fc41 = nn.Linear(1600, 1600)
        # Dividing line

        self.fc5 = nn.Linear(1600, 1600, bias=False)
        self.fc6 = nn.Linear(1600, 16384, bias=False)

    def encode(self, x):
        #         print(
        #             'encode', x.shape)
        h1 = F.selu(self.fc1(x))
        h2 = F.selu(self.fc2(h1))
        h3 = F.selu(self.fc3(h2))
        return self.fc41(h3)  # , self.fc22(h1)

    def reparameterize(self, loc):
        beta = 2 / 3.0
        gamma = -0.1
        zeta = 1.1
        gamma_zeta_ratio = math.log(-gamma / zeta)
        if self.training:
            u = torch.rand_like(loc)
            s = F.sigmoid((torch.log(u) - torch.log(1 - u) + loc) / beta)
            s = s * (zeta - gamma) + gamma
            penalty = F.sigmoid(loc - beta * gamma_zeta_ratio)  # .sum()
            return hard_sigmoid(s), penalty
        else:
            return hard_sigmoid(F.sigmoid(loc) * (zeta - gamma) + gamma), torch.zeros_like(loc)

    def decode(self, z):
        #         return F.sigmoid(self.fc6(z))
        h3 = F.selu(self.fc5(z))
        return F.sigmoid(self.fc6(h3))

    def forward(self, x1, x2=None):
        #         print(x1.shape)
        mu1 = self.encode(x1.view(-1, 16384))
        #         print(mu1.shape)
        z1, penalty1 = self.reparameterize(mu1)
        # Delete
        mu2 = self.encode(x2.view(-1, 16384))
        z2, penalty2 = self.reparameterize(mu2)

        #         penalty = torch.cat([penalty1, penalty2])
        adj = torch.matmul(torch.t(penalty1), penalty1)
        adj = adj * \
              (torch.ones_like(adj, device=device) - torch.eye(adj.shape[0], device=device))

        #         adj = torch.inverse(torch.eye(adj.shape[0], device=device) - eps * adj)
        #         km_loss = D1 * torch.abs(D1) * cooc
        #         k_means_loss = torch.max(torch.sum(km_loss * torch.abs(km_loss)), torch.zeros(1).to(device))
        # Delete
        return self.decode(z1), z2, penalty1, penalty2, adj.sum()  # - penalty) #, mu#, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, penalty, penalty2, lam=1.0e-3, lam2=1e-1):  # , # mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 16384), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + lam * penalty + lam2 * penalty2  # + KLD


def siamese_loss_function(recon_x1, x1, recon_x2, x2, encoding1,
                          encoding2, target, penalty, lam=1.0e-6, lam2=1e-1):  # , # mu, logvar):
    # #     print(target)
    BCE = F.binary_cross_entropy(recon_x1, x1.view(-1, 16384), size_average=False)
    #     BCE += F.binary_cross_entropy(recon_x2, x2.view(-1, 784), size_average=True)
    #     print(BCE)
    #     print( target.shape)
    #     print(encoding1.shape, encoding2.shape)
    #     print( F.binary_cross_entropy(encoding1, encoding2.detach(), reduce = False).shape)
    pos_encoding_loss = (target * torch.t(F.binary_cross_entropy(encoding1, encoding2.detach(), reduce=False))).sum()
    #     print( pos_encoding_loss)
    neg_encoding_loss = ((1 - target) * torch.t(encoding1 * encoding2)).sum()
    #     print( neg_encoding_loss)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #     KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #     print(BCE, lam * pos_encoding_loss, lam2 * neg_encoding_loss, 1e-4 * penalty)
    #     print(BCE, pos_encoding_loss, neg_encoding_loss, penalty)
    #     print('')
    return BCE + lam * (16384 / 1600) * (pos_encoding_loss + 6.90775 * neg_encoding_loss) + lam2 * penalty, \
           BCE, pos_encoding_loss, neg_encoding_loss, penalty  # + KLD


def train(epoch, lam=1e-6, lam2=1e-1):
    model.train()
    train_loss = 0
    for batch_idx, (data1, data2, dist) in enumerate(train_loader):
        #         print(data1.shape)
        #         print(data2.shape)
        #         print(dist.shape)
        data1 = data1.to(device)
        data2 = data2.to(device)
        #         print( data.shape)
        dist = dist.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        recon_batch1, recon_batch2, assignments1, assignments2, penalty = model(data1, data2)
        loss, bce, pos, neg, penalty = siamese_loss_function(recon_batch1, data1, recon_batch2, data2,
                                                             assignments1, assignments2, dist, penalty, lam, lam2)
        if epoch % 20 == 0:
            print('BCE:', bce, 'Pos:', pos, 'Neg:', neg, 'L0:', penalty)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.item() / len(data1)))
            #             print('Loss Components', loss, penalty, penalty2)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _, _) in enumerate(test_loader):
            #             print(data.shape)
            data = data.to(device).view(batch_size, 1, 128, 128)
            y = torch.ones(data.shape[0]).to(device)
            recon_batch = model.decode(model.reparameterize(model.encode(data.view(-1, 16384)))[0])
            #             test_loss += loss_function(recon_batch, data, penalty, penalty2).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, 1, 128, 128)[:n]])
                save_image(comparison.cpu(),
                           'reconstruction_' + str(epoch) + '.png', nrow=n)
                break

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))