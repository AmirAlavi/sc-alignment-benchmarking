import numpy as np
import torch

def fit_transform_rigid(A, B):
    # See http://nghiaho.com/?page_id=671
    # center
    A_centroid = np.mean(A, axis=0)
    B_centroid = np.mean(B, axis=0)

    H = np.dot((A - A_centroid).T, B - B_centroid)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    detR = np.linalg.det(R)
    x = np.identity(Vt.T.shape[1])
    x[x.shape[0]-1, x.shape[1]-1] = detR
    R = np.linalg.multi_dot([Vt.T, x, U.T])

    t = B_centroid.T - np.dot(R, A_centroid.T)

    return R, t
    
def fit_transform_affine(A, B, optim='adam', lr=1e-3, epochs=1000):
    d = A.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device {}'.format(device))
    A = torch.from_numpy(A).float().to(device)
    B = torch.from_numpy(B).float().to(device)
    f = torch.nn.Sequential()
    f.add_module('lin', torch.nn.Linear(d, d, bias=True))
    f.to(device)
    if optim == 'adam':
        optimizer = torch.optim.Adam(f.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(f.parameters(), lr=lr, momentum=0.9, nesterov=True)
    f.train()
    for e in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean(torch.norm(f(A) - B, p=2, dim=1)**2)
        if e % 100 == 0:
            print(f'\tEpoch: {e}/{epochs}, loss: {loss.item()}')
        loss.backward()
        optimizer.step()
    theta = np.zeros((d + 1, d + 1))
    theta[:d, :d] = f[0].weight.data.cpu().numpy()
    theta[:d, -1] = f[0].bias.data.cpu().numpy()
    theta[-1, -1] = 1.
    return theta, f[0].weight.data.cpu().numpy(), f[0].bias.data.cpu().numpy()


        

