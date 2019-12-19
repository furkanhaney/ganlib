import numpy as np
from scipy import linalg

def calculate_fid(act1, act2):
    mu1 = act1.mean(axis=0)
    mu2 = act2.mean(axis=0)
    sigma1 = np.cov(act1, rowvar=False)
    sigma2 = np.cov(act2, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def main():
    num_samples = 10000
    act_dim = 2048

    act1 = np.random.randn(num_samples, act_dim)
    act2 = np.random.randn(num_samples, act_dim)

    distance = calculate_fid(act1, act2)
    print(distance)

if __name__ == "__main__":
    main()
