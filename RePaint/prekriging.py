import os
import argparse
import numpy as np
import scipy
from scipy.spatial.distance import pdist, cdist
import scipy.optimize
import matplotlib.pyplot as plt
from PIL import Image
import yaml

def image_to_binary_matrix(image_path, threshold=128):
    """
    Converts a black and white image into a matrix of 1s and 0s.
    """
    image = Image.open(image_path).convert('L')
    img_array = np.array(image)
    return np.where(img_array < threshold, 1, 0)

def exponential_semivariogram(h, c, tau):
    """
    Computes an exponential semivariogram.
    """
    return c * (1 - np.exp(-h / tau))

def fit_semivariogram(x_known, y_known, data_known):
    """
    Fits an empirical semivariogram to the data.
    """
    points = np.column_stack((x_known, y_known))
    h = pdist(points)
    gamma_c = pdist(data_known[:, None], lambda u, v: 0.5 * (u - v)**2)

    bin_edges = np.linspace(0, 300, 51)
    hd = (bin_edges[:-1] + bin_edges[1:]) / 2
    inds = np.digitize(h, bin_edges) - 1
    sum_gamma = np.bincount(inds, weights=gamma_c, minlength=len(hd))
    count_gamma = np.bincount(inds, minlength=len(hd))
    gamma_s = np.where(count_gamma > 0, sum_gamma / count_gamma, 0)

    sill_lower, sill_upper = np.min(gamma_s), np.max(gamma_s)
    range_lower, range_upper = (bin_edges[1] - bin_edges[0]), np.max(h)

    popt, _ = scipy.optimize.curve_fit(
        exponential_semivariogram, hd, gamma_s,
        bounds=([sill_lower, range_lower], [sill_upper, range_upper])
    )
    return popt[0], popt[1]

def ordinary_kriging(data, mask):
    """
    Performs ordinary kriging on a given data matrix and a mask matrix.
    """
    known_idx = np.argwhere(mask == 0)
    unknown_idx = np.argwhere(mask == 1)

    x_known, y_known = known_idx[:,1], known_idx[:,0]
    x_unknown, y_unknown = unknown_idx[:,1], unknown_idx[:,0]
    data_known = data[mask == 0]

    var, tau = fit_semivariogram(x_known, y_known, data_known)
    points_known = np.column_stack((x_known, y_known))
    points_unknown = np.column_stack((x_unknown, y_unknown))

    dist_C = cdist(points_unknown, points_known)
    dist_Sigma = cdist(points_known, points_known)
    C = var * np.exp(-dist_C / tau)
    Sigma = var * np.exp(-dist_Sigma / tau)

    num_known = Sigma.shape[0]
    Sigma_prime = np.zeros((num_known+1, num_known+1))
    Sigma_prime[:num_known, :num_known] = Sigma
    Sigma_prime[num_known, :num_known] = 1
    Sigma_prime[:num_known, num_known] = 1

    C_prime = np.zeros((num_known+1, C.shape[0]))
    C_prime[:num_known, :] = C.T
    C_prime[num_known, :] = 1

    W = np.linalg.solve(Sigma_prime, C_prime)
    zstar = W[:num_known, :].T.dot(data_known)
    mse = var - np.sum(W[:num_known, :]*C_prime[:num_known, :], axis=0) - W[-1, :]

    interp = data.copy().astype(float)
    varmap = np.zeros_like(data, dtype=float)
    interp[unknown_idx[:,0], unknown_idx[:,1]] = zstar
    varmap[unknown_idx[:,0], unknown_idx[:,1]] = mse

    return interp, varmap

def parse_args():
    parser = argparse.ArgumentParser(description="Ordinary kriging smoothing of ground truth data")
    parser.add_argument('--gt_filepath', type=str, required=True, help='Path to the ground truth image')
    parser.add_argument('--mask_filepath', type=str, required=True, help='Path to the binary mask image')
    parser.add_argument('--output_dir', type=str, default='smoothed', help='Directory for smoothed outputs')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)

    # Load data and mask
    data = np.array(Image.open(args.gt_filepath).convert('L')).astype(np.int16)
    mask = image_to_binary_matrix(args.mask_filepath).astype(np.int16)

    interp, varmap = ordinary_kriging(data, mask)
    stdmap = np.sqrt(varmap)

    unknown = (mask == 1)
    thresh = np.percentile(stdmap[unknown], 5)
    low_var = np.argwhere((stdmap < thresh) & unknown)

    smoothed = data.copy().astype(float)
    smoothed[low_var[:,0], low_var[:,1]] = interp[low_var[:,0], low_var[:,1]]
    new_mask = mask.copy()
    new_mask[low_var[:,0], low_var[:,1]] = 0

    gt_name = os.path.splitext(os.path.basename(args.gt_filepath))[0]
    mask_name = os.path.splitext(os.path.basename(args.mask_filepath))[0]
    gt_out = os.path.join(args.output_dir, f"{gt_name}_smoothed.png")
    mask_out = os.path.join(args.output_dir, f"{mask_name}_smoothed.png")
    Image.fromarray(smoothed.astype(np.uint8)).save(gt_out)
    Image.fromarray(new_mask.astype(np.uint8)*255).save(mask_out)
    print(f"Saved smoothed GT to {gt_out}")
    print(f"Saved smoothed mask to {mask_out}")

if __name__ == '__main__':
    main()
