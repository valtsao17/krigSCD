import numpy as np
import scipy
from scipy.spatial.distance import pdist, cdist
import scipy.optimize
import matplotlib.pyplot as plt
from PIL import Image

def image_to_binary_matrix(image_path, threshold=128):
    """
    Converts a black and white image into a matrix of 1s and 0s.
    """
    image = Image.open(image_path).convert('L')
    img_array = np.array(image)
    binary_matrix = np.where(img_array < threshold, 1, 0)
    return binary_matrix

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
    
    return popt[0], popt[1]  # returns sill (variance) and range (tau)

def ordinary_kriging(data, mask):
    """
    Performs ordinary kriging on a given data matrix and a mask matrix.
    """
    known_idx = np.argwhere(mask == 0)
    unknown_idx = np.argwhere(mask == 1)
    
    x_known = known_idx[:, 1]
    y_known = known_idx[:, 0]
    x_unknown = unknown_idx[:, 1]
    y_unknown = unknown_idx[:, 0]
    
    data_known = data[mask == 0]
    var, tau = fit_semivariogram(x_known, y_known, data_known)
    
    points_known = np.column_stack((x_known, y_known))
    points_unknown = np.column_stack((x_unknown, y_unknown))
    
    dist_C = cdist(points_unknown, points_known)
    dist_Sigma = cdist(points_known, points_known)
    
    C = var * np.exp(-dist_C / tau)
    Sigma = var * np.exp(-dist_Sigma / tau)
    
    num_known = Sigma.shape[0]
    Sigma_prime = np.zeros((num_known + 1, num_known + 1))
    Sigma_prime[:num_known, :num_known] = Sigma
    Sigma_prime[num_known, :num_known] = 1
    Sigma_prime[:num_known, num_known] = 1
    
    C_prime = np.zeros((num_known + 1, C.shape[0]))
    C_prime[:num_known, :] = C.T
    C_prime[num_known, :] = 1
    
    W = np.linalg.solve(Sigma_prime, C_prime)
    zstar = np.dot(W[:num_known, :].T, data_known)
    mse = var - np.sum(W[:num_known, :] * C_prime[:num_known, :], axis=0) - W[-1, :]
    
    interpolated_data = np.copy(data)
    variance_data = np.zeros_like(data, dtype=float) 
    interpolated_data[unknown_idx[:, 0], unknown_idx[:, 1]] = zstar
    variance_data[unknown_idx[:, 0], unknown_idx[:, 1]] = mse
    

    return interpolated_data, variance_data

def main():
    filepath = '/home/vt55/RePaint/data/datasets/hrrr/final_tmp_test_64/20180807_hrrrt20.png'
    img = Image.open(filepath).convert("L")
    data = np.array(img).astype(np.int16)
    original_data = np.copy(data)

    mask = image_to_binary_matrix("/home/vt55/RePaint/data/datasets/gt_keep_masks/hrrr_tmp_final_results_64/1.5perknown_0.9swath.png").astype(np.int16)
    orig_mask = np.copy(mask)
    interpolated_data, kriging_variance = ordinary_kriging(data, mask)
    kriging_std = np.sqrt(kriging_variance)
    
    unknown_mask = (mask == 1)
    threshold = np.percentile(kriging_std[unknown_mask], 5)
    
    low_variance_indices = np.argwhere((kriging_std < threshold) & (mask == 1))
    
    data[low_variance_indices[:, 0], low_variance_indices[:, 1]] = \
        interpolated_data[low_variance_indices[:, 0], low_variance_indices[:, 1]]
    mask[low_variance_indices[:, 0], low_variance_indices[:, 1]] = 0

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    
    im0 = axs[0, 0].imshow(original_data, cmap='coolwarm', vmin=0, vmax=256)
    axs[0, 0].set_title("Original Data")
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)
    
    im1 = axs[0, 1].imshow(interpolated_data, cmap='coolwarm', vmin=0, vmax=256)
    axs[0, 1].set_title("Kriging Interpolated Data")
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])
    fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

    im2 = axs[0, 2].imshow(orig_mask, cmap='gray_r')
    axs[0, 2].set_title("Original Mask")
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])
    fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)
    
    im3 = axs[1, 0].imshow(data, cmap='coolwarm', vmin=0, vmax=256)
    axs[1, 0].set_title("Smoothed Ground Truth")
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])
    fig.colorbar(im3, ax=axs[1, 0], fraction=0.046, pad=0.04)

    im4 = axs[1, 1].imshow(kriging_std, cmap='inferno')
    axs[1, 1].set_title("Kriging Variance")
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])
    fig.colorbar(im4, ax=axs[1, 1], fraction=0.046, pad=0.04)
    
    im5 = axs[1, 2].imshow(mask, cmap='gray_r')
    axs[1, 2].set_title("New Mask (5th Percentile)")
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])
    fig.colorbar(im5, ax=axs[1, 2], fraction=0.046, pad=0.04)
    

    plt.tight_layout()
    plt.savefig("kriging_updated_results.png", dpi=500, transparent=False, facecolor="w")
    plt.show()
    
    plt.imsave("new_ground_truth.png", data, cmap='gray', vmin=0, vmax=256)
    plt.imsave("new_mask.png", mask, cmap='gray')

if __name__ == "__main__":
    main()
