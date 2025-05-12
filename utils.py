import numpy as np
from scipy.stats import f, chi2
from scipy.io import mmread
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from typing import Callable, List, Any

gray = '#8b96ad'
red = '#c74546'

def estimate_s(N1, N2=None, plot=True, ax=None, min_mean=0.1, max_mean=np.inf, 
               bins=np.arange(-0.5, 1.5, 0.01) - 0.005, histcolor='lightgray', modcolor='#0070c0', 
               meancolor='#3d405b'):
    """
    Estimates the extrinsic noise `s`.

    This function computes normalized covariance to estimate `s`.
    Optionally, it can plot a histogram of the normalized covariance values and highlight the mean and
    mode.

    Parameters
    ----------
    N1 : ndarray
        A 2D numpy array representing the first gene count matrix with
        cells as rows and genes as columns.

    N2 : ndarray, optional
        A 2D numpy array representing the second gene count matrix with
        cells as rows and genes as columns. If `None`, the calculation is performed
        only on `N1`. Default is `None`.

    plot : bool, optional
        If `True`, a histogram of the covariance values is plotted. Default is `True`.

    ax : matplotlib.axes.Axes, optional
        A matplotlib axis object where the histogram will be plotted. If `None`,
        a new figure and axis are created. Default is `None`.

    min_mean : float, optional
        The minimum mean expression threshold for genes to be included in the
        calculation. Default is 0.1.

    max_mean : float, optional
        The maximum mean expression threshold for genes to be included in the
        calculation. Default is `np.inf`.

    bins : ndarray, optional
        The bins for the histogram. Default is `np.arange(0, 1, 0.01) - 0.005`.

    color : str, optional
        The color of the histogram bars. Default is `'lightgray'`.

    modcolor : str, optional
        The color of the vertical line indicating the mode of the histogram. Default is `'#0070c0'`.

    meancolor : str, optional
        The color of the vertical line indicating the mean of the histogram. Default is `'#3d405b'`.

    Returns
    -------
    s_mod : float
        The mode of the normalized covariance values calculated as the midpoint of the most
        frequent bin in the histogram.

    """
    ### calculate normalized covariance
    if N2 is None:
        idx = (N1.mean(0) > min_mean) & (N1.mean(0) < max_mean)
        X = N1[:, idx]
        X_mean = X.mean(0)
        p = len(X_mean)
        eta = np.cov(X, rowvar=False) / X_mean[:, None] / X_mean[None, :]
        np.fill_diagonal(eta, np.nan)
        eta = eta[~np.isnan(eta)]

    else:
        idx1 = (N1.mean(0) > min_mean) & (N1.mean(0) < max_mean)
        idx2 = (N2.mean(0) > min_mean) & (N2.mean(0) < max_mean)
        X = np.concatenate((N1[:, idx1], N2[:, idx2]), axis=1)
        X_mean = X.mean(0)
        p1 = idx1.sum()
        p2 = idx2.sum()
        eta = np.cov(X, rowvar=False) / X_mean[:, None] / X_mean[None, :]
        eta = eta[p1:, :p1]
        
    ### calculate s as the mean
    s = np.mean(eta)

    ### calculate s_mod as the midpoint of the most frequent bin in the histogram.
    if plot is False:
        hist, bins = np.histogram(eta.flatten(), bins=bins)
        s_mod = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if N2 is None:
            hist, bins, patches = ax.hist(eta.flatten(), bins=bins, label=str(p) + ' genes', color=histcolor)
        else:
            hist, bins, patches = ax.hist(eta.flatten(), bins=bins, label=str(p1) + r'$\times$' + str(p2) + ' genes', color=histcolor)
        s_mod = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
        ax.axvline(x=s, c=meancolor, zorder=0, linewidth=6, label='mean=' + str(np.around(s, 3)))
        ax.axvline(x=s_mod, c=modcolor, zorder=0, linewidth=6, label='mode=' + str(np.around(s_mod, 3)))
        ax.legend(loc='upper right')
    return {'mod':s_mod,'mean':s}
    
def estimate_s_(N1, N2=None, plot=True, ax=None, min_mean=0.1, max_mean=np.inf, 
               bins=np.arange(-0.5, 1.5, 0.01) - 0.005, color='lightgray', modcolor='#0070c0', 
               meancolor='#3d405b'):
    ### calculate normalized covariance
    if N2 is None:
        idx = (N1.mean(0) > min_mean) & (N1.mean(0) < max_mean)
        X = N1[:, idx]
        X_mean = X.mean(0)
        p = len(X_mean)
        eta = np.cov(X, rowvar=False) / X_mean[:, None] / X_mean[None, :]
        np.fill_diagonal(eta, np.nan)
        eta = eta[~np.isnan(eta)]

    else:
        idx1 = (N1.mean(0) > min_mean) & (N1.mean(0) < max_mean)
        idx2 = (N2.mean(0) > min_mean) & (N2.mean(0) < max_mean)
        X = np.concatenate((N1[:, idx1], N2[:, idx2]), axis=1)
        X_mean = X.mean(0)
        p1 = idx1.sum()
        p2 = idx2.sum()
        eta = np.cov(X, rowvar=False) / X_mean[:, None] / X_mean[None, :]
        eta = eta[p1:, :p1]
        
    ### calculate s as the mean
    s = np.mean(eta)

    ### calculate s_mod as the midpoint of the most frequent bin in the histogram.
    if plot is False:
        hist, bins = np.histogram(eta.flatten(), bins=bins)
        s_mod = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if N2 is None:
            hist, bins, patches = ax.hist(eta.flatten(), bins=bins, label='sd='+str(np.around(np.std(eta), 3)), color=color)
        else:
            hist, bins, patches = ax.hist(eta.flatten(), bins=bins, label=str(p1) + r'$\times$' + str(p2) + ' genes', color=color)
        s_mod = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
        ax.axvline(x=s, c=meancolor, zorder=0, linewidth=6, label='mean=' + str(np.around(s, 3)))
        ax.axvline(x=s_mod, c=modcolor, zorder=0, linewidth=6, label='mode=' + str(np.around(s_mod, 3)))
        ax.legend(loc='upper right')
    return
    
def bootstrapping_func(
    func: Callable,
    datasets: List[np.ndarray],
    B: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    n_cores: int = 1
):
    """
    Performs bootstrapping to estimate the statistics calculated by func given a list of datasets.

    Parameters
    ----------
    func : Callable
        Function used to calculate statistics like extrinsic noise on each bootstrap sample.
    datasets : List[np.ndarray]
        List of 2D datasets to compute func. Each dataset has shape (n_samples, n_features).
    B : int, optional
        Number of bootstrap samples. Default is 1000.
    alpha : float, optional
        Significance level for confidence intervals. Default is 0.05 for 95% confidence.
    seed : int, optional
        Seed for random number generation. Default is 0.
    n_cores : int, optional
        Number of cores to use for parallel processing. Default is 1.
        
    Returns
    -------
    lower_bound : ndarray
        The lower confidence bound for each feature.
    upper_bound : ndarray
        The upper confidence bound for each feature.
    s_bootstrap : ndarray
        The bootstrap estimates of extrinsic noise across bootstrap samples.
    """ 

    # Set random seed for reproducibility
    bootstrap_indices = np.arange(B)

    args = [
        (datasets, indices)  # Each arg contains both datasets and bootstrap_idx
        for indices in bootstrap_indices
    ]
    
    # Use multiprocessing Pool to perform bootstrapping in parallel
    with Pool(processes=n_cores) as pool:
        s_bootstrap = pool.starmap(func, args)
    s_bootstrap = np.array(s_bootstrap)   

    # Calculate the confidence interval for the residues
    lower_bound = np.nanpercentile(s_bootstrap, alpha / 2 * 100, axis=0)
    upper_bound = np.nanpercentile(s_bootstrap, (1 - alpha / 2) * 100, axis=0)

    # Optionally, you can return both the bootstrap estimates and the confidence intervals
    return lower_bound, upper_bound, s_bootstrap

def overdispersion(sampled_datasets,idx,eps=0):
    """
    This function computes the overdispersion for the sampled datasets. 
    The normalized variance is defined as:
    
        eta = (variance - mean) / mean^2 - eps / mean
    
    Parameters
    ----------
    sampled_datasets : list of ndarray
        A list of 2D arrays (datasets), where each dataset represents a set of samples with features.
        Each dataset should have shape `(n_samples, n_features)`.
    
    eps : float or 1D array, optional, default=0
        A small constant to adjust the normalized variance formula. It can be used to prevent division by zero or
        to apply a bias to the calculated residue.

    Returns
    -------
    s : list of ndarray
        A list containing the normalized variance for each dataset in `sampled_datasets`.
        Each entry in the list corresponds to a dataset and has shape `(n_features,)`.
    """
    
    np.random.seed(idx)
    assert len(sampled_datasets)==1
    n_samples = sampled_datasets[0].shape[0]
    b_idx = np.random.choice(a=n_samples,size=n_samples)
    X = sampled_datasets[0][b_idx]
    assert len(np.shape(X))==2, "sampled_datasets needs to be a list of 2D arrays"
    
    bootstrap_var = X.var(axis=0)
    bootstrap_mean = X.mean(0)

    # Calculate residue (normalized variance)
    eta = (bootstrap_var - bootstrap_mean) / bootstrap_mean**2 - eps / bootstrap_mean
    
    return np.array(eta)

def delta_eta(sampled_datasets,idx,offset=0):
    """
    This function computes the normalized covariance between nascent and mature of each gene. 
    The normalized covariance is defined as:
    
        Δeta = eta_nm - eta_mm
    
    Parameters
    ----------
    sampled_datasets : list of ndarray
        A list of 2D arrays (datasets), where each dataset represents a set of samples with features.
        Each dataset should have shape `(n_samples, n_features)`.
    offset : float
        The expected difference between eta_nm and eta_mm: eta_mm - eta_nm - offset
    
    Returns
    -------
    s : list
        A list containing the Δeta for each gene.
    """
    s = []
    np.random.seed(idx)
    n_samples = sampled_datasets[0].shape[0]
    b_idx = np.random.choice(a=n_samples,size=n_samples)
    N = sampled_datasets[0][b_idx]
    M = sampled_datasets[1][b_idx]
    n, p = np.shape(N)
    N_mean = np.mean(N, axis=0)
    M_mean = np.mean(M, axis=0)
    M_var = np.var(M, axis=0)

    cov = np.sum((N - N_mean[None,:]) * (M - M_mean[None,:]), axis=0) / (n - 1)   
    eta_nm = cov/N_mean/M_mean
    eta_mm = (M_var-M_mean)/M_mean**2
        
    return eta_mm - eta_nm
    
def CCC(y1, y2):
    """
    Calculates the Concordance Correlation Coefficient (CCC) between two sets of ratings.

    The CCC evaluates the agreement between two variables by measuring both precision 
    (the Pearson correlation) and accuracy (the deviation from the 45-degree line through the origin).

    Parameters
    ----------
    y1 : array-like
        First set of ratings or measurements.
    y2 : array-like
        Second set of ratings or measurements.

    Returns
    -------
    CCC : float
        The Concordance Correlation Coefficient between `y1` and `y2`, ranging from -1 to 1.
        A value of 1 indicates perfect concordance, 0 indicates no concordance, and -1 indicates perfect discordance.
    """
    # Convert ratings to numpy arrays
    y1_array = np.array(y1)
    y2_array = np.array(y2)
    
    # Calculate means
    mean_y1 = np.mean(y1_array)
    mean_y2 = np.mean(y2_array)
    
    # Calculate variances
    var_y1 = np.var(y1_array, ddof=1)
    var_y2 = np.var(y2_array, ddof=1)
    
    # Calculate covariance
    cov_y1y2 = np.cov(y1_array, y2_array)[0, 1]
    
    # Calculate bias correction factor
    CCC = 2 * cov_y1y2 / (var_y1 + var_y2 + (mean_y1 - mean_y2) ** 2)
    
    return CCC

def load_10x(datadir):
    """
    Loads 10x Genomics single-cell RNA-seq data from a specified directory.

    This function reads the barcodes, features, and matrix files typically found
    in a 10x Genomics output directory and constructs an `AnnData` object.

    Parameters
    ----------
    datadir : str
        Path to the directory containing the 10x Genomics data files
        (`barcodes.tsv`, `features.tsv`, and `matrix.mtx`).

    Returns
    -------
    tenx : AnnData
        An `AnnData` object containing the expression matrix with cells as observations
        and genes/features as variables.
    """
    barcode = pd.read_csv(datadir + '/barcodes.tsv', sep='\t', header=None)
    feature = pd.read_csv(datadir + '/features.tsv', sep='\t', header=None)
    matrix = mmread(datadir + '/matrix.mtx')
    tenx = ad.AnnData(matrix.T, obs=barcode, var=feature)
    return tenx

def intersect_idx(kb_bcd, tenx_bcd):
    """
    Finds the indices of common barcodes between two lists.

    Parameters
    ----------
    kb_bcd : array-like
        List or array of barcodes from the first dataset.
    tenx_bcd : array-like
        List or array of barcodes from the second dataset.

    Returns
    -------
    kb_common_bc_idx : ndarray
        Indices of the common barcodes in `kb_bcd`.
    tenx_common_bc_idx : ndarray
        Indices of the common barcodes in `tenx_bcd`.
    """
    common_bc = np.intersect1d(np.array(kb_bcd), np.array(tenx_bcd))

    # Find indices of common barcodes in both lists
    kb_common_bc_idx = np.array([np.where(np.array(kb_bcd) == bc)[0][0] for bc in common_bc])
    tenx_common_bc_idx = np.array([np.where(np.array(tenx_bcd) == bc)[0][0] for bc in common_bc])

    return kb_common_bc_idx, tenx_common_bc_idx

def get_ensg_id(gene_name):
    base_url = "https://rest.ensembl.org"
    endpoint = f"/xrefs/symbol/human/{gene_name}"

    url = f"{base_url}{endpoint}"

    response = requests.get(url, headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        data = response.json()
        if data:
            ensg_id = data[0]['id']
            return ensg_id
        else:
            return None
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

def calculate_gene_length(gtf_file):
    gene_lengths = {}

    with open(gtf_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header lines
            fields = line.strip().split('\t')
            if fields[2] == 'gene':
                gene_id = fields[8].split(';')[0].split('"')[1]
                start = int(fields[3])
                end = int(fields[4])
                length = end - start + 1  # Add 1 to include both start and end positions
                if gene_id not in gene_lengths:
                    gene_lengths[gene_id] = length
                else:
                    gene_lengths[gene_id] += length

    return gene_lengths

def calculate_exon_number(gtf_file):
    exon_counts = {}

    with open(gtf_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header lines
            fields = line.strip().split('\t')
            if fields[2] == 'exon':
                gene_id = fields[8].split(';')[0].split('"')[1]
                if gene_id not in exon_counts:
                    exon_counts[gene_id] = 1
                else:
                    exon_counts[gene_id] += 1

    return exon_counts