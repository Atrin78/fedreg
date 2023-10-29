from thop import profile
import os
import json
from torch.utils.data import TensorDataset
import numpy as np
import torch
from loguru import logger
from PIL import Image
import h5py


def FSGM(model, inp, label, iters, eta):
    inp.requires_grad = True
    minv, maxv = float(inp.min().detach().cpu().numpy()), float(inp.max().detach().cpu().numpy())
    for _ in range(iters):
        pred = model.forward(inp)
        loss = model.loss(pred, label).mean()
        dp = torch.sign(torch.autograd.grad(loss, inp)[0])
        inp.data.add_(eta*dp.detach()).clamp(minv, maxv)
    return inp


class CusDataset(TensorDataset):
    x_mean = None
    x_std = None
    def __init__(self, data, transform=None):
        assert "x" in data
        assert "y" in data
        self.data = {}
        self.data["x"] = (data["x"])
        self.data["y"] = (data["y"])
        self.transform = transform

    def __getitem__(self, item):
        if self.transform is None:
            ret = torch.tensor(self.data['x'][item])
            if CusDataset.x_mean is not None: 
                ret = (ret - CusDataset.x_mean) / (CusDataset.x_std+0.00001)

        else:
            ret = np.array(self.data["x"][item]).astype("uint8")
            if ret.shape[-1] == 3:
                ret = ret
            elif ret.shape[0] == 3:
                ret = ret.transpose(1, 2, 0)
            else:
                ret = ret
            if CusDataset.x_mean is not None: 
                ret = ret.reshape((28, 28, 1))
                ret = self.transform(ret).float()
                ret = (ret-ret.min())/(ret.max()-ret.min())
            else:
                ret = self.transform(Image.fromarray(ret))

        return [ret, torch.tensor(self.data["y"][item])]

    def __len__(self):
        return len(self.data["x"])


class ImageDataset(TensorDataset):
    def __init__(self,  data, transform=None, image_path=None):
        self.transform = transform

        assert "x" in data
        assert "y" in data
        self.data = {}
        self.data["x"] = (data["x"])
        self.data["y"] = (data["y"])
        if len(self.data["x"]) < 20000:
            File = h5py.File(image_path, "r")
            self.image_path = {}
            for name in self.data["x"]:
                name = name.replace(".png", "")
                self.image_path[name+"_X"] = np.array(File[name+"_X"])
                self.image_path[name+"_Y"] = np.array(File[name+"_Y"])
            File.close()
        else:
            self.image_path = h5py.File(image_path, "r")

    def __getitem__(self, item):
        path = self.data["x"][item]
        path = path.replace(".png", "")
        image, y = Image.fromarray((np.array(self.image_path[path+"_X"])*255).transpose(1, 2, 0).astype(np.uint8)), self.image_path[path+"_Y"]
        if self.transform is None:
            ret = torch.tensor(image)
        else:
            try:
                assert image.mode == "RGB"
            except:
                image = image.convert("RGB")
            ret = self.transform(image)

        return [ret, torch.tensor(self.data["y"][item])]

    def __len__(self):
        return len(self.data["x"])


def Flops(model, inp):
    return profile(model, inputs=(inp,), verbose=False)[0]

# Here is where data is devided among clients
def read_data(train_data_path, test_data_path):
    if not isinstance(test_data_path, list):
        test_data_path = [test_data_path, ]
    groups = []
    train_data = {}
    test_data = [{} for _ in test_data_path]
    train_files = os.listdir(train_data_path)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_path, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])
    for F, td in zip(test_data_path, test_data):
        test_files = os.listdir(F)
        test_files = [f for f in test_files if f.endswith(".json")]
        for f in test_files:
            file_path = os.path.join(F, f)
            with open(file_path, "r") as inf:
                cdata = json.load(inf)
            td.update(cdata["user_data"])
    clients = list(sorted(train_data.keys()))
    return clients, groups, train_data, test_data


def decode_stat(stat):
    if len(stat) == 4:
        ids, groups, num_samples, tot_correct = stat
        if isinstance(num_samples[0], list):
            assert len(num_samples) == len(tot_correct)
            idx = 0
            for a, b in zip(tot_correct, num_samples):
                logger.info("Test_{} Accuracy: {}".format(idx, sum(a) * 1.0 / sum(b)))
                idx += 1
        else:
            logger.info("Accuracy: {}".format(sum(tot_correct) / sum(num_samples)))
    elif len(stat) == 5:
        ids, groups, num_samples, tot_correct, losses = stat
        logger.info("Accuracy: {} Loss: {}".format(sum(tot_correct) / sum(num_samples), sum(losses) / sum(num_samples)))
    else:
        raise ValueError
    

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
    x: A num_examples x num_features matrix of features.

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
    A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
        gram: A num_examples x num_examples symmetric matrix.
        unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
        A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.
        debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
        The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    logger.info("CKA: {}".format(scaled_hsic ))
    logger.info("CKA: {}".format(normalization_x ))
    logger.info("CKA: {}".format(normalization_y ))
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
        xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
        + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
        features_x: A num_examples x num_features matrix of features.
        features_y: A num_examples x num_features matrix of features.
        debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
        The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)