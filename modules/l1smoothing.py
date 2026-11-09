import math
from typing import Optional

import cv2
import numba
import numpy as np
from numba import prange
from scipy.sparse.linalg import LinearOperator, cg

from .graphbasedsegmentation import GraphBasedSegmentation


def _local_stencil(height, width, radius, use_gradient):
    # Keep only the orientation with a positive flattened offset.
    offsets = np.array([
        (dy, dx)
        for dy in range(min(radius, height - 1) + 1)
        for dx in range(-min(radius, width - 1), min(radius, width - 1) + 1)
        if dy > 0 or dx > 0
    ], dtype=np.int32).reshape(-1, 2)
    strides = offsets[:, 0].astype(np.int64) * width + offsets[:, 1]

    paths = np.zeros((len(offsets), radius + 1, 2), dtype=np.int32)
    if use_gradient:
        for k, (dy, dx) in enumerate(offsets):
            steps = max(dy, abs(dx))
            for step in range(steps + 1):
                paths[k, step] = (round(dy * step / steps), round(dx * step / steps))
    return offsets, strides, paths


def _reflection_counts(length, radius):
    # Count BORDER_REFLECT_101 offsets that land on each actual displacement.
    counts = np.zeros((length, 2 * radius + 1), dtype=np.int32)
    positions = np.arange(length)
    for offset in range(-radius, radius + 1):
        if length == 1:
            reflected = positions
        else:
            period = 2 * (length - 1)
            wrapped = (positions + offset) % period
            reflected = np.minimum(wrapped, period - wrapped)
        counts[positions, reflected - positions + radius] += 1
    return counts


@numba.njit(inline="always")
def _multiplicity(source, dest, width, radius, counts_y, counts_x):
    sy, sx = divmod(source, width)
    dy, dx = divmod(dest, width)
    y_offset, x_offset = dy - sy, dx - sx
    forward = counts_y[sy, radius + y_offset] * counts_x[sx, radius + x_offset]
    reverse = counts_y[dy, radius - y_offset] * counts_x[dx, radius - x_offset]
    return forward + reverse


@numba.njit(inline="always")
def _distance_squared(features, source, dest):
    distance = 0.0
    for channel in range(features.shape[1]):
        diff = features[source, channel] - features[dest, channel]
        distance += diff * diff
    return distance


@numba.njit(inline="always")
def _affinity(distance, sigma_squared):
    if sigma_squared == 0:
        return 1.0 if distance == 0 else 0.0
    return math.exp(-0.5 * distance / sigma_squared)


@numba.njit(parallel=True)
def _local_weights(features, gradient_squared, offsets, strides, paths, width,
                   radius, counts_y, counts_x, eta, sigma_squared):
    height = counts_y.shape[0]
    weights = np.zeros((height * width, len(strides)), dtype=np.float32)
    for pixel in prange(height * width):
        source = np.int64(pixel)
        y, x = divmod(source, width)
        for k in range(len(strides)):
            dy, dx = offsets[k]
            if y + dy >= height or not 0 <= x + dx < width:
                continue
            dest = source + strides[k]
            distance = _distance_squared(features, source, dest)
            if gradient_squared.size:
                peak = 0.0
                for step in range(max(dy, abs(dx)) + 1):
                    py, px = paths[k, step]
                    peak = max(peak, gradient_squared[y + py, x + px])
                distance = max(distance, eta * peak)
            multiplicity = _multiplicity(source, dest, width, radius, counts_y, counts_x)
            weights[source, k] = math.sqrt(multiplicity) * _affinity(distance, sigma_squared)
    return weights


@numba.njit(parallel=True)
def _reduce_local(weights, strides, width, radius, counts_y, counts_x, scratch,
                  block_size, scale, z, dual, shrink_limit):
    """Reduce the diagonal, normal product, or next split-Bregman right-hand side."""
    size = weights.shape[0]
    block_errors = np.empty(scratch.shape[0], dtype=np.float64)
    for block in prange(scratch.shape[0]):
        start = np.int64(block) * block_size
        partial = scratch[block]
        partial.fill(0)
        block_error = 0.0
        for source in range(start, min(start + block_size, size)):
            total = 0.0
            for k in range(len(strides)):
                weight = np.float64(weights[source, k])
                if weight == 0:
                    continue
                dest = source + strides[k]
                if dual is not None:
                    # Fold identical rows (including reverse orientations) into one.
                    # Multiplicity m scales both the row and shrink threshold by sqrt(m).
                    limit = shrink_limit * math.sqrt(
                        _multiplicity(source, dest, width, radius, counts_y, counts_x)
                    )
                    old_dual = dual[source, k]
                    q = weight * (z[source] - z[dest]) + old_dual
                    dual[source, k] = min(max(q, -limit), limit)
                    change = dual[source, k] - old_dual
                    block_error += change * change
                    # b' = clip(q), d' = q - b', so d' - b' = q - 2b'.
                    value = weight * (q - 2 * dual[source, k])
                else:
                    value = weight * weight
                    if z is not None:
                        value *= z[source] - z[dest]
                total += value
                if z is None:
                    partial[dest - start] += value
                else:
                    partial[dest - start] -= value
            partial[source - start] += total
        block_errors[block] = block_error

    # Each block scatters into its own buffer. A pixel can also receive
    # contributions from the preceding block's halo, which is at most one block.
    result = np.empty(size, dtype=np.float64)
    for pixel in prange(size):
        block, index = divmod(np.int64(pixel), block_size)
        value = scratch[block, index]
        if block > 0 and index + block_size < scratch.shape[1]:
            value += scratch[block - 1, index + block_size]
        result[pixel] = scale * value
    primal_error = 0.0
    for block_error in block_errors:
        primal_error += block_error
    return result, primal_error


@numba.njit
def _representatives(pixels, labels, count):
    channels = pixels.shape[1]
    sums = np.zeros((count, channels), dtype=np.float64)
    sizes = np.zeros(count, dtype=np.int64)
    for index, label in enumerate(labels):
        sizes[label] += 1
        for channel in range(channels):
            sums[label, channel] += pixels[index, channel]
    for label in range(count):
        sums[label] /= sizes[label]

    closest = np.full(count, np.inf)
    representatives = np.empty(count, dtype=np.int64)
    for index, label in enumerate(labels):
        distance = 0.0
        for channel in range(channels):
            diff = pixels[index, channel] - sums[label, channel]
            distance += diff * diff
        if distance < closest[label]:
            closest[label] = distance
            representatives[label] = index
    return representatives


@numba.njit
def _global_weights(features, source, dest, sigma_squared):
    weights = np.empty(len(source), dtype=np.float64)
    for k in range(len(source)):
        distance = _distance_squared(features, source[k], dest[k])
        # The paper's global energy counts both orientations.
        weights[k] = math.sqrt(2.0) * _affinity(distance, sigma_squared)
    return weights


def L1Smooth(img: np.ndarray,
             alpha: float = 50.0,
             beta: float = 5.0,
             gamma: float = 2.5,
             _lambda: float = 5.0,
             maxIter: int = 4,
             kappa: float = 0.3,
             sigma: float = 1,
             half_window: int = 3,
             eta: float = 15,
             edge_preserving: int = 0,
             global_size: int = 20,
             threshold: float = 0.001,
             msk: Optional[np.ndarray] = None,
             *,
             rgb: bool = False):
    """Flatten a grayscale or BGR image in [0, 1], or RGB with rgb=True.

    Alpha and beta weight the local and global L1 terms. Gamma weights
    fidelity, with a mask of 1 removing fidelity at that pixel.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim != 2 and (img.ndim != 3 or img.shape[2] != 3):
        raise ValueError("L1 smoothing expects a grayscale or three-channel image")
    height, width = img.shape[:2]
    size = height * width
    if not size:
        raise ValueError("L1 smoothing expects a nonempty image")
    settings = (alpha, beta, gamma, _lambda, kappa, sigma, eta, threshold)
    if any(not math.isfinite(value) or value < 0 for value in settings):
        raise ValueError("L1 smoothing weights and threshold must be finite and nonnegative")
    if half_window < 0 or maxIter < 0 or global_size < 1:
        raise ValueError("Invalid L1 smoothing window, iteration count, or superpixel limit")

    beta = 0.0 if edge_preserving else beta
    if not maxIter or not _lambda or (not (alpha and half_window) and not beta):
        return img.copy()

    fidelity = np.full(size, gamma, dtype=np.float64)
    if msk is not None:
        mask = np.asarray(msk, dtype=np.float64)
        if mask.shape == (height, width, 1):
            mask = mask[..., 0]
        if mask.shape not in ((height, width), (size,)) or not np.all((mask >= 0) & (mask <= 1)):
            raise ValueError("L1 smoothing mask must match the image and have values in [0, 1]")
        fidelity *= 1 - mask.ravel()

    color = img.ndim == 3
    channels = 3 if color else 1
    target = img.reshape(size, channels)
    if color:
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB if rgb else cv2.COLOR_BGR2LAB)
        lab[..., 0] /= 100.0
        lab[..., 1:] /= 255.0
    else:
        lab = img[..., None].copy()

    if beta:
        # Segmentation's K is in 8-bit color units. Keep the segmentation
        # lightness weighting independent of the affinity's kappa.
        segmentation_features = lab * 255.0
        if color:
            segmentation_features[..., 0] *= 0.3
        labels, count = GraphBasedSegmentation(segmentation_features, 0.5, 500, global_size)
        representatives = _representatives(target, labels.ravel(), count)
        del segmentation_features, labels
        first, second = np.triu_indices(count, k=1)
        source, dest = representatives[first], representatives[second]
    else:
        source = dest = np.empty(0, dtype=np.int64)

    lab[..., 0] *= kappa
    features = lab.reshape(size, channels)
    sigma_squared = sigma * sigma
    global_weights = _global_weights(features, source, dest, sigma_squared)

    radius = half_window if alpha else 0
    use_gradient = bool(edge_preserving and eta and radius)
    offsets, strides, paths = _local_stencil(height, width, radius, use_gradient)
    counts_y = _reflection_counts(height, radius)
    counts_x = _reflection_counts(width, radius)
    if use_gradient:
        # Sobel of the channel mean is the mean of the channel Sobel gradients.
        gray = img.mean(axis=2) if color else img
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, borderType=cv2.BORDER_REFLECT_101)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_REFLECT_101)
        gradient_squared = gx * gx + gy * gy
        del gray, gx, gy
    else:
        gradient_squared = np.empty((0, 0), dtype=np.float32)
    local_weights = _local_weights(
        features, gradient_squared, offsets, strides, paths, width, radius,
        counts_y, counts_x, eta, sigma_squared,
    )
    del features, lab, gradient_squared
    halo = int(strides[-1]) if len(strides) else 0
    block_size = max(halo, 1024)
    blocks = (size + block_size - 1) // block_size
    scratch = np.empty((blocks, block_size + halo), dtype=np.float64)

    # Use penalties lambda*alpha and lambda*beta for the two splits.
    # This gives the weighted L1 objective with a common 1/lambda shrink limit.
    local_scale, global_scale = _lambda * alpha, _lambda * beta
    shrink_limit = 1.0 / _lambda
    global_normal = global_scale * global_weights**2

    def reduce_local(z=None, dual=None):
        if not len(strides):
            return np.zeros(size, dtype=np.float64), 0.0
        return _reduce_local(
            local_weights, strides, width, radius, counts_y, counts_x, scratch,
            block_size, local_scale, z, dual, shrink_limit,
        )

    diagonal, _ = reduce_local()
    np.add.at(diagonal, source, global_normal)
    np.add.at(diagonal, dest, global_normal)
    connected = diagonal > 0
    if not connected.any():
        return img.copy()
    diagonal += fidelity
    free = diagonal == 0
    if not fidelity.any():
        # A constant minimizes both sparsity terms when there is no fidelity.
        result = target.copy()
        result[connected] = target[connected].mean(axis=0)
        return result.reshape(img.shape)
    diagonal[free] = 1.0

    def matvec(z):
        z = z.ravel()
        out, _ = reduce_local(z)
        values = global_normal * (z[source] - z[dest])
        np.add.at(out, source, values)
        np.add.at(out, dest, -values)
        out += fidelity * z
        return out

    normal = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
    preconditioner = LinearOperator(
        (size, size), matvec=lambda z: z.ravel() / diagonal, dtype=np.float64,
    )
    result = np.empty_like(target)
    local_dual = np.empty_like(local_weights)
    for channel in range(channels):
        local_dual.fill(0)
        global_dual = np.zeros_like(global_weights)
        original = target[:, channel].astype(np.float64)
        data_rhs = fidelity * original
        rhs = data_rhs
        z = original
        for iteration in range(maxIter):
            previous = z
            z, info = cg(normal, rhs, x0=previous, M=preconditioner)
            if info or not np.isfinite(z).all():
                raise RuntimeError(f"L1 smoothing conjugate gradient failed (info={info})")
            z[free] = original[free]
            if iteration + 1 == maxIter:
                break

            rhs, primal_error = reduce_local(z, local_dual)
            q = global_weights * (z[source] - z[dest]) + global_dual
            new_dual = np.clip(q, -math.sqrt(2.0) * shrink_limit, math.sqrt(2.0) * shrink_limit)
            dual_change = new_dual - global_dual
            primal_error += np.dot(dual_change, dual_change)
            global_dual = new_dual
            if max(np.dot(z - previous, z - previous), primal_error) <= threshold / channels:
                break

            values = global_scale * global_weights * (q - 2 * global_dual)
            np.add.at(rhs, source, values)
            np.add.at(rhs, dest, -values)
            rhs += data_rhs
        result[:, channel] = z
    return result.reshape(img.shape)


if __name__ == '__main__':
    img = cv2.imread('bg.png', cv2.IMREAD_COLOR)
    img = cv2.resize(img, (300, 300))
    img2 = L1Smooth(img.astype(np.float32) / 255.0, edge_preserving=1)
    img2 = (img2 * 255).clip(0, 255).astype(np.uint8)
    cv2.imshow('1', img)
    cv2.imshow('2', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
