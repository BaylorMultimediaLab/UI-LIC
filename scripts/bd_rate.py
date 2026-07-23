#!/usr/bin/env python3
"""
Bjontegaard Delta (BD) Rate and BD-Metric Calculator
Implements standard Bjontegaard Delta (BD-Rate and BD-Metric) algorithms
for evaluating Rate-Distortion (RD) curves in image and video compression.

References:
- G. Bjontegaard, "Calculation of average PSNR differences between RD-curves," ITU-T SG16/Q6 Doc. VCEG-M33, 2001.
- ITU-T H.Series Supplement 13 (07/2021) "Performance assessment of video coding technologies".
"""

import numpy as np
import scipy.interpolate
import scipy.integrate

def _make_strictly_increasing(x, y):
    """
    Sorts (x, y) by x, and ensures x is strictly increasing by averaging y
    for duplicate x values.
    """
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]
    unique_x, counts = np.unique(x_s, return_counts=True)
    if len(unique_x) == len(x_s):
        return x_s, y_s
    unique_y = [np.mean(y_s[x_s == ux]) for ux in unique_x]
    return np.array(unique_x, dtype=np.float64), np.array(unique_y, dtype=np.float64)

def bd_rate(rate1, dist1, rate2, dist2, piece_wise=True):
    """
    Computes BD-Rate (%) between reference anchor curve (rate1, dist1) and target model curve (rate2, dist2).
    - Negative (%) = Bitrate savings (target model is better).
    - Positive (%) = Bitrate penalty (target model is worse).
    """
    rate1 = np.asarray(rate1, dtype=np.float64)
    dist1 = np.asarray(dist1, dtype=np.float64)
    rate2 = np.asarray(rate2, dtype=np.float64)
    dist2 = np.asarray(dist2, dtype=np.float64)

    valid1 = rate1 > 0
    valid2 = rate2 > 0
    rate1, dist1 = rate1[valid1], dist1[valid1]
    rate2, dist2 = rate2[valid2], dist2[valid2]

    if len(rate1) < 2 or len(rate2) < 2:
        return np.nan

    dist1_s, log_rate1_s = _make_strictly_increasing(dist1, np.log(rate1))
    dist2_s, log_rate2_s = _make_strictly_increasing(dist2, np.log(rate2))

    if len(dist1_s) < 2 or len(dist2_s) < 2:
        return np.nan

    min_dist = max(np.min(dist1_s), np.min(dist2_s))
    max_dist = min(np.max(dist1_s), np.max(dist2_s))

    if min_dist >= max_dist:
        return np.nan

    if piece_wise:
        interp1 = scipy.interpolate.PchipInterpolator(dist1_s, log_rate1_s)
        interp2 = scipy.interpolate.PchipInterpolator(dist2_s, log_rate2_s)
        
        grid = np.linspace(min_dist, max_dist, 100)
        int1 = scipy.integrate.simpson(y=interp1(grid), x=grid)
        int2 = scipy.integrate.simpson(y=interp2(grid), x=grid)
    else:
        poly1 = np.polyfit(dist1_s, log_rate1_s, min(3, len(dist1_s) - 1))
        poly2 = np.polyfit(dist2_s, log_rate2_s, min(3, len(dist2_s) - 1))

        poly_int1 = np.polyint(poly1)
        poly_int2 = np.polyint(poly2)

        int1 = np.polyval(poly_int1, max_dist) - np.polyval(poly_int1, min_dist)
        int2 = np.polyval(poly_int2, max_dist) - np.polyval(poly_int2, min_dist)

    avg_diff = (int2 - int1) / (max_dist - min_dist)
    return (np.exp(avg_diff) - 1.0) * 100.0


def bd_metric(rate1, dist1, rate2, dist2, piece_wise=True):
    """
    Computes BD-Metric (e.g., BD-PSNR in dB) between reference anchor curve 1 and target model curve 2.
    Positive value means target model achieves higher quality at equivalent bitrates.
    """
    rate1 = np.asarray(rate1, dtype=np.float64)
    dist1 = np.asarray(dist1, dtype=np.float64)
    rate2 = np.asarray(rate2, dtype=np.float64)
    dist2 = np.asarray(dist2, dtype=np.float64)

    valid1 = rate1 > 0
    valid2 = rate2 > 0
    rate1, dist1 = rate1[valid1], dist1[valid1]
    rate2, dist2 = rate2[valid2], dist2[valid2]

    if len(rate1) < 2 or len(rate2) < 2:
        return np.nan

    log_rate1_s, dist1_s = _make_strictly_increasing(np.log(rate1), dist1)
    log_rate2_s, dist2_s = _make_strictly_increasing(np.log(rate2), dist2)

    if len(log_rate1_s) < 2 or len(log_rate2_s) < 2:
        return np.nan

    min_rate = max(np.min(log_rate1_s), np.min(log_rate2_s))
    max_rate = min(np.max(log_rate1_s), np.max(log_rate2_s))

    if min_rate >= max_rate:
        return np.nan

    if piece_wise:
        interp1 = scipy.interpolate.PchipInterpolator(log_rate1_s, dist1_s)
        interp2 = scipy.interpolate.PchipInterpolator(log_rate2_s, dist2_s)
        
        grid = np.linspace(min_rate, max_rate, 100)
        int1 = scipy.integrate.simpson(y=interp1(grid), x=grid)
        int2 = scipy.integrate.simpson(y=interp2(grid), x=grid)
    else:
        poly1 = np.polyfit(log_rate1_s, dist1_s, min(3, len(log_rate1_s) - 1))
        poly2 = np.polyfit(log_rate2_s, dist2_s, min(3, len(log_rate2_s) - 1))

        poly_int1 = np.polyint(poly1)
        poly_int2 = np.polyint(poly2)

        int1 = np.polyval(poly_int1, max_rate) - np.polyval(poly_int1, min_rate)
        int2 = np.polyval(poly_int2, max_rate) - np.polyval(poly_int2, min_rate)

    return (int2 - int1) / (max_rate - min_rate)

if __name__ == "__main__":
    r1 = [0.1, 0.2, 0.4, 0.8]
    d1 = [30.0, 32.5, 35.0, 37.5]
    r2 = [0.08, 0.16, 0.32, 0.64]
    d2 = [30.0, 32.5, 35.0, 37.5]
    print(f"Self-Test BD-Rate: {bd_rate(r1, d1, r2, d2):.2f}%")
    print(f"Self-Test BD-PSNR: {bd_metric(r1, d1, r2, d2):.2f} dB")
