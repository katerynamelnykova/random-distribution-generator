import numpy as np
from scipy.stats import chi2, norm, uniform, expon

class DistributionsTests:
    def calculate_mean_var(self, data: list, dist_type: str, params: tuple):
        if dist_type == 'normal':
            mu, sigma = params
            E = mu
            var_th = sigma**2
        elif dist_type == 'uniform':
            a, b = params
            E = (a + b)/2
            var_th = (b - a)**2 / 12
        elif dist_type == 'exponential':
            la = params[0]
            E = 1/la
            var_th = 1/(la**2)
            
        mean = np.mean(data)
        var = np.var(data)
        mean_deviation = ((mean - E) / E)
        variance_deviation = ((var - var_th) / var_th)
        return mean_deviation, variance_deviation, (mean, var, E, var_th)

    def chi_squared_test(self, data: list, dist_type: str, params: tuple, bins: int = 30):
        hist, bin_edges = np.histogram(data, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        expected_freq = []

        if dist_type == 'normal':
            mu, sigma = params
            expected_freq = [norm.pdf(x, mu, sigma) for x in bin_centers]
            p = 2
        elif dist_type == 'uniform':
            a, b = params
            expected_freq = [uniform.pdf(x, a, b - a) for x in bin_centers]
            p = 2
        elif dist_type == 'exponential':
            la = params[0]
            expected_freq = [expon.pdf(x, scale=1/la) for x in bin_centers]
            p = 1

        chi2_stat = sum((o - e)**2 / e for o, e in zip(hist, expected_freq))

        # Кількість ступенів свободи
        df = bins - 1 - p

        # Табличне значення χ² для рівня значущості 0.05
        chi2_critical = chi2.ppf(0.95, df)

        if chi2_stat < chi2_critical:
            result = f"χ² = {chi2_stat:.4f} < χ²_кр = {chi2_critical:.4f} → Розподіл відповідає гіпотезі."
            desigion = True
        else:
            result = f"χ² = {chi2_stat:.4f} ≥ χ²_кр = {chi2_critical:.4f} → Розподіл не відповідає гіпотезі."
            desigion = False
        
        return desigion, (result, chi2_stat, chi2_critical)

    def lambda_criterion(self, data: list, dist_type: str, params: tuple):
        n = len(data)
        data_sorted = np.sort(data)
        empirical_cdf = np.arange(1, n + 1) / n

        if dist_type == 'normal':
            mu, sigma = params
            theoretical_cdf = norm.cdf(data_sorted, mu, sigma)
        elif dist_type == 'uniform':
            a, b = params
            theoretical_cdf = uniform.cdf(data_sorted, a, b - a)
        elif dist_type == 'exponential':
            la = params[0]
            theoretical_cdf = expon.cdf(data_sorted, scale=1/la)

        lambda_stat = np.max(np.abs(empirical_cdf - theoretical_cdf)) * np.sqrt(n)
        lambda_critical = 1.36   # для рівня значущості 0.05

        if lambda_stat < lambda_critical:
            result = f"λ = {lambda_stat:.4f} < λ_кр = {lambda_critical:.4f} → Розподіл відповідає гіпотезі."
            desigion = True
        else:
            result = f"λ = {lambda_stat:.4f} ≥ λ_кр = {lambda_critical:.4f} → Розподіл не відповідає гіпотезі."
            desigion = False

        return desigion, (result, lambda_stat, lambda_critical)