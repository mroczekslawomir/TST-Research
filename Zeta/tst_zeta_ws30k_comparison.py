#!/usr/bin/env python3
"""
COMPLETE TST RIEMANN HYPOTHESIS ANALYSIS FOR WS_30k
- Runs TST simulation for WS_30k network (30,000 nodes, 1200 eigenvalues)
- Downloads Riemann zeta zeros
- Performs comparative analysis
- Generates visualizations and statistics
"""

import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
from scipy.special import erf
from scipy import stats
from scipy.spatial.distance import pdist
import requests
import time
from datetime import datetime
import matplotlib.pyplot as plt
import json

class CompleteTSTAnalysisWS30k:
    def __init__(self):
        self.results = {}
        np.set_printoptions(precision=4, suppress=True)
        
    def create_network(self, network_type, num_nodes, k, p, seed=42):
        """Creates a network of specified type"""
        print(f"Creating {network_type} network with {num_nodes:,} nodes...")
        if network_type == 'WS':
            G = nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)
        elif network_type == 'BA':
            G = nx.barabasi_albert_graph(num_nodes, k, seed=seed)
        else:
            raise ValueError(f"Unknown network type: {network_type}")
        return G

    def build_desynchronization_operator(self, G, params):
        """Builds the TST desynchronization operator"""
        n = G.number_of_nodes()
        degrees = np.array([d for _, d in G.degree()])
        
        # Gradient term (spatial)
        L = nx.laplacian_matrix(G).astype(float)
        
        # Synchronization term with degree influence
        I_sync = diags(params['A'] * (1 + 0.1 * degrees / np.max(degrees)), 0)
        
        # Entropic term with clustering coefficient
        clustering = np.array(list(nx.clustering(G).values()))
        if len(clustering) > 0:
            I_ent = diags(params['C'] * clustering, 0)
        else:
            I_ent = diags(params['C'] * np.ones(n), 0)
        
        return params['B'] * L + I_sync + I_ent

    def compute_eigenvalues(self, operator, num_eigenvalues):
        """Computes eigenvalues of the operator"""
        print(f"Computing {num_eigenvalues} eigenvalues...")
        eigenvalues = eigsh(
            operator,
            k=num_eigenvalues,
            which='SM',
            tol=1e-8,
            maxiter=5000,
            ncv=min(3*num_eigenvalues, operator.shape[0]-1)
        )[0]
        return eigenvalues

    def analyze_spectrum(self, eigenvalues):
        """Analyzes spectrum for Riemann hypothesis properties"""
        eigenvalues_norm = (eigenvalues - np.min(eigenvalues)) / (np.max(eigenvalues) - np.min(eigenvalues))
        
        bands = {
            'strict': [0.49, 0.51],
            'narrow': [0.48, 0.52], 
            'medium': [0.47, 0.53],
            'broad': [0.45, 0.55],
            'very_broad': [0.40, 0.60]
        }
        
        results = {}
        for band_name, (low, high) in bands.items():
            results[band_name] = np.sum((eigenvalues_norm >= low) & (eigenvalues_norm <= high))
        
        # Statistical significance
        total = len(eigenvalues)
        expected = total * 0.1  # Expected 10% for [0.45,0.55]
        
        if results['broad'] > expected:
            z_score = (results['broad'] - expected) / np.sqrt(expected)
            p_value = 2 * (1 - 0.5 * (1 + erf(z_score / np.sqrt(2))))
        else:
            p_value = 1.0
        
        return {
            'eigenvalues_norm': eigenvalues_norm,
            'band_counts': results,
            'p_value': p_value,
            'mean': np.mean(eigenvalues_norm),
            'std': np.std(eigenvalues_norm)
        }

    def download_zeta_zeros(self, n_zeros=100000):
        """Downloads Riemann zeta zeros from Odlyzko's database"""
        print(f"Downloading first {n_zeros} Riemann zeta zeros...")
        try:
            # Download from Odlyzko's website
            file_num = 1 + (n_zeros - 1) // 100000
            url = f"http://www.dtc.umn.edu/~odlyzko/zeta_tables/zeros{file_num}"
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse the zeros
            zeros = []
            for line in response.text.split('\n')[:n_zeros]:
                if line.strip():
                    zeros.append(float(line.strip()))
            
            return np.array(zeros)
        except Exception as e:
            print(f"Error downloading zeta zeros: {e}")
            print("Using approximate zeros for demonstration...")
            # Generate approximate zeros if download fails
            n = np.arange(1, n_zeros + 1)
            # Return only imaginary parts for approximate zeros
            return 14.134725 + n * (2 * np.pi / np.log(n / (2 * np.pi)))

    def calculate_pair_correlation(self, zeros, max_dist=5.0, bin_width=0.1):
        """Calculates pair correlation function"""
        if len(zeros) < 100:
            return None, None
        
        # Check if range is zero to avoid division by zero
        if np.max(zeros) - np.min(zeros) == 0:
            return None, None
        
        distances = pdist(zeros.reshape(-1, 1))
        bins = np.arange(0, max_dist + bin_width, bin_width)
        hist, bin_edges = np.histogram(distances, bins=bins)
        
        # Normalization
        n = len(zeros)
        range_size = np.max(zeros) - np.min(zeros)
        expected_density = n / range_size
        hist_normalized = hist / (0.5 * n * expected_density * bin_width * (n - 1))
        
        return bin_edges[:-1], hist_normalized

    def run_complete_analysis(self):
        """Runs the complete analysis pipeline for WS_30k"""
        print("🌌 TST RIEMANN HYPOTHESIS COMPLETE ANALYSIS FOR WS_30k")
        print("=" * 70)
        
        # 1. Run TST simulation for WS_30k
        print("\n🚀 Step 1: Running TST simulation for WS_30k")
        start_time = time.time()
        
        G = self.create_network('WS', 30000, 4, 0.1)
        operator = self.build_desynchronization_operator(G, {'A': 0.9, 'B': 1.1, 'C': 0.1})
        eigenvalues = self.compute_eigenvalues(operator, 1200)
        tst_analysis = self.analyze_spectrum(eigenvalues)
        
        tst_time = time.time() - start_time
        print(f"✅ TST simulation completed in {tst_time:.2f}s")
        print(f"   Broad band [0.45,0.55]: {tst_analysis['band_counts']['broad']}/1200 ({tst_analysis['band_counts']['broad']/1200*100:.1f}%)")
        print(f"   p-value: {tst_analysis['p_value']:.2e}")
        
        # 2. Download zeta zeros
        print("\n📥 Step 2: Downloading Riemann zeta zeros")
        zeta_zeros = self.download_zeta_zeros(100000)
        print(f"✅ Downloaded {len(zeta_zeros)} zeta zeros")
        
        # 3. Prepare data for comparison
        print("\n📊 Step 3: Preparing data for comparison")
        # Use only the first 1200 zeros for fair comparison
        zeta_imag = zeta_zeros[:1200]  # Already imaginary parts
        
        # Map TST eigenvalues to the same range as zeta zeros
        zeta_min = np.min(zeta_imag)
        zeta_max = np.max(zeta_imag)
        tst_mapped = zeta_min + (zeta_max - zeta_min) * tst_analysis['eigenvalues_norm']
        
        # 4. Statistical comparison
        print("\n🔍 Step 4: Statistical comparison")
        # KS test
        ks_stat, ks_pvalue = stats.ks_2samp(zeta_imag, tst_mapped)
        
        # Gap analysis
        zeta_gaps = np.diff(zeta_imag)
        tst_gaps = np.diff(tst_mapped)
        
        # Pair correlation
        zeta_bins, zeta_corr = self.calculate_pair_correlation(zeta_imag)
        tst_bins, tst_corr = self.calculate_pair_correlation(tst_mapped)
        
        # 5. Save results
        print("\n💾 Step 5: Saving results")
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tst_simulation': {
                'nodes': 30000,
                'eigenvalues': 1200,
                'broad_band_count': int(tst_analysis['band_counts']['broad']),
                'broad_band_percentage': float(tst_analysis['band_counts']['broad']/1200*100),
                'p_value': float(tst_analysis['p_value']),
                'computation_time': float(tst_time)
            },
            'zeta_zeros': {
                'count': len(zeta_zeros),
                'imag_range': [float(np.min(zeta_imag)), float(np.max(zeta_imag))]
            },
            'comparison': {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'zeta_mean_gap': float(np.mean(zeta_gaps)) if len(zeta_gaps) > 0 else 0,
                'tst_mean_gap': float(np.mean(tst_gaps)) if len(tst_gaps) > 0 else 0,
                'zeta_std_gap': float(np.std(zeta_gaps)) if len(zeta_gaps) > 0 else 0,
                'tst_std_gap': float(np.std(tst_gaps)) if len(tst_gaps) > 0 else 0
            }
        }
        
        # Add pair correlation results if available
        if zeta_corr is not None and tst_corr is not None:
            min_len = min(len(zeta_corr), len(tst_corr))
            rms_diff = np.sqrt(np.mean((zeta_corr[:min_len] - tst_corr[:min_len])**2))
            self.results['comparison']['pair_correlation_rms_diff'] = float(rms_diff)
        
        with open('tst_zeta_ws30k_complete_analysis.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 6. Create visualizations
        print("\n📈 Step 6: Creating visualizations")
        self.create_visualizations(zeta_imag, tst_mapped, zeta_bins, zeta_corr, tst_bins, tst_corr)
        
        total_time = time.time() - start_time
        print(f"\n✅ Analysis completed in {total_time:.2f}s")
        print(f"💾 Results saved to 'tst_zeta_ws30k_complete_analysis.json'")
        
        return self.results

    def create_visualizations(self, zeta_imag, tst_mapped, zeta_bins, zeta_corr, tst_bins, tst_corr):
        """Creates comparison visualizations"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Distribution of imaginary parts
        ax1.hist(zeta_imag, bins=50, alpha=0.7, density=True, label='Zeta Zeros')
        ax1.hist(tst_mapped, bins=50, alpha=0.7, density=True, label='TST Mapped')
        ax1.set_xlabel('Imaginary Part')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.set_title('Distribution Comparison - WS_30k')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pair correlation
        if zeta_corr is not None and tst_corr is not None:
            ax2.plot(zeta_bins, zeta_corr, 'b-', label='Zeta Zeros', linewidth=2)
            ax2.plot(tst_bins, tst_corr, 'r--', label='TST Mapped', linewidth=2)
            ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.7)
            ax2.set_xlabel('Distance')
            ax2.set_ylabel('Pair Correlation')
            ax2.legend()
            ax2.set_title('Pair Correlation Function - WS_30k')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, 10)
        
        # Plot 3: Gap distribution
        zeta_gaps = np.diff(zeta_imag)
        tst_gaps = np.diff(tst_mapped)
        
        if len(zeta_gaps) > 0 and len(tst_gaps) > 0:
            ax3.hist(zeta_gaps, bins=50, alpha=0.7, density=True, label='Zeta Zeros')
            ax3.hist(tst_gaps, bins=50, alpha=0.7, density=True, label='TST Mapped')
            ax3.set_xlabel('Gap Size')
            ax3.set_ylabel('Density')
            ax3.legend()
            ax3.set_title('Gap Distribution - WS_30k')
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(0, 5)
        
        # Plot 4: Cumulative distribution
        zeta_sorted = np.sort(zeta_imag)
        tst_sorted = np.sort(tst_mapped)
        
        ax4.plot(zeta_sorted, np.arange(len(zeta_sorted))/len(zeta_sorted), 
                'b-', label='Zeta Zeros', linewidth=2)
        ax4.plot(tst_sorted, np.arange(len(tst_sorted))/len(tst_sorted),
                'r--', label='TST Mapped', linewidth=2)
        ax4.set_xlabel('Imaginary Part')
        ax4.set_ylabel('Cumulative Probability')
        ax4.legend()
        ax4.set_title('Cumulative Distribution - WS_30k')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tst_zeta_ws30k_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations saved to 'tst_zeta_ws30k_comparison.png'")

    def print_summary(self):
        """Prints a summary of the results"""
        if not self.results:
            print("No results to summarize. Run analysis first.")
            return
        
        print("\n🎯 COMPLETE ANALYSIS SUMMARY FOR WS_30k")
        print("=" * 60)
        
        tst = self.results['tst_simulation']
        comp = self.results['comparison']
        
        print(f"TST Simulation (WS_30k):")
        print(f"  • Broad band [0.45,0.55]: {tst['broad_band_count']}/1200 ({tst['broad_band_percentage']:.1f}%)")
        print(f"  • p-value: {tst['p_value']:.2e}")
        print(f"  • Computation time: {tst['computation_time']:.2f}s")
        
        print(f"\nStatistical Comparison:")
        print(f"  • KS statistic: {comp['ks_statistic']:.4f}")
        print(f"  • KS p-value: {comp['ks_pvalue']:.2e}")
        
        # Avoid division by zero
        if comp['zeta_mean_gap'] != 0:
            ratio = comp['tst_mean_gap'] / comp['zeta_mean_gap']
            print(f"  • Mean gap ratio (TST/Zeta): {ratio:.4f}")
        else:
            print(f"  • Mean gap ratio (TST/Zeta): undefined (zeta mean gap is zero)")
        
        if comp['ks_pvalue'] > 0.05:
            print(f"  ✅ Distributions are statistically indistinguishable")
        else:
            print(f"  ⚠️  Distributions are statistically different")
        
        if 'pair_correlation_rms_diff' in comp:
            print(f"  • Pair correlation RMS difference: {comp['pair_correlation_rms_diff']:.4f}")

# Run the complete analysis
if __name__ == "__main__":
    analyzer = CompleteTSTAnalysisWS30k()
    results = analyzer.run_complete_analysis()
    analyzer.print_summary()