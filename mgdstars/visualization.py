"""
Visualization tools for MGD Stars analysis.

This module provides comprehensive plotting and animation capabilities
for analyzing MGD star configurations, including:
- Physical property profiles
- Model comparisons
- Convergence diagnostics
- Interactive plots
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Union
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class StarPlotter:
    """
    Comprehensive plotting class for MGD star analysis.
    
    Provides methods to visualize physical properties, compare models,
    and create publication-quality figures.
    """
    
    def __init__(self, star):
        """
        Initialize the plotter with an MGD star instance.
        
        Parameters:
        -----------
        star : MGDStars
            Solved MGD star instance
        """
        self.star = star
        if not star.solution_found:
            raise ValueError("Star must be solved before plotting. Call solve_star_configuration() first.")
    
    def plot_profiles(self, r_range=None, properties=None, figsize=(15, 10)):
        """
        Plot physical property profiles.
        
        Parameters:
        -----------
        r_range : array-like, optional
            Radius values for evaluation
        properties : list, optional
            Properties to plot ['rho', 'pr', 'pt', 'anisotropy']
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        fig, axes : matplotlib objects
        """
        if r_range is None:
            # Use sensible default range
            R_max = 12.0  # Assume typical neutron star radius
            r_range = np.linspace(0.1, R_max * 0.95, 100)
        
        if properties is None:
            properties = ['rho', 'pr', 'pt', 'anisotropy']
        
        # Evaluate physical quantities
        quantities = self.star.evaluate_at_radius(r_range)
        
        # Create subplots
        n_plots = len(properties)
        n_cols = 2
        n_rows = (n_plots + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        plot_configs = {
            'rho': {
                'data': quantities['rho'],
                'ylabel': r'Energy Density $\rho(r)$',
                'title': 'Energy Density Profile',
                'color': 'blue'
            },
            'pr': {
                'data': quantities['pr'],
                'ylabel': r'Radial Pressure $p_r(r)$',
                'title': 'Radial Pressure Profile',
                'color': 'red'
            },
            'pt': {
                'data': quantities['pt'],
                'ylabel': r'Tangential Pressure $p_t(r)$',
                'title': 'Tangential Pressure Profile',
                'color': 'green'
            },
            'anisotropy': {
                'data': quantities['pt'] - quantities['pr'],
                'ylabel': r'Anisotropy $\Delta = p_t - p_r$',
                'title': 'Pressure Anisotropy Profile',
                'color': 'purple'
            }
        }
        
        for i, prop in enumerate(properties):
            if i < len(axes):
                config = plot_configs[prop]
                axes[i].plot(r_range, config['data'], 
                           color=config['color'], linewidth=2.5, alpha=0.8)
                axes[i].set_xlabel('Radius (km)')
                axes[i].set_ylabel(config['ylabel'])
                axes[i].set_title(config['title'], fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                
                if prop == 'anisotropy':
                    axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Hide unused subplots
        for i in range(len(properties), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_model_comparison(self, other_stars, r_range=None, figsize=(12, 8)):
        """
        Compare this star with other models.
        
        Parameters:
        -----------
        other_stars : dict
            Dictionary of {model_name: MGDStars_instance}
        r_range : array-like, optional
            Radius values for comparison
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig, axes : matplotlib objects
        """
        if r_range is None:
            r_range = np.linspace(0.1, 11.9, 100)
        
        # Include this star in comparison
        all_stars = {'Current': self.star}
        all_stars.update(other_stars)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(all_stars)))
        
        properties = ['rho', 'pr', 'pt', 'anisotropy']
        titles = ['Energy Density', 'Radial Pressure', 'Tangential Pressure', 'Anisotropy']
        ylabels = [r'$\rho(r)$', r'$p_r(r)$', r'$p_t(r)$', r'$\Delta(r)$']
        
        for name, star in all_stars.items():
            if star.solution_found:
                quantities = star.evaluate_at_radius(r_range)
                
                for i, prop in enumerate(properties):
                    if prop == 'anisotropy':
                        data = quantities['pt'] - quantities['pr']
                    else:
                        data = quantities[prop]
                    
                    axes[i].plot(r_range, data, label=f'{name} ({star.seed_model})', 
                               linewidth=2, alpha=0.8)
        
        for i, (title, ylabel) in enumerate(zip(titles, ylabels)):
            axes[i].set_xlabel('Radius (km)')
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(title, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            if i == 3:  # Anisotropy plot
                axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig, axes
    
    def plot_convergence_metrics(self, solve_result, figsize=(10, 6)):
        """
        Plot convergence metrics and diagnostics.
        
        Parameters:
        -----------
        solve_result : dict
            Result from solve_star_configuration
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig, axes : matplotlib objects
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Solution parameters
        params = ['A0', 'B0', 'C0']
        values = [solve_result[param] for param in params]
        
        bars = axes[0].bar(params, values, alpha=0.7, color=['blue', 'red', 'green'])
        axes[0].set_title('Solution Parameters', fontweight='bold')
        axes[0].set_ylabel('Parameter Value')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[0].annotate(f'{value:.4f}', 
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", 
                           ha='center', va='bottom')
        
        # Plot 2: Convergence info
        info_text = f"""
        Convergence: {'✓ SUCCESS' if solve_result['success'] else '✗ FAILED'}
        Residual: {solve_result.get('residual', 'N/A'):.2e}
        Attempts: {solve_result.get('attempts', 'N/A')}
        Seed Model: {self.star.seed_model}
        Alpha: {self.star.alpha}
        """
        
        axes[1].text(0.1, 0.5, info_text, transform=axes[1].transAxes, 
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
        axes[1].set_title('Convergence Information', fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig, axes


class AnimationMaker:
    """
    Create animations of MGD star evolution and parameter sweeps.
    """
    
    def __init__(self, star):
        """
        Initialize the animation maker.
        
        Parameters:
        -----------
        star : MGDStars
            MGD star instance
        """
        self.star = star
    
    def create_parameter_sweep_animation(self, param_name, param_range, 
                                       M_val, R_val, save_path=None):
        """
        Create animation showing how profiles change with parameter variation.
        
        Parameters:
        -----------
        param_name : str
            Parameter to vary ('alpha')
        param_range : array-like
            Range of parameter values
        M_val, R_val : float
            Fixed mass and radius values
        save_path : str, optional
            Path to save animation
            
        Returns:
        --------
        animation : matplotlib animation object
        """
        from matplotlib.animation import FuncAnimation
        
        # This is a template - full implementation would require
        # frame-by-frame calculation and plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        
        def animate(frame):
            ax.clear()
            # Update parameter and solve
            if param_name == 'alpha':
                self.star.alpha = param_range[frame]
            
            # Solve and plot (simplified)
            try:
                result = self.star.solve_star_configuration(M_val, R_val, verbose=False)
                if result['success']:
                    r_range = np.linspace(0.1, R_val * 0.95, 50)
                    quantities = self.star.evaluate_at_radius(r_range)
                    ax.plot(r_range, quantities['rho'], 'b-', linewidth=2)
                    ax.set_title(f'α = {param_range[frame]:.4f}')
                    ax.set_xlabel('Radius (km)')
                    ax.set_ylabel('Density')
                    ax.grid(True, alpha=0.3)
            except:
                ax.text(0.5, 0.5, 'Failed to converge', 
                       transform=ax.transAxes, ha='center')
        
        animation = FuncAnimation(fig, animate, frames=len(param_range), 
                                 interval=500, repeat=True)
        
        if save_path:
            animation.save(save_path, writer='pillow')
        
        return animation


def quick_plot(star, r_range=None, property_name='rho'):
    """
    Convenience function for quick plotting.
    
    Parameters:
    -----------
    star : MGDStars
        Solved MGD star instance
    r_range : array-like, optional
        Radius values
    property_name : str
        Property to plot ('rho', 'pr', 'pt', 'anisotropy')
        
    Returns:
    --------
    fig, ax : matplotlib objects
    """
    if not star.solution_found:
        raise ValueError("Star must be solved first")
    
    if r_range is None:
        r_range = np.linspace(0.1, 11.9, 100)
    
    quantities = star.evaluate_at_radius(r_range)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if property_name == 'anisotropy':
        data = quantities['pt'] - quantities['pr']
        ylabel = r'Anisotropy $\Delta = p_t - p_r$'
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    else:
        data = quantities[property_name]
        ylabel = f'{property_name.capitalize()}(r)'
    
    ax.plot(r_range, data, linewidth=2.5, alpha=0.8)
    ax.set_xlabel('Radius (km)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{property_name.capitalize()} Profile ({star.seed_model})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

