import sympy as sp
import numpy as np
from scipy.optimize import fsolve, root
import warnings

class MGDStars:
    """
    Minimal Geometrical Deformation (MGD) Stars solver class.
    
    This class handles the symbolic computation and numerical solution
    of MGD star configurations with boundary conditions.
    """
    
    def __init__(self, alpha=0.1, k2_factor=8*np.pi, seed_model='TolmanIV'):
        """
        Initialize the MGD Stars solver.
        
        Parameters:
        -----------
        alpha : float
            MGD coupling parameter
        k2_factor : float
            Factor for k2 (default: 8π)
        """
        self.alpha = alpha
        self.k2_factor = k2_factor
        self.seed_model = seed_model

        # Store solved parameters
        self.A0_solved = None
        self.B0_solved = None
        self.C0_solved = None
        self.solution_found = False
        
        # Initialize symbolic variables and expressions
        self._setup_symbolic_variables()
        self._derive_expressions(seed_model)
        self._create_numerical_functions()
    
    def _setup_symbolic_variables(self):
        """Setup all symbolic variables used in the computation."""
        self.r, self.A0, self.B0, self.C0 = sp.symbols('r A0 B0 C0', real=True, positive=True)
        self.k2 = sp.symbols('k2', real=True, positive=True)
        self.alpha_sym = sp.symbols('alpha', real=True, positive=True)
        self.M, self.R = sp.symbols('M R', real=True, positive=True)
    
    def _derive_expressions(self, seed_model='TolmanIV'):
        """Derive all the symbolic expressions for the MGD star."""
        print("Deriving symbolic expressions...")
        
        if seed_model == 'TolmanIV':
            # Metric components
            self.e_nu = self.B0**2 * (self.r**2 / self.A0**2 + 1)
            self.nu = sp.log(self.e_nu)
            
            # Calculate derivatives of nu
            self.nu_prime = sp.diff(self.nu, self.r)
            self.nu_double_prime = sp.diff(self.nu_prime, self.r)
            
            # Define mu without f_expr dependency initially
            self.mu_base = (self.A0**2 + self.r**2) * (self.C0**2 - self.r**2) / (self.C0**2 * (self.A0**2 + 2 * self.r**2))

        elif seed_model == 'HeintzmannIIa':
            # Metric components
            self.e_nu = self.A0**2 * (self.B0 * self.r**2 + 1)**3
            self.nu = sp.log(self.e_nu)
            
            # Calculate derivatives of nu
            self.nu_prime = sp.diff(self.nu, self.r)
            self.nu_double_prime = sp.diff(self.nu_prime, self.r)
            
            # Original HeintzmannIIa mu_base expression
            self.mu_base = (1 - 3 * self.B0 * self.r**2 * 
                        (1 + self.C0/sp.sqrt(4 * self.B0 * self.r**2 + 1)) /
                        (2 * (self.B0 * self.r**2 + 1)))
        
        elif seed_model == 'WymanIIa':
            # Metric components  
            self.e_nu = (self.A0 - self.B0 * self.r**2)**2
            self.nu = sp.log(self.e_nu)
            
            # Calculate derivatives of nu
            self.nu_prime = sp.diff(self.nu, self.r)
            self.nu_double_prime = sp.diff(self.nu_prime, self.r)
            
            # Original WymanIIa mu_base expression with safety checks
            # Use Abs to handle potential negative values safely
            self.mu_base = 1 + self.C0 * self.r**2 / sp.Abs(self.A0 - 3 * self.B0 * self.r**2)**(sp.Rational(2,3))

        else:
            raise ValueError(f"Unknown seed model: {seed_model}")
        
        # Define f_expr using the base mu
        self.f_expr = -self.mu_base + 1/(1 + self.nu_prime * self.r)
        self.f_prime = sp.diff(self.f_expr, self.r)
        
        # Complete mu definition
        self.mu = self.mu_base - self.alpha_sym * self.f_expr
        self.mu_prime = sp.diff(self.mu, self.r)
        
        # Einstein tensor components (GR part)
        self.rho = ((1 - self.mu)/self.r**2 - self.mu_prime/self.r)/self.k2
        self.pr = (-1/self.r**2 + self.mu * (1/self.r**2 + self.nu_prime/self.r))/self.k2
        self.pt = ((self.mu/4)*(2*self.nu_double_prime + self.nu_prime**2 + 2*self.nu_prime/self.r) + 
                  (self.mu_prime/4)*(self.nu_prime + 2/self.r))/self.k2
        
        # MGD components (theta)
        self.theta_00 = -(self.f_expr / self.r**2 + self.f_prime / self.r)/self.k2
        self.theta_11 = -self.f_expr * (1/self.r**2 + self.nu_prime / self.r)/self.k2
        self.theta_22 = (-(self.f_expr/4)*(2*self.nu_double_prime + self.nu_prime**2 + 2*self.nu_prime/self.r) - 
                        (self.f_prime/4)*(self.nu_prime + 2/self.r))/self.k2
        
        print("Symbolic expressions derived successfully!")

    
    def _create_numerical_functions(self):
        """Create numerical functions from symbolic expressions."""
        print("Creating numerical functions...")
        
        # Evaluate all derivatives and substitute alpha before lambdification
        nu_prime_eval = self.nu_prime.doit()
        nu_double_prime_eval = self.nu_double_prime.doit()
        mu_prime_eval = self.mu_prime.doit()
        f_prime_eval = self.f_prime.doit()
        
        # Substitute alpha value and k2
        rho_eval = self.rho.doit().subs([(self.alpha_sym, self.alpha), (self.k2, self.k2_factor)])
        pr_eval = self.pr.doit().subs([(self.alpha_sym, self.alpha), (self.k2, self.k2_factor)])
        pt_eval = self.pt.doit().subs([(self.alpha_sym, self.alpha), (self.k2, self.k2_factor)])
        mu_eval = self.mu.subs(self.alpha_sym, self.alpha)
        
        # Convert to numerical functions
        self.numerical_rho = sp.lambdify((self.r, self.A0, self.B0, self.C0), rho_eval, 'numpy')
        self.numerical_pr = sp.lambdify((self.r, self.A0, self.B0, self.C0), pr_eval, 'numpy')
        self.numerical_pt = sp.lambdify((self.r, self.A0, self.B0, self.C0), pt_eval, 'numpy')
        self.numerical_e_nu = sp.lambdify((self.r, self.A0, self.B0), self.e_nu, 'numpy')
        self.numerical_mu = sp.lambdify((self.r, self.A0, self.B0, self.C0), mu_eval, 'numpy')
        
        print("Numerical functions created successfully!")
    
    def _boundary_conditions(self, params, M_val, R_val):
        """Calculate boundary condition residuals."""
        A0_val, B0_val, C0_val = params
        r_val = R_val
        
        try:
            # Boundary condition 1: pr(R) = 0
            pr_R_val = self.numerical_pr(r_val, A0_val, B0_val, C0_val)
            
            # Boundary condition 2: mu(R) = 1 - 2M/R - alpha*f(R)
            mu_R_val = self.numerical_mu(r_val, A0_val, B0_val, C0_val)
            mu_expected = 1 - 2*M_val/R_val
            
            # Boundary condition 3: e^nu(R) = 1 - 2M/R
            e_nu_R_val = self.numerical_e_nu(r_val, A0_val, B0_val)
            e_nu_expected = 1 - 2*M_val/R_val
            
            return [
                pr_R_val,  # Should be 0
                mu_R_val - mu_expected,  # Should be 0
                e_nu_R_val - e_nu_expected  # Should be 0
            ]
        except Exception as e:
            print(f"Error in boundary_conditions: {e}")
            return [1e6, 1e6, 1e6]  # Large values to indicate failure
    
    def _generate_initial_guesses(self, M_val, R_val):
        """Generate multiple physically motivated initial guesses."""
        guesses = []
        
        # Model-specific parameter scaling based on convergence testing
        if self.seed_model == 'TolmanIV':
            # TolmanIV is robust - use standard scaling
            A0_base = R_val
            B0_base = 1.0 / (M_val + 1e-10)
            C0_base = R_val * 1.5
            
        elif self.seed_model == 'HeintzmannIIa':
            # HeintzmannIIa works better with smaller A0, very small B0
            A0_base = 0.8  # Much smaller than R_val
            B0_base = 0.0004  # Very small positive value
            C0_base = 2.3  # Moderate value
            
        elif self.seed_model == 'WymanIIa':
            # WymanIIa needs very specific parameter ranges
            A0_base = 0.84  # Small positive value
            B0_base = -0.0006 if np.random.rand() > 0.5 else 0.0006  # Can be negative
            C0_base = -0.002 if np.random.rand() > 0.5 else 0.002  # Can be negative
            
        else:
            # Default scaling for unknown models
            A0_base = R_val
            B0_base = 1.0 / (M_val + 1e-10)
            C0_base = R_val * 1.5
        
        # Strategy 1: Base guess
        guesses.append([A0_base, B0_base, C0_base])
        
        # Strategy 2: Small perturbations around base
        np.random.seed(42)  # For reproducibility
        for i in range(4):
            if self.seed_model in ['HeintzmannIIa', 'WymanIIa']:
                # Small perturbations for sensitive models
                perturbation = 1.0 + 0.1 * (np.random.rand(3) - 0.5)
            else:
                # Larger perturbations for robust models
                perturbation = 1.0 + 0.3 * (np.random.rand(3) - 0.5)
                
            perturbed_guess = [A0_base * perturbation[0], 
                             B0_base * perturbation[1], 
                             C0_base * perturbation[2]]
            guesses.append(perturbed_guess)
        
        # Strategy 3: Conservative fallback for all models
        guesses.append([0.5 * R_val, 0.1, 0.5 * R_val])
        
        # Strategy 4: Original default
        guesses.append([1.2, 0.05, 0.1])
        
        return guesses
    
    def _check_convergence(self, residuals, tolerance=1e-6):
        """Check if the solution has converged to acceptable accuracy."""
        residual_norm = np.linalg.norm(residuals)
        max_residual = np.max(np.abs(residuals))
        
        return {
            'converged': residual_norm < tolerance and max_residual < tolerance * 10,
            'residual_norm': residual_norm,
            'max_residual': max_residual,
            'tolerance': tolerance
        }
    
    def _validate_solution(self, params, M_val, R_val):
        """Validate that the solution is physically reasonable."""
        A0_val, B0_val, C0_val = params
        
        # Check for NaN or infinite values
        if not all(np.isfinite([A0_val, B0_val, C0_val])):
            return False, "Solution contains NaN or infinite values"
        
        # Check positivity constraints where expected
        if A0_val <= 0:
            return False, "A0 must be positive"
        
        # Check that parameters are within reasonable bounds
        if abs(A0_val) > 1e6 or abs(B0_val) > 1e6 or abs(C0_val) > 1e6:
            return False, "Parameters are unreasonably large"
        
        try:
            # Test evaluation at a few points to check for numerical issues
            test_radii = np.array([0.1 * R_val, 0.5 * R_val, 0.9 * R_val])
            
            # Check that physical quantities can be computed
            rho_test = self.numerical_rho(test_radii, A0_val, B0_val, C0_val)
            pr_test = self.numerical_pr(test_radii, A0_val, B0_val, C0_val)
            
            if not np.all(np.isfinite(rho_test)) or not np.all(np.isfinite(pr_test)):
                return False, "Physical quantities contain NaN or infinite values"
            
            # Check for reasonable density values (should be positive)
            if np.any(rho_test < 0):
                return False, "Density becomes negative"
                
        except Exception as e:
            return False, f"Error evaluating physical quantities: {e}"
        
        return True, "Solution is physically valid"
    
    def diagnose_convergence_failure(self, M_val, R_val, tolerance=1e-8):
        """
        Diagnose why the solver is failing to converge for given parameters.
        
        Returns:
        --------
        dict : Diagnostic information about convergence issues
        """
        diagnostics = {
            'parameter_scaling': {},
            'boundary_condition_analysis': {},
            'initial_guess_quality': {},
            'recommendations': []
        }
        
        # Analyze parameter scaling
        diagnostics['parameter_scaling']['M_R_ratio'] = M_val / R_val
        diagnostics['parameter_scaling']['compactness'] = 2 * M_val / R_val
        
        if diagnostics['parameter_scaling']['compactness'] > 0.8:
            diagnostics['recommendations'].append("High compactness (>0.8) may cause numerical issues")
        
        # Test different initial guesses
        initial_guesses = self._generate_initial_guesses(M_val, R_val)
        residual_norms = []
        
        for i, guess in enumerate(initial_guesses[:3]):  # Test first 3 guesses
            try:
                residuals = self._boundary_conditions(guess, M_val, R_val)
                residual_norm = np.linalg.norm(residuals)
                residual_norms.append(residual_norm)
            except:
                residual_norms.append(np.inf)
        
        diagnostics['initial_guess_quality']['residual_norms'] = residual_norms
        diagnostics['initial_guess_quality']['best_initial_norm'] = min(residual_norms)
        
        if min(residual_norms) > 1.0:
            diagnostics['recommendations'].append("All initial guesses show large residuals - consider parameter rescaling")
        
        # Analyze boundary conditions separately
        try:
            best_guess_idx = np.argmin(residual_norms)
            best_guess = initial_guesses[best_guess_idx]
            A0_val, B0_val, C0_val = best_guess
            
            # Individual boundary condition analysis
            pr_R = self.numerical_pr(R_val, A0_val, B0_val, C0_val)
            mu_R = self.numerical_mu(R_val, A0_val, B0_val, C0_val)
            mu_expected = 1 - 2*M_val/R_val
            e_nu_R = self.numerical_e_nu(R_val, A0_val, B0_val)
            e_nu_expected = 1 - 2*M_val/R_val
            
            diagnostics['boundary_condition_analysis'] = {
                'pressure_at_surface': abs(pr_R),
                'mu_error': abs(mu_R - mu_expected),
                'e_nu_error': abs(e_nu_R - e_nu_expected)
            }
            
            # Find the most problematic boundary condition
            errors = [abs(pr_R), abs(mu_R - mu_expected), abs(e_nu_R - e_nu_expected)]
            condition_names = ['pressure', 'mu', 'e_nu']
            worst_condition = condition_names[np.argmax(errors)]
            diagnostics['boundary_condition_analysis']['worst_condition'] = worst_condition
            
        except Exception as e:
            diagnostics['boundary_condition_analysis']['error'] = str(e)
        
        # Generate recommendations
        if diagnostics['parameter_scaling']['compactness'] < 0.1:
            diagnostics['recommendations'].append("Very low compactness - consider using simpler approximations")
        
        return diagnostics
    
    def solve_star_configuration(self, M_val, R_val, initial_guess=None, 
                               tolerance=1e-8, max_attempts=10, verbose=True):
        """
        Solve for the star configuration parameters with improved numerical stability.
        
        Parameters:
        -----------
        M_val : float
            Mass value
        R_val : float  
            Radius value
        initial_guess : list, optional
            Initial guess for [A0, B0, C0]. If None, multiple strategies will be tried.
        tolerance : float
            Convergence tolerance for residuals
        max_attempts : int
            Maximum number of solving attempts with different strategies
        verbose : bool
            Whether to print detailed solving information
        
        Returns:
        --------
        dict : Solution result with convergence information
        """
        
        if verbose:
            print(f"Solving system with M={M_val}, R={R_val}, alpha={self.alpha}")
            print(f"Target tolerance: {tolerance}")
        
        best_solution = None
        best_residual_norm = np.inf
        best_convergence = None
        attempts_made = 0
        
        # Determine initial guesses to try
        if initial_guess is not None:
            initial_guesses = [initial_guess]
        else:
            initial_guesses = self._generate_initial_guesses(M_val, R_val)
        
        # Try different solver methods and configurations
        solver_configs = [
            {'method': 'fsolve', 'options': {'xtol': tolerance, 'maxfev': 1000}},
            {'method': 'fsolve', 'options': {'xtol': tolerance * 10, 'maxfev': 2000}},
            {'method': 'root', 'solver': 'hybr', 'options': {'xtol': tolerance}},
            {'method': 'root', 'solver': 'lm', 'options': {'xtol': tolerance}},
        ]
        
        for attempt, guess in enumerate(initial_guesses):
            if attempts_made >= max_attempts:
                break
                
            for config in solver_configs:
                if attempts_made >= max_attempts:
                    break
                    
                attempts_made += 1
                
                try:
                    if verbose and attempts_made <= 3:
                        print(f"Attempt {attempts_made}: trying {config['method']} with guess {guess}")
                    
                    # Suppress warnings temporarily for cleaner output
                    with warnings.catch_warnings():
                        if not verbose:
                            warnings.simplefilter("ignore")
                        
                        if config['method'] == 'fsolve':
                            solution = fsolve(self._boundary_conditions, guess, 
                                            args=(M_val, R_val), **config['options'])
                        else:
                            result = root(self._boundary_conditions, guess, 
                                        args=(M_val, R_val), method=config['solver'], 
                                        options=config['options'])
                            solution = result.x
                    
                    # Validate the solution
                    is_valid, validation_msg = self._validate_solution(solution, M_val, R_val)
                    if not is_valid:
                        if verbose and attempts_made <= 3:
                            print(f"  Invalid solution: {validation_msg}")
                        continue
                    
                    # Check convergence
                    residuals = self._boundary_conditions(solution, M_val, R_val)
                    convergence_info = self._check_convergence(residuals, tolerance)
                    
                    if verbose and attempts_made <= 3:
                        print(f"  Residual norm: {convergence_info['residual_norm']:.2e}")
                    
                    # Keep track of the best solution found so far
                    if convergence_info['residual_norm'] < best_residual_norm:
                        best_solution = solution.copy()
                        best_residual_norm = convergence_info['residual_norm']
                        best_convergence = convergence_info
                        
                        # If we found a good enough solution, we can stop early
                        if convergence_info['converged']:
                            if verbose:
                                print(f"  ✓ Converged solution found on attempt {attempts_made}")
                            break
                            
                except Exception as e:
                    if verbose and attempts_made <= 3:
                        print(f"  Solver failed: {e}")
                    continue
            
            # If we found a converged solution, break out of the guess loop too
            if best_convergence and best_convergence['converged']:
                break
        
        # Finalize results
        if best_solution is not None:
            self.A0_solved, self.B0_solved, self.C0_solved = best_solution
            final_residuals = self._boundary_conditions(best_solution, M_val, R_val)
            self.solution_found = best_convergence['converged']
            
            result = {
                'A0': self.A0_solved,
                'B0': self.B0_solved, 
                'C0': self.C0_solved,
                'residuals': final_residuals,
                'success': best_convergence['converged'],
                'residual_norm': best_convergence['residual_norm'],
                'max_residual': best_convergence['max_residual'],
                'tolerance_used': tolerance,
                'attempts_made': attempts_made,
                'converged': best_convergence['converged']
            }
            
            if verbose:
                print(f"\nFinal solution after {attempts_made} attempts:")
                print(f"A0 = {self.A0_solved:.6f}")
                print(f"B0 = {self.B0_solved:.6f}")
                print(f"C0 = {self.C0_solved:.6f}")
                print(f"Residual norm: {best_convergence['residual_norm']:.2e}")
                print(f"Max residual: {best_convergence['max_residual']:.2e}")
                print(f"Converged: {best_convergence['converged']}")
                if not best_convergence['converged']:
                    print(f"Warning: Solution did not meet convergence criteria (tolerance: {tolerance:.2e})")
                    # Run diagnostics for failed convergence
                    print("\nRunning convergence diagnostics...")
                    diagnostics = self.diagnose_convergence_failure(M_val, R_val, tolerance)
                    print(f"Compactness (2M/R): {diagnostics['parameter_scaling']['compactness']:.3f}")
                    if diagnostics['recommendations']:
                        print("Recommendations:")
                        for rec in diagnostics['recommendations']:
                            print(f"  - {rec}")
            
            return result
            
        else:
            # Complete failure
            if verbose:
                print(f"All {attempts_made} attempts failed to find a valid solution")
                print("\nRunning failure diagnostics...")
                diagnostics = self.diagnose_convergence_failure(M_val, R_val, tolerance)
                print(f"Compactness (2M/R): {diagnostics['parameter_scaling']['compactness']:.3f}")
                if 'worst_condition' in diagnostics['boundary_condition_analysis']:
                    print(f"Most problematic boundary condition: {diagnostics['boundary_condition_analysis']['worst_condition']}")
                if diagnostics['recommendations']:
                    print("Recommendations:")
                    for rec in diagnostics['recommendations']:
                        print(f"  - {rec}")
            
            self.solution_found = False
            return {
                'A0': None,
                'B0': None,
                'C0': None,
                'residuals': [np.inf, np.inf, np.inf],
                'success': False,
                'error': f"Failed to converge after {attempts_made} attempts",
                'attempts_made': attempts_made,
                'converged': False,
                'diagnostics': self.diagnose_convergence_failure(M_val, R_val, tolerance) if verbose else None
            }
    
    def evaluate_at_radius(self, r_vals):
        """Evaluate all physical quantities at given radius values."""
        
        if not self.solution_found:
            raise ValueError("No solution found. Call solve_star_configuration first.")
        
        r_vals = np.atleast_1d(r_vals)
        
        return {
            'r': r_vals,
            'rho': self.numerical_rho(r_vals, self.A0_solved, self.B0_solved, self.C0_solved),
            'pr': self.numerical_pr(r_vals, self.A0_solved, self.B0_solved, self.C0_solved),
            'pt': self.numerical_pt(r_vals, self.A0_solved, self.B0_solved, self.C0_solved),
            'mu': self.numerical_mu(r_vals, self.A0_solved, self.B0_solved, self.C0_solved),
            'e_nu': self.numerical_e_nu(r_vals, self.A0_solved, self.B0_solved)
        }
    

    
    def get_solution_parameters(self):
        """Get the solved parameters."""
        if not self.solution_found:
            return None
        return {
            'A0': self.A0_solved,
            'B0': self.B0_solved,
            'C0': self.C0_solved
        }
    
    def solve_with_high_precision(self, M_val, R_val, tolerance=1e-12):
        """
        Convenience method for high-precision solving with comprehensive output.
        
        Parameters:
        -----------
        M_val, R_val : float
            Mass and radius values
        tolerance : float
            Target precision (default: very strict)
            
        Returns:
        --------
        dict : Comprehensive solution results with quality metrics
        """
        return self.solve_star_configuration(
            M_val=M_val, 
            R_val=R_val, 
            tolerance=tolerance,
            max_attempts=15,
            verbose=True
        )
