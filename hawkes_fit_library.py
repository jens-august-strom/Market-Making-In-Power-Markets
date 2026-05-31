import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import Bounds, minimize, differential_evolution, NonlinearConstraint
from scipy.stats import kstest, expon
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from numba import jit, njit
#
print("I was reloaded") 
#### IDA 

# IDA_values = [val_1, ..., val_k]
# IDA_config = [{'start': start_1, 'end': end_1, 'type': type_1, 'name': name_1}, ...]
# REMEMBER WE MUST CONSTRAIN THE MULTIPLICATIVE ONES TO BE ABOVE 0 STRICTLY

# params = [vu, gamma, eta, alpha_1, ..., alpha_d, beta_1, ..., beta_d, val_1, ..., val_k]

def mu_sigmoid(t, base, peak, b, c, IDA_values, IDA_config):
    out = base + peak / (1 + np.exp(-b * (t - c)))
    for val, config in zip(IDA_values, IDA_config):
        if config['start'] <= t < config['end']:
            if config['type'] == 'add':
                if out + val < 0:
                    raise ValueError(f"Invalid parameter combination: mu + IDA value is negative at time {t}. Consider adjusting the parameters or the IDA values.")
                else:
                    out += val
            elif config['type'] == 'mult':
                out *= val
    return out    

# define functions for mu and phi
def mu_exp(t, vu, gamma, eta, IDA_values, IDA_config): 
    out = vu + gamma * np.exp(eta*t)
    for val, config in zip(IDA_values, IDA_config):
        if config['start'] <= t < config['end']:
            if config['type'] == 'add':
                if out + val < 0:
                    print(out, val, t,vu, gamma, eta, vu + gamma * np.exp(eta*t))
                    raise ValueError(f"Invalid parameter combination: mu + IDA value is negative at time {t}. Consider adjusting the parameters or the IDA values.")
                else:
                    out += val
            elif config['type'] == 'mult':
                out *= val
    return out

#@jit
def mu_exp_vec(t_arr, vu, gamma, eta, IDA_values, IDA_config):
    out = vu + gamma * np.exp(eta * t_arr)          # fully vectorized
    for val, config in zip(IDA_values, IDA_config):
        mask = (t_arr >= config['start']) & (t_arr < config['end'])
        if config['type'] == 'add':
            out[mask] += val
        elif config['type'] == 'mult':
            out[mask] *= val
    if np.any(out < 0):
        raise ValueError("mu + IDA value is negative")
    return out

# Note that beta and alpha are in this case going to be 2x1 vectors and phi should return a 2x1 vector
def phi(t, beta, alpha):
    return alpha*np.exp(-beta*t)

########## HAWKES SIMULATION ##########
# TODO: Lave bedre bounds, når vi gør det på denne her måde skal vi rejecte fucking mange, især på exponential baseline
def sim_NHPP_thinning(t0, T_max,  method, params, IDA_config, cutoff = False):

    if method == 'mu_sigmoid':
        base, peak, b, c = params[0], params[1], params[2], params[3]

        d = (len(params) - 4 - len(IDA_config)) // 2
        IDA_values = params[4+2*d:]

        t = t0
        event_times = []

        # Calculate max of the baseline intensity before and after a cutoff to save computing power.
        time_points_before = np.arange(t0, T_max - cutoff + 0.1, 0.1)
        time_points_after = np.arange(T_max - cutoff, T_max + 0.1, 0.1)

        max_before = max([mu_sigmoid(time, base, peak, b, c, IDA_values, IDA_config) for time in time_points_before])
        max_after = max([mu_sigmoid(time, base, peak, b, c, IDA_values, IDA_config) for time in time_points_after])



        while t < T_max:
            # Take the max from either before or after the cutoff based on where they are.
            if t <= T_max - cutoff:
                mu_max = max_before
            else:
                mu_max = max_after


            # Generate next arrival as if it was homogenous poisson with max intensity. (time steps are exp distributed)
            t_step = np.random.exponential(1/mu_max)
            t += t_step
            
            if t > T_max:
                break  

                
            # Accept or reject the event based on the intensity at time t
            acceptance_probability = mu_sigmoid(t, base, peak, b, c, IDA_values, IDA_config) / mu_max

            if  np.random.uniform(0, 1) <= acceptance_probability:
                event_times.append(t)

    if method == 'mu_exp':
        vu, gamma, eta = params[0], params[1], params[2]
        d = (len(params) - 3 - len(IDA_config)) // 2
        IDA_values = params[3+2*d:]

        if isinstance(cutoff, float) and cutoff > 0:
            # Calculate max of the baseline intensity before and after a cutoff to save computing power.
            time_points_before = np.arange(0, T_max - cutoff + 0.1, 0.01)
            time_points_after = np.arange(T_max - cutoff, T_max + 0.1, 0.01)

            max_before = max([mu_exp(time, vu, gamma, eta, IDA_values, IDA_config) for time in time_points_before])
            max_after = max([mu_exp(time, vu, gamma, eta, IDA_values, IDA_config) for time in time_points_after])
            

            t = t0
            event_times = []
            
            while t < T_max:
                # Generate next arrival as if it was homogenous poisson with max intensity. (time steps are exp distributed)

                if t <= T_max - cutoff:
                    mu_max = max_before
                else:
                    mu_max = max_after
                
                t_step = np.random.exponential(1/mu_max)
                t += t_step
                
                if t > T_max:
                    break
                    
                # Accept or reject the event based on the intensity at time t
                acceptance_probability = mu_exp(t, vu, gamma, eta, IDA_values, IDA_config) / mu_max

                if  np.random.uniform(0, 1) <= acceptance_probability:
                    event_times.append(t)
        else:
            if t0==T_max:
                time_points = np.array([t0])
            else:
                time_points = np.arange(t0, T_max, min(0.01, 0.5*(T_max - t0)))

            mu_max = max([mu_exp(time, vu, gamma, eta, IDA_values, IDA_config) for time in time_points])

            event_times = []
            
            t = t0

            while t < T_max:
                # Generate next arrival as if it was homogenous poisson with max intensity. (time steps are exp distributed)
                
                t_step = np.random.exponential(1/mu_max)
                t += t_step
                
                if t > T_max:
                    break
                    
                # Accept or reject the event based on the intensity at time t
                acceptance_probability = mu_exp(t, vu, gamma, eta, IDA_values, IDA_config) / mu_max

                if  np.random.uniform(0, 1) <= acceptance_probability:
                    event_times.append(t)

    if method == 'phi':
        beta, alpha = params[0], params[1]

        # Since phi is a decreasing function 
        phi_max = phi(t0, beta, alpha)
        if phi_max == 0:
            return np.array([])

        event_times = []
        
        t = t0

        while t < T_max:
            # Generate next arrival as if it was homogenous poisson with max intensity. (time steps are exp distributed)

            t_step = np.random.exponential(1/phi_max)
            t += t_step
            
            if t > T_max:
                break
                
            # Accept or reject the event based on the intensity at time t
            acceptance_probability = phi(t, beta, alpha) / phi_max

            if  np.random.uniform(0, 1) <= acceptance_probability:
                event_times.append(t)   
    
    return np.array(event_times)


def sim_hawkes(t, T_max, param_arrays, alpha, beta, IDA_values, IDA_config, baseline = None, history = None, print_progress = False, save_exc_taus = False):
    n = len(param_arrays)
    gen = 0
    next_gen = 0
    if not isinstance(history, list):
        history = [[] for _ in range(n)]
    
    excitation_taus_save = [list(history[i]) for i in range(n)]
    event_times_save = [[] for _ in range(n)]
    # Here we loop over the partial processes
    for i in range(n):
        if print_progress:
            print(f"Simulating process {i} with baseline {baseline[i]}")
    
        # Start by simulating the events from the baseline intensity in the interval.

        baseline_i = baseline[i] if isinstance(baseline, list) else baseline
        if baseline_i == 'mu_exp':
            vu_i, gamma_i, eta_i = param_arrays[i][0], param_arrays[i][1], param_arrays[i][2]
            # simulate non_homogenous poisson process to get "generation 0"
            event_times = sim_NHPP_thinning(t, T_max, baseline_i, list([vu_i, gamma_i, eta_i]) + list(IDA_values[i]), IDA_config[i]).tolist()
            
        elif baseline_i == 'mu_sigmoid':
            base_i, peak_i, b_i, c_i = param_arrays[i][0], param_arrays[i][1], param_arrays[i][2], param_arrays[i][3]
            # simulate non_homogenous poisson process to get "generation 0"
            event_times = sim_NHPP_thinning(t, T_max, baseline_i, list([base_i, peak_i, b_i, c_i]) + list(IDA_values[i]), IDA_config[i]).tolist()
        if any(len(h) > 0 for h in history):
            # Now we extend these event times with excitation from history. 
            for j in range(n):
                # The previous events from the partial process j
                tau_hist = np.asarray(history[j], dtype = float)
                L_hist = T_max - tau_hist
                tau_hist_valid = tau_hist[L_hist > 0]
                L_hist_valid = L_hist[L_hist > 0]

                if len(tau_hist_valid) > 0:
                    a_ij = alpha[i,j]
                    b_ij = beta[i,j]
                    if a_ij > 0 and b_ij > 0:
                        exp_term = np.exp(-b_ij * L_hist_valid)
                        m = (a_ij / b_ij) * (1.0 - exp_term)

                        K = np.random.poisson(m)
                        total_offspring = np.sum(K)

                        if total_offspring>0:
                            parent_rep = np.repeat(tau_hist_valid, K)
                            exp_term_rep = np.repeat(exp_term, K)

                            U = np.random.uniform(0, 1, total_offspring)
                            S = - (1.0/b_ij) * np.log(1.0 - U * (1.0 - exp_term_rep))
                            offspring_from_event = parent_rep + S

                            # Add the offspring that fall within the interval to generation 0
                            valid_offspring = offspring_from_event[(offspring_from_event >= t) & (offspring_from_event <= T_max)].tolist()
                            event_times.extend(valid_offspring)     
        
        # If there are any events, then there is a next generation that we need to find offspring for
        if len(event_times) > 0:
            next_gen = 1
            event_times_save[i] = event_times

    # Initialize the event times from the zero'th generation
    event_times_gen = [event_times_save[i] for i in range(n)]

    while next_gen > 0:
        if print_progress:
            print(f"Simulating generation {gen} with {len(event_times_gen[0])} LO events and {len(event_times_gen[1])} MO events")
        # Clear event times from the next generation
        event_times_next_gen = [[] for _ in range(n)]
        gen += 1
        next_gen = 0
        # Loop over each partial process, to find the offspring from the previous generation
        for i in range(n):
            # Loop over the effects of each partial process to ensure that we count cross excitation (note that self-excitation is also contained in this)
            for j in range(n):
                # The previous events from the partial process j
                tau = np.asarray(event_times_gen[j], dtype = float)
                excitation_taus_save[j].extend(event_times_gen[j])
                L = T_max - tau
                tau_valid = tau[L > 0]
                L_valid = L[L > 0]

                if len(tau_valid) > 0:
                    a_ij = alpha[i,j]
                    b_ij = beta[i,j]
                    if a_ij > 0 and b_ij > 0:
                        exp_term = np.exp(-b_ij * L_valid)
                        m = (a_ij / b_ij) * (1.0 - exp_term)

                        K = np.random.poisson(m)
                        total_offspring = np.sum(K)

                        if total_offspring>0:
                            parent_rep = np.repeat(tau_valid, K)
                            exp_term_rep = np.repeat(exp_term, K)

                            U = np.random.uniform(0, 1, total_offspring)
                            S = - (1.0/b_ij) * np.log(1.0 - U * (1.0 - exp_term_rep))
                            offspring_from_event = parent_rep + S
                            event_times_next_gen[i].extend(offspring_from_event.tolist())
                    
            ### Prepare for the next iteration if there were events in the following generation
            if len(event_times_next_gen[i]) > 0:
                next_gen = 1            
            ### Saving the data            
            # save the event times for the final summary
            event_times_save[i].extend(event_times_next_gen[i])
        
        # save the event times of this generation to be used as the current generation
        for i in range(n):
            event_times_gen[i] = event_times_next_gen[i]

    # Save and sort the event times of all generations
    event_times_save = [sorted(event_times_save[i]) for i in range(n)]
    if save_exc_taus:
        return event_times_save, excitation_taus_save
    
    return event_times_save

##### LOG LIKELIHOOD CALCULATION #####
# Helper functions for calculating the integral of phi for use in lambda calculation
@njit(cache=True)
def phi_integral_cross(data, beta, alpha, exc):
    # data and exc must be sorted 1D float arrays
    n = data.size
    m = exc.size
    out = np.zeros(n, dtype=np.float64)

    i = 0
    j = 0
    R = 0.0
    t_prev = 0.0

    while i < n or j < m:
        take_data = (j >= m) or (i < n and data[i] <= exc[j])
        t = data[i] if take_data else exc[j]
        dt = t - t_prev
        if dt > 0.0:
            R *= np.exp(-beta * dt)
            t_prev = t

        if take_data:
            out[i] = R
            i += 1
        else:
            R += 1.0
            j += 1

    return alpha * out



@njit(cache=True)
def phi_integral_self(data, beta, alpha):
    n_data = len(data)
    
    dt = np.zeros(n_data,dtype=np.float64)
    dt[0] = 0.0
    dt[1:] = data[1:] - data[:-1]
    integral = np.zeros(n_data,dtype=np.float64)
    decays = np.exp(-beta * dt)
    integral[0] = 0.0
    for i in range(1, n_data):
        integral[i] = (integral[i-1]+1)*decays[i]
    
    return alpha * integral

@njit
def mu_integral_interval_exp(vu, gamma, eta, start, end):
    return(vu * (end - start) + (gamma/eta) * (np.exp(eta * end) - np.exp(eta * start)))

@njit
def mu_integral_interval_sigmoid(base, peak, b, c, start, end):
    return(base * end + (peak / b) * np.log(1 + np.exp(b * (end - c))) - (base * start + (peak / b) * np.log(1 + np.exp(b * (start - c)))))

# Actual negative log likelihood function
#@jit
def neg_log_likelihood(params, T, data, IDA_config, excitation, excitation_data_, baseline = "mu_exp", vectorized = False, self_exc_idx=None):
    if vectorized == False:    
        if baseline == "mu_exp":

            # Convert excitation data to numpy array
            excitation_data = [np.array(d) for d in excitation_data_]

            # Save the parameters for use in this function
            vu, gamma, eta = params[0], params[1], params[2]
            # Save the alpha and beta parameters
            d = sum(excitation)
            alphas = params[3:3+d]
            betas = params[3+d:3+2*d]

            # Save the IDA parameters
            IDA_values = params[3+2*d:]

            phi_together = np.zeros(len(data))

            # This is the right excitation data, because we apply the excitation data filter.
            for i in range(len(alphas)):
                if i == self_exc_idx:
                    phi_together += phi_integral_self(data, betas[i], alphas[i])
                else:
                    phi_together += phi_integral_cross(data, betas[i], alphas[i], excitation_data[i])

            # Calculate the first part of the negative log likelihood:
            first_part = - np.sum([np.log(mu_exp(data[i],vu, gamma, eta, IDA_values, IDA_config) + phi_together[i]) for i in range(len(data))])
            # Calculate the second part of the negative log likelihood:

            # Calculate integral of mu
            if eta == 0:
                mu_integral = vu * T + gamma * T
            else:
                mu_integral = mu_integral_interval_exp(vu, gamma, eta, 0, T)
            # Add IDA integrals
            for val, config in zip(IDA_values, IDA_config):
                if config['type'] == 'add':
                    mu_integral += val * (config['end'] - config['start']) if T >= config['end'] else (val * (T - config['start']) if T > config['start'] else 0)
                elif config['type'] == 'mult':
                    # We remove the original value and add in the scaled value
                    integral_in_interval = mu_integral_interval_exp(vu, gamma, eta, config['start'], config['end'])
                    mu_integral += integral_in_interval * val - integral_in_interval
            # Calculate integral of phi
            phi_integral_val = np.sum([np.sum((alphas[i] / betas[i]) * (1 - np.exp(-betas[i] * (T - excitation_data[i])))) for i in range(len(alphas))])
            second_part = mu_integral + phi_integral_val - T

            if first_part + second_part < np.inf:
                return first_part + second_part
            else:
                return 1e12
        
        if baseline == "mu_sigmoid":

            # Convert excitation data to numpy array
            excitation_data = [np.array(d) for d in excitation_data_]

            # Save the parameters for use in this function
            base, peak, b, c = params[0], params[1], params[2], params[3]
            # Save the alpha and beta parameters
            d = sum(excitation)
            alphas = params[4:4+d]
            betas = params[4+d:4+2*d]

            # Save the IDA parameters
            IDA_values = params[4+2*d:]

            phi_together = np.zeros(len(data))

            # This is the right excitation data, because we apply the excitation data filter.
            for i in range(len(alphas)):
                if i == self_exc_idx:
                    phi_together += phi_integral_self(data, betas[i], alphas[i])
                else:
                    phi_together += phi_integral_cross(data, betas[i], alphas[i], excitation_data[i])


            # Calculate the first part of the negative log likelihood:
            first_part = - np.sum([np.log(mu_sigmoid(data[i], base, peak, b, c, IDA_values, IDA_config) + phi_together[i]) for i in range(len(data))])
            # Calculate the second part of the negative log likelihood:

            # Calculate integral of mu

            mu_integral = mu_integral_interval_sigmoid(base, peak, b, c, 0, T)

            # Add IDA integrals
            for val, config in zip(IDA_values, IDA_config):
                if config['type'] == 'add':
                    mu_integral += val * (config['end'] - config['start']) if T >= config['end'] else (val * (T - config['start']) if T > config['start'] else 0)
                elif config['type'] == 'mult':
                    # We remove the original value and add in the scaled value
                    integral_in_interval = mu_integral_interval_sigmoid(base, peak, b, c, config['start'], config['end'])
                    mu_integral += integral_in_interval * val - integral_in_interval
            # Calculate integral of phi
            phi_integral_val = np.sum([np.sum((alphas[i] / betas[i]) * (1 - np.exp(-betas[i] * (T - excitation_data[i])))) for i in range(len(alphas))])
            second_part = mu_integral + phi_integral_val - T

            if first_part + second_part < np.inf:
                return first_part + second_part
            else:
                return 1e12
    if vectorized == True:

        excitation_data = [np.asarray(d) for d in excitation_data_]
        d = sum(excitation)
        n = data.size
        
        def _compute_phi_together(alphas, betas):
            phi_together = np.zeros(n, dtype=float)
            for k, (alpha, beta, exc_data) in enumerate(zip(alphas, betas, excitation_data)):
                if k == self_exc_idx:
                    phi_together += phi_integral_self(data, beta, alpha)
                else:
                    phi_together += phi_integral_cross(data, beta, alpha, exc_data)
            return phi_together

        def _compute_phi_integral_val(alphas, betas):
            total = 0.0
            for alpha, beta, exc_data in zip(alphas, betas, excitation_data):
                total += np.sum((alpha / beta) * (1.0 - np.exp(-beta * (T - exc_data))))
            return total

        def _add_ida_integrals(mu_integral, IDA_values, interval_integral_func):
            for val, config in zip(IDA_values, IDA_config):
                start = config["start"]
                end = config["end"]

                if config["type"] == "add":
                    if T >= end:
                        mu_integral += val * (end - start)
                    elif T > start:
                        mu_integral += val * (T - start)

                elif config["type"] == "mult":
                    integral_in_interval = interval_integral_func(start, end)
                    mu_integral += integral_in_interval * val - integral_in_interval

            return mu_integral

        if baseline == "mu_exp":
            vu, gamma, eta = params[0], params[1], params[2]
            alphas = params[3:3 + d]
            betas = params[3 + d:3 + 2 * d]
            IDA_values = params[3 + 2 * d:]

            phi_together = _compute_phi_together(alphas, betas)

            mu_vals = mu_exp_vec(data, vu, gamma, eta, IDA_values, IDA_config)
            first_part = -np.sum(np.log(mu_vals + phi_together))

            if eta == 0:
                mu_integral = vu * T + gamma * T
            else:
                mu_integral = mu_integral_interval_exp(vu, gamma, eta, 0, T)

            mu_integral = _add_ida_integrals(
                mu_integral,
                IDA_values,
                lambda start, end: mu_integral_interval_exp(vu, gamma, eta, start, end)
            )

            phi_integral_val = _compute_phi_integral_val(alphas, betas)
            second_part = mu_integral + phi_integral_val - T

            total = first_part + second_part
            return total if total < np.inf else 1e12

        if baseline == "mu_sigmoid":
            base, peak, b, c = params[0], params[1], params[2], params[3]
            alphas = params[4:4 + d]
            betas = params[4 + d:4 + 2 * d]
            IDA_values = params[4 + 2 * d:]

            phi_together = _compute_phi_together(alphas, betas)

            mu_vals = np.fromiter(
                (mu_sigmoid(t, base, peak, b, c, IDA_values, IDA_config) for t in data),
                dtype=float,
                count=n
            )
            first_part = -np.sum(np.log(mu_vals + phi_together))

            mu_integral = mu_integral_interval_sigmoid(base, peak, b, c, 0, T)

            mu_integral = _add_ida_integrals(
                mu_integral,
                IDA_values,
                lambda start, end: mu_integral_interval_sigmoid(base, peak, b, c, start, end)
            )

            phi_integral_val = _compute_phi_integral_val(alphas, betas)
            second_part = mu_integral + phi_integral_val - T

            total = first_part + second_part
            return total if total < np.inf else 1e12

        
# IMPORTANT THAT THE IDA_config_i is the dict that corresponds to the process we are fitting.
#@jit
def total_nll(params0, T, data_list, IDA_config_i, excitation, excitation_data_list, baseline = "mu_exp", scale = None, vectorized = False, self_exc_idx=None):
    function_params = np.array(params0, dtype=float)
    d = sum(excitation)

    if scale:
        if baseline == "mu_exp":
            function_params[0:3] *= np.array(scale["mu_exp"])
            for j in range(d):
                function_params[3 + j] *= scale["alpha"]
                function_params[3 + d + j] *= scale["beta"]
            for j, config in enumerate(IDA_config_i):
                if config['type'] == 'add':
                    function_params[3 + 2*d + j] *= scale.get("IDA_add", 1)

        elif baseline == "mu_sigmoid":
            function_params[0:4] *= np.array(scale["mu_sigmoid"])
            for j in range(d):
                function_params[4 + j] *= scale["alpha"]
                function_params[4 + d + j] *= scale["beta"]
            for j, config in enumerate(IDA_config_i):
                if config['type'] == 'add':
                    function_params[4 + 2*d + j] *= scale.get("IDA_add", 1)

    data_list = [np.array(data) for data in data_list]
    return np.sum(neg_log_likelihood(function_params, T, data, IDA_config_i, excitation = excitation, excitation_data_ = excitation_data_list[i], baseline = baseline, vectorized= vectorized, self_exc_idx=self_exc_idx) for i,data in enumerate(data_list))


########## HAWKES FITTING ##########

# Helper functions -------------------------------------------------------------------------------






def progress_bar(xk, convergence):
    # xk is the best parameter vector found so far
    # convergence is the fractional value of convergence
    print(f"Generation completed. Current Convergence: {convergence:.10f}. Current best parameters: {xk}")

def get_params0(baseline_params, alpha_, beta_, IDA_values_, IDA_config, index, excitation = [False], baseline_ = ["mu_exp"], scale = {}, direction = None):
    baseline = baseline_[index]
    # Copy the parameters to avoid modifying the original arrays
    baseline_params_ = baseline_params[index].copy()
    alpha = alpha_.copy()
    beta = beta_.copy()
    IDA_values = IDA_values_[index].copy()


    if direction == None:
        # This is for DE, then we won't scale anything
        if baseline == "mu_exp":
            vu, gamma, eta = baseline_params[index][0], baseline_params[index][1], baseline_params[index][2]
            return np.array([vu, gamma, eta] + list(alpha_[index]) + list(beta_[index]) + list(IDA_values_[index]))
        
        if baseline == "mu_sigmoid":
            print(baseline_params)
            base, peak, b, c = baseline_params[index][0], baseline_params[index][1], baseline_params[index][2], baseline_params[index][3]
            print([base, peak, b, c])
            return np.array([base, peak, b, c] + list(alpha_[index]) + list(beta_[index]) + list(IDA_values_[index]))

    if direction == "multiply":    
        # Scale IDA values   
        for i, config in enumerate(IDA_config[index]):
            if config['type'] == 'add':
                IDA_values[i] = float(IDA_values[i]) * scale["IDA_add"]
                
            elif config['type'] == 'mult':
                IDA_values[i] = IDA_values[i]

        # Scale alpha and beta parameters
        num_excitation = sum(excitation)
        
        alpha_scaled = np.zeros(num_excitation)
        beta_scaled = np.zeros(num_excitation)

        idx = 0
        for i in range(len(excitation)):
            if excitation[i]:
                alpha_scaled[idx] = alpha[index, i] * scale["alpha"]
                beta_scaled[idx] = beta[index, i] * scale["beta"]
                idx += 1
        
        # Scale baseline parameters and return the list of parameters
        if baseline == "mu_exp":
            params_scaled = np.array(baseline_params_[0:3])*np.array(scale["mu_exp"])
            vu, gamma, eta = params_scaled[0], params_scaled[1], params_scaled[2]

            return np.array([vu, gamma, eta] + list(alpha_scaled) + list(beta_scaled) + list(IDA_values))
        
        if baseline == "mu_sigmoid":
            params_scaled = np.array(baseline_params_[0:4])*np.array(scale["mu_sigmoid"])
            base, peak, b, c = params_scaled[0], params_scaled[1], params_scaled[2], params_scaled[3]

            return np.array([base, peak, b, c] + list(alpha_scaled) + list(beta_scaled) + list(IDA_values))

    if direction == "divide":    
        # Scale IDA values   
        for i, config in enumerate(IDA_config[index]):
            if config['type'] == 'add':
                IDA_values[i] = float(IDA_values[i]) / scale["IDA_add"]
                
            elif config['type'] == 'mult':
                IDA_values[i] = IDA_values[i]

        # Scale alpha and beta parameters
        num_excitation = sum(excitation)
        
        alpha_scaled = np.zeros(num_excitation)
        beta_scaled = np.zeros(num_excitation)

        idx = 0
        for i in range(len(excitation)):
            if excitation[i]:
                alpha_scaled[idx] = alpha[index, i] / scale["alpha"]
                beta_scaled[idx] = beta[index, i] / scale["beta"]
                idx += 1
        
        # Scale baseline parameters and return the list of parameters
        if baseline == "mu_exp":
            params_scaled = np.array(baseline_params_[0:3])/np.array(scale["mu_exp"])
            vu, gamma, eta = params_scaled[0], params_scaled[1], params_scaled[2]

            return np.array([vu, gamma, eta] + list(alpha_scaled) + list(beta_scaled) + list(IDA_values))
        
        if baseline == "mu_sigmoid":
            params_scaled = np.array(baseline_params_[0:4])/np.array(scale["mu_sigmoid"])
            base, peak, b, c = params_scaled[0], params_scaled[1], params_scaled[2], params_scaled[3]

            return np.array([base, peak, b, c] + list(alpha_scaled) + list(beta_scaled) + list(IDA_values))


def sol_to_params(sol, IDA_config, excitation = [False], baseline = "mu_exp", scale = {}, direction = "multiply"):
    if direction == None:
        # Here we make no adjustments
        if baseline == "mu_exp":
            vu = sol.x[0]
            gamma = sol.x[1]
            eta = sol.x[2]
            d = sum(excitation)
            alpha_ = np.zeros(d)
            beta_ = np.zeros(d)
            idx = 0
            for i in range(len(excitation)):
                if excitation[i]:
                    alpha_[idx] = sol.x[3 + idx]
                    beta_[idx] = sol.x[3 + d + idx]
                    idx += 1
            IDA_values = sol.x[3 + 2*d:]
            return np.array([vu, gamma, eta] + list(alpha_) + list(beta_) + list(IDA_values)) 
        
        if baseline == "mu_sigmoid":
            base = sol.x[0] 
            peak = sol.x[1]
            b = sol.x[2]
            c = sol.x[3]

            d = sum(excitation)
            alpha_ = np.zeros(d)
            beta_ = np.zeros(d)
            idx = 0
            for i in range(len(excitation)):
                if excitation[i]:
                    alpha_[idx] = sol.x[4 + idx]
                    beta_[idx] = sol.x[4 + d + idx] 
                    idx += 1

            IDA_values = sol.x[4 + 2*d:]

            return np.array([base, peak, b, c] + list(alpha_) + list(beta_) + list(IDA_values))               

    if direction == "multiply":
            d = sum(excitation)
            alpha_ = np.zeros(d)
            beta_ = np.zeros(d)
            idx = 0
            start_idx = (3 if baseline == "mu_exp" else 4)
            for i in range(len(excitation)):
                if excitation[i]:
                    alpha_[idx] = sol.x[start_idx + idx] * scale["alpha"]
                    beta_[idx] = sol.x[start_idx + d + idx] * scale["beta"]
                    idx += 1
            # Scale additive IDA values
            for i, config in enumerate(IDA_config):
                if config['type'] == 'add':
                    sol.x[start_idx + 2*d + i] *= scale["IDA_add"]
            
            IDA_values = sol.x[start_idx + 2*d:]

            if baseline == "mu_exp":
                params_scaled = sol.x[0:3]*np.array(scale["mu_exp"])
                vu, gamma, eta = params_scaled[0], params_scaled[1], params_scaled[2]
                return np.array([vu, gamma, eta] + list(alpha_) + list(beta_) + list(IDA_values))
        
            if baseline == "mu_sigmoid":
                params_scaled = np.array(sol.x[0:4])*np.array(scale["mu_sigmoid"])
                base, peak, b, c = params_scaled[0], params_scaled[1], params_scaled[2], params_scaled[3]
                return np.array([base, peak, b, c] + list(alpha_) + list(beta_) + list(IDA_values))  
            
    if direction == "divide":
            d = sum(excitation)
            alpha_ = np.zeros(d)
            beta_ = np.zeros(d)
            idx = 0
            start_idx = (3 if baseline == "mu_exp" else 4)
            for i in range(len(excitation)):
                if excitation[i]:
                    alpha_[idx] = sol.x[start_idx + idx] / scale["alpha"]
                    beta_[idx] = sol.x[start_idx + d + idx] / scale["beta"]
                    idx += 1
            # Scale additive IDA values
            for i, config in enumerate(IDA_config):
                if config['type'] == 'add':
                    sol.x[start_idx + 2*d + i] /= scale["IDA_add"]
            
            IDA_values = sol.x[start_idx + 2*d:]

            if baseline == "mu_exp":
                params_scaled = sol.x[0:3]/np.array(scale["mu_exp"])
                vu, gamma, eta = params_scaled[0], params_scaled[1], params_scaled[2]
                return np.array([vu, gamma, eta] + list(alpha_) + list(beta_) + list(IDA_values))
        
            if baseline == "mu_sigmoid":
                params_scaled = np.array(sol.x[0:4])/np.array(scale["mu_sigmoid"])
                base, peak, b, c = params_scaled[0], params_scaled[1], params_scaled[2], params_scaled[3]
                return np.array([base, peak, b, c] + list(alpha_) + list(beta_) + list(IDA_values))  


def excitation_data_filter(excitation_data, excitation):
        excitation_data_i = []
        # For each simulation
        for k in range(len(excitation_data)):
            # We make an empty list for excitation data for this simulation (would be all 3 if all were true)
            excitation_k = []
            for j in range(len(excitation)):
                # if the excitation is true for the process we add it to the list
                if excitation[j] == True:
                    excitation_k.append(excitation_data[k][j])
            # Then we add the excitation data for this simulation to the list of excitation data for this process
            excitation_data_i.append(excitation_k)
        return excitation_data_i

# Main functions --------------------------------------------------------------------------------------

# A function that fits a single hawkes process
def hawkes_fitter(params0, T, data, IDA_config_i,excitation = [False], excitation_data = None, type = "", include_results = True, algo = 'SLSQP', workers = -1, popsize = 10, strategy = 'rand1bin', baseline = "mu_exp", polisher = 'L-BFGS-B', param_bounds = None, scale = {}, vectorized = False, self_exc_idx=None):
    d = sum(excitation)
    k = len(IDA_config_i)

    
    if baseline == "mu_exp":
        lb = [param_bounds["vu_l"]/scale["mu_exp"][0], param_bounds["gamma_l"]/scale["mu_exp"][1], param_bounds["eta_l"]/scale["mu_exp"][2]] +  [param_bounds["alpha_l"][_]/scale["alpha"] for _ in range(d)] + [param_bounds["beta_l"][_]/scale["beta"] for _ in range(d)]

        ub = [param_bounds["vu_u"]/scale["mu_exp"][0], param_bounds["gamma_u"]/scale["mu_exp"][1], param_bounds["eta_u"]/scale["mu_exp"][2]] + [param_bounds["alpha_u"][_]/scale["alpha"] for _ in range(d)] + [param_bounds["beta_u"][_]/scale["beta"] for _ in range(d)]

        # No scaling for the initial differential evolution
        lb_DE = [param_bounds["vu_l"], param_bounds["gamma_l"], param_bounds["eta_l"]] + [param_bounds["alpha_l"][_] for _ in range(d)] + [param_bounds["beta_l"][_] for _ in range(d)]
        ub_DE = [param_bounds["vu_u"], param_bounds["gamma_u"], param_bounds["eta_u"]] + [param_bounds["alpha_u"][_] for _ in range(d)] + [param_bounds["beta_u"][_] for _ in range(d)]
    
    if baseline == "mu_sigmoid":
        lb = [param_bounds["base_l"]/scale["mu_sigmoid"][0], param_bounds["peak_l"]/scale["mu_sigmoid"][1], param_bounds["b_l"]/scale["mu_sigmoid"][2], param_bounds["c_l"]/scale["mu_sigmoid"][3]] +  [param_bounds["alpha_l"][_]/scale["alpha"] for _ in range(d)] + [param_bounds["beta_l"][_]/scale["beta"] for _ in range(d)]

        ub = [param_bounds["base_u"]/scale["mu_sigmoid"][0], param_bounds["peak_u"]/scale["mu_sigmoid"][1], param_bounds["b_u"]/scale["mu_sigmoid"][2], param_bounds["c_u"]/scale["mu_sigmoid"][3]] + [param_bounds["alpha_u"][_]/scale["alpha"] for _ in range(d)] + [param_bounds["beta_u"][_]/scale["beta"] for _ in range(d)]
        # No scaling for the initial differential evolution
        lb_DE = [param_bounds["base_l"], param_bounds["peak_l"], param_bounds["b_l"], param_bounds["c_l"]] +  [param_bounds["alpha_l"][_] for _ in range(d)] + [param_bounds["beta_l"][_] for _ in range(d)]
        ub_DE = [param_bounds["base_u"], param_bounds["peak_u"], param_bounds["b_u"], param_bounds["c_u"]] + [param_bounds["alpha_u"][_] for _ in range(d)] + [param_bounds["beta_u"][_] for _ in range(d)]

    for config in IDA_config_i:
        lb.append(config['limits'][0]/(scale["IDA_add"] if config["type"] == "add" else 1))
        ub.append(config['limits'][1]/(scale["IDA_add"] if config["type"] == "add" else 1))
        lb_DE.append(config['limits'][0])
        ub_DE.append(config['limits'][1])
    
    if type == 'no_constant':
        lb[0] = 0.0
        ub[0] = 0.0

    bounds = Bounds(lb, ub)
    bounds_DE = Bounds(lb_DE, ub_DE)

    if algo == 'SLSQP':
        print("Starting SLSQP optimization...")

        sol = minimize(total_nll, params0, args = (T, data, IDA_config_i, excitation, excitation_data, baseline, scale, vectorized, self_exc_idx), method = 'SLSQP', bounds = bounds, options={'eps': 1e-4, 'ftol': 1e-12})

    if algo == 'L-BFGS-B':
        convergence_history = []
        print("Starting L-BFGS-B optimization...")
        def nm_callback(xk):
            current_val = total_nll(
                xk, T, data, IDA_config_i, excitation, excitation_data, baseline, scale, vectorized, self_exc_idx
            )
            convergence_history.extend([current_val])
            print(convergence_history)

        sol = minimize(total_nll, params0, args = (T, data, IDA_config_i, excitation, excitation_data, baseline, scale, vectorized, self_exc_idx), method = 'L-BFGS-B', callback=nm_callback, bounds = bounds, options={'eps': 1e-4, 'ftol': 1e-12, 'maxiter': 500})
        return sol

    if algo == 'Nelder-Mead':
        convergence_history = []
        print("Starting Nelder-Mead optimization...")

        iteration_state = {'iter': 0, 'best_val': np.inf}

        def nm_callback(xk):
            iteration_state['iter'] += 1
            current_val = total_nll(
                xk, T, data, IDA_config_i, excitation, excitation_data, baseline, scale, vectorized, self_exc_idx
            )
            convergence_history.extend([current_val])

        sol = minimize(
            total_nll,
            params0,
            args=(T, data, IDA_config_i, excitation, excitation_data, baseline, scale, vectorized, self_exc_idx),
            method='Nelder-Mead',
            bounds=bounds,
            callback=nm_callback,
            options={
                'maxiter': 10000,
                'xatol': 1e-6,
                'fatol': 1e-10,
                'adaptive': True,
                'return_all': True,
                'disp': True,
            },
        )
        return sol, convergence_history

    if algo == 'differential_evolution':
        sol_init = differential_evolution(total_nll, bounds = bounds_DE, args = (T, data, IDA_config_i, excitation, excitation_data, baseline, None, vectorized, self_exc_idx), strategy = strategy, popsize = popsize, tol = 0.001, workers = workers, callback = progress_bar, polish = False)
        if polisher == "False":
            sol = sol_init
        else:
            params0_new = sol_to_params(sol_init, IDA_config_i, excitation, baseline, scale = scale, direction = "divide")
            print(f"Differential evolution completed, now polishing with {polisher}...")
            sol = minimize(total_nll, params0_new, args = (T, data, IDA_config_i, excitation, excitation_data, baseline, scale, vectorized, self_exc_idx), method = polisher, bounds = bounds)
            print(sol_to_params(sol, IDA_config_i, excitation, baseline, scale = scale, direction = "multiply"))
            # Return both DE and polished results
            return {'de': sol_init, 'polished': sol}
    
    if not include_results:
        return sol

    if include_results and baseline == "mu_exp":
        print(sol.success)
        print(sol.message)
        print(f"vu: {sol.x[0]:.4f}")
        print(f"gamma: {sol.x[1]:.4f}")
        print(f"eta: {sol.x[2]:.4f}")
        for i in range(d):
            print(f"alpha_{i+1}: {sol.x[3 + i]*1000:.4f}")
        for i in range(d):
            print(f"beta_{i+1}: {sol.x[3 + d + i]*1000:.4f}")
        for j in range(k):
            if IDA_config_i[j]['type'] == 'add':
                print(f"IDA_value_{j+1} (additive): {sol.x[3 + 2*d + j]*100:.4f}")
            elif IDA_config_i[j]['type'] == 'mult':
                print(f"IDA_value_{j+1} (multiplicative): {sol.x[3 + 2*d + j]:.4f}")      
        return sol
    
    if include_results and baseline == "mu_sigmoid":
        print(sol.success)
        print(sol.message)
        print(f"base: {sol.x[0]:.4f}")
        print(f"peak: {sol.x[1]:.4f}")
        print(f"b: {sol.x[2]:.4f}")
        print(f"c: {sol.x[3]:.4f}")
        for i in range(d):
            print(f"alpha_{i+1}: {sol.x[4 + i]*1000:.4f}")
        for i in range(d):
            print(f"beta_{i+1}: {sol.x[4 + d + i]*1000:.4f}")
        for j in range(k):
            if IDA_config_i[j]['type'] == 'add':
                print(f"IDA_value_{j+1} (additive): {sol.x[4 + 2*d + j]*100:.4f}")
            elif IDA_config_i[j]['type'] == 'mult':
                print(f"IDA_value_{j+1} (multiplicative): {sol.x[4 + 2*d + j]:.4f}")  

# A function that loops over the data and fits multiple modes iteratively
def fit_all(baseline_params, alpha0, beta0, IDA_values, IDA_config, process_data, excitation_data, T, excitation_list, type_list, algo = 'SLSQP', 
            workers = -1, popsize = 10, strategy = 'rand1bin', baseline = ["mu_exp"], polisher = 'L-BFGS-B', param_bounds = None, scale = {}, vectorized = False, indices_to_fit = None):
    if len(baseline) == 1:
        baseline = baseline * len(process_data)
    
    if param_bounds is None:
        print("Using default parameter bounds. Consider providing custom bounds for better performance.")
        param_bounds = [
            {
                "vu_l": 0.0,               "vu_u": 50.0,
                "gamma_l": 0.0,            "gamma_u": 50.0,
                "eta_l": 1e-2,             "eta_u": 1.0,
                "base_l": 0.0,             "base_u": 50.0,
                "peak_l": 0.0,             "peak_u": 1000.0,
                "b_l": 1e-2,               "b_u": 10.0,
                "c_l": 1e-2,               "c_u": 20.0,
                
                "alpha_l": [1e-4 for _ in range(len(excitation_list[0]))],
                "alpha_u": [10.0 for _ in range(len(excitation_list[0]))],
                "beta_l":  [1e-4 for _ in range(len(excitation_list[0]))],
                "beta_u":  [10.0 for _ in range(len(excitation_list[0]))],
            }
            for _ in range(len(process_data))
        ]

    results = []
    convergence_history = []
    if algo == "L-BFGS-B":
        for i in range(len(process_data)):
            if indices_to_fit is not None and i not in indices_to_fit:
                continue
            print(f"Fitting model for process {i+1}/{len(process_data)}...")
            excitation = excitation_list[i]
            params0 = get_params0(baseline_params, alpha0, beta0, IDA_values, IDA_config, index = i, excitation = excitation, baseline_ = baseline, scale = scale, direction = "divide")
            
            excitation_data_i = excitation_data_filter(excitation_data, excitation)
            true_indices = [j for j, e in enumerate(excitation) if e]
            self_exc_idx = true_indices.index(i) if i in true_indices else None
            type = type_list[i]
            sol = hawkes_fitter(params0, T, process_data[i], IDA_config[i], excitation = excitation, excitation_data = excitation_data_i, type = type, include_results = False, algo = algo, baseline = baseline[i], param_bounds = param_bounds[i], scale = scale, vectorized = vectorized, self_exc_idx=self_exc_idx)
            results.append(sol_to_params(sol, IDA_config[i], excitation = excitation, baseline = baseline[i], scale = scale, direction = "multiply"))
        return results
    else:
        for i in range(len(process_data)):
            if indices_to_fit is not None and i not in indices_to_fit:
                continue
            print(f"Fitting model for process {i+1}/{len(process_data)}...")
            excitation = excitation_list[i]
            params0 = get_params0(baseline_params, alpha0, beta0, IDA_values, IDA_config, index = i, excitation = excitation, baseline_ = baseline, scale = scale, direction = "divide")
            
            excitation_data_i = excitation_data_filter(excitation_data, excitation)
            true_indices = [j for j, e in enumerate(excitation) if e]
            self_exc_idx = true_indices.index(i) if i in true_indices else None
            type = type_list[i]
            if algo == 'differential_evolution':
                sol = hawkes_fitter(params0, T, process_data[i], IDA_config[i], excitation = excitation, excitation_data = excitation_data_i, type = type, include_results = False, algo = algo, workers = workers, popsize = popsize, strategy = strategy, baseline = baseline[i], vectorized= vectorized, polisher = polisher, param_bounds = param_bounds[i], scale = scale, self_exc_idx=self_exc_idx)
                
                # Handle both DE and polished results if returned as dict
                if isinstance(sol, dict) and 'de' in sol and 'polished' in sol:
                    de_params = sol_to_params(sol['de'], IDA_config[i], excitation = excitation, baseline = baseline[i], scale = scale, direction = None)
                    polished_params = sol_to_params(sol['polished'], IDA_config[i], excitation = excitation, baseline = baseline[i], scale = scale, direction = "multiply")
                    results.append({'de': de_params, 'polished': polished_params})
                else:
                    # Fallback for non-dict results
                    results.append(sol_to_params(sol, IDA_config[i], excitation = excitation, baseline = baseline[i], scale = scale, direction = (None if polisher == 'False' else "multiply")))
            else:
                sol = hawkes_fitter(params0, T, process_data[i], IDA_config[i], excitation = excitation, excitation_data = excitation_data_i, type = type, include_results = False, algo = algo, baseline = baseline[i], param_bounds = param_bounds[i], scale = scale, vectorized = vectorized, self_exc_idx=self_exc_idx)
                results.append(sol_to_params(sol, IDA_config[i], excitation = excitation, baseline = baseline[i], scale = scale, direction = "multiply"))
        return results

############ GOODNESS OF FIT AND PLOTS ##########

# Helper functions --------------------------------------------------------------------------------
# TODO: Add the Markov recursive trick here
def Lambda_t(t, baseline_params_1d, alpha, beta, IDA_values, IDA_config, excitation_data_, baseline = "mu_exp"):
    excitation_data = [np.array(d) for d in excitation_data_]

    if baseline == "mu_exp":
        vu, gamma, eta =  baseline_params_1d[0], baseline_params_1d[1], baseline_params_1d[2] 
        if eta == 0:
            mu_integral = vu * t + gamma * t
        else:
            mu_integral = mu_integral_interval_exp(vu, gamma, eta, 0, t)
        
        for val, config in zip(IDA_values, IDA_config):
            interval_start = config['start']
            interval_end = min(config['end'], t)        
            if config['type'] == 'add':            
                mu_integral += val * (interval_end - interval_start) if interval_end > interval_start else 0
            elif config['type'] == 'mult':
                # We remove the original value and add in the scaled value
                integral_in_interval = mu_integral_interval_exp(vu, gamma, eta, interval_start, interval_end)
                mu_integral += integral_in_interval * val - integral_in_interval if interval_end > interval_start else 0
    
    if baseline == "mu_sigmoid":
        base, peak, b, c =  baseline_params_1d[0], baseline_params_1d[1], baseline_params_1d[2], baseline_params_1d[3]
        mu_integral = mu_integral_interval_sigmoid(base, peak, b, c, 0, t)
        
        for val, config in zip(IDA_values, IDA_config):
            interval_start = config['start']
            interval_end = min(config['end'], t)        
            if config['type'] == 'add':            
                mu_integral += val * (interval_end - interval_start) if interval_end > interval_start else 0
            elif config['type'] == 'mult':
                # We remove the original value and add in the scaled value
                integral_in_interval = mu_integral_interval_sigmoid(base, peak, b, c, interval_start, interval_end)
                mu_integral += integral_in_interval * val - integral_in_interval if interval_end > interval_start else 0

    phi_integral_val = 0
    for i, data_exc in enumerate(excitation_data):
        relevant = data_exc[data_exc < t]
        phi_integral_val += np.sum((alpha[i] / beta[i]) * (1 - np.exp(-beta[i] * (t - relevant))))

    return mu_integral + phi_integral_val

def _transform_single_sim(args):
    """Transform one simulation's event times and return inter-arrival durations."""
    data_i, exc_data_i, baseline_params_1d, alpha, beta, IDA_values, IDA_config_i, baseline = args
    transformed_i = [
        Lambda_t(t, baseline_params_1d, alpha, beta, IDA_values, IDA_config_i, excitation_data_=exc_data_i, baseline = baseline)
        for t in data_i
    ]
    durations = []
    for j in range(len(transformed_i) - 1):
        durations.append(transformed_i[j + 1] - transformed_i[j])
    return durations

def exponential_test_para(process_data, excitation_data, results, excitation_list, IDA_config, max_workers=None, baseline = "mu_exp"):
    """Parallel version of test_results: transforms data and runs exponential test for each process."""

    if max_workers is None:
        max_workers = min(len(process_data[0]), os.cpu_count() or 1)

    for i in range(len(process_data)):
        baseline_i = baseline[i]
        excitation = excitation_list[i]
        params = results[i]
        
        if baseline_i == "mu_exp":
            vu, gamma, eta = params[0], params[1], params[2]
            d = sum(excitation)
            idx = 0
            alpha = np.zeros(d)
            beta = np.zeros(d)
            for j in range(len(excitation)):
                if excitation[j]:
                    alpha[idx] = params[3 + idx]
                    beta[idx] = params[3 + d + idx]
                    idx += 1

            k = len(IDA_config[i])
            IDA_values = np.zeros(k)
            for j in range(k):
                IDA_values[j] = params[3 + 2 * d + j]

            # Build args for each simulation (one per worker)
            args_list = []
            for sim_idx in range(len(process_data[i])):
                exc_data_sim = [excitation_data[sim_idx][j]
                            for j in range(len(excitation)) if excitation[j]]
                args_list.append((
                    process_data[i][sim_idx], exc_data_sim,
                    [vu, gamma, eta], alpha, beta, IDA_values,
                    IDA_config[i], "mu_exp"
                ))

            with ProcessPoolExecutor(max_workers=max_workers,
                                    mp_context=mp.get_context("spawn")) as executor:
                all_durations_nested = list(executor.map(_transform_single_sim, args_list))

            # Flatten durations from all simulations
            all_durations = []
            for dur_list in all_durations_nested:
                all_durations.extend(dur_list)

            print(f"\n--- Process {i + 1} ---")
            exponential_test(all_durations) 

        if baseline_i == "mu_sigmoid":
            base, peak, b, c = params[0], params[1], params[2], params[3]
            d = sum(excitation)
            idx = 0
            alpha = np.zeros(d)
            beta = np.zeros(d)
            for j in range(len(excitation)):
                if excitation[j]:
                    alpha[idx] = params[4 + idx]
                    beta[idx] = params[4 + d + idx]
                    idx += 1

            k = len(IDA_config[i])
            IDA_values = np.zeros(k)
            for j in range(k):
                IDA_values[j] = params[4 + 2 * d + j]

            # Build args for each simulation (one per worker)
            args_list = []
            for sim_idx in range(len(process_data[i])):
                exc_data_sim = [excitation_data[sim_idx][j]
                            for j in range(len(excitation)) if excitation[j]]
                args_list.append((
                    process_data[i][sim_idx], exc_data_sim,
                    [base, peak, b, c], alpha, beta, IDA_values,
                    IDA_config[i], "mu_sigmoid"
                ))

            with ProcessPoolExecutor(max_workers=max_workers,
                                    mp_context=mp.get_context("spawn")) as executor:
                all_durations_nested = list(executor.map(_transform_single_sim, args_list))

            # Flatten durations from all simulations
            all_durations = []
            for dur_list in all_durations_nested:
                all_durations.extend(dur_list)

            print(f"\n--- Process {i + 1} ---")
            exponential_test(all_durations)   


def exponential_test(durations_):
    durations = np.array(durations_)
    # --- 1. Kolmogorov-Smirnov test against Exp(1) ---
    ks_stat, p_value = kstest(durations, 'expon', args=(0, 1))
    print(f"KS statistic: {ks_stat:.4f}")
    print(f"p-value:      {p_value:.4f}")

    # --- 2. Visual check ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram vs Exp(1) pdf
    x = np.linspace(0, min(durations.max(), 10), 200)
    axes[0].hist(durations, bins=50, density=True, alpha=0.6, label='Transformed durations')
    axes[0].plot(x, expon.pdf(x, scale=1), 'r-', lw=2, label='Exp(1) pdf')
    axes[0].set_xlabel('Duration')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Histogram vs Exp(1)')
    axes[0].legend()

    # QQ plot against Exp(1)
    n = len(durations)
    quantiles_empirical = np.sort(durations)
    quantiles_theoretical = expon.ppf(np.arange(1, n + 1) / (n + 1), scale=1)
    axes[1].scatter(quantiles_theoretical, quantiles_empirical, s=1, alpha=0.5)
    axes[1].plot([0, quantiles_theoretical.max()], [0, quantiles_theoretical.max()], 'r--', lw=2, label='y = x')
    axes[1].set_xlabel('Theoretical Exp(1) quantiles')
    axes[1].set_ylabel('Empirical quantiles')
    axes[1].set_title('QQ plot vs Exp(1)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()



# Main functions --------------------------------------------------------------------------------------

# Prints the parameters in a table
def print_results(result, IDA_config, excitation_list, print_it = False, baseline = "mu_exp", indices_fitted = None):

    results_dict = {}

    if indices_fitted is None:
        process_indices = list(range(len(result)))
    else:
        process_indices = list(indices_fitted)
        if len(process_indices) != len(result):
            raise ValueError(
                "indices_fitted must have the same length as result. "
                f"Got len(indices_fitted)={len(process_indices)} and len(result)={len(result)}"
            )

    for local_i, params in enumerate(result):
        proc_i = process_indices[local_i]

        if isinstance(baseline, str):
            baseline_i = baseline
        else:
            baseline_i = baseline[proc_i]

        process_name = f"Process {proc_i+1}"
        if baseline_i == "mu_exp":
            results_dict[process_name] = {
                'vu': params[0],
                'gamma': params[1],
                'eta': params[2]
            }
            # Add the alpha and beta parameters
            idx = 3
            for j in range(len(excitation_list[proc_i])):
                if excitation_list[proc_i][j] == True:
                    results_dict[process_name][f'alpha_{j+1}'] = params[idx]
                    idx += 1
                else:
                    results_dict[process_name][f'alpha_{j+1}'] = "N/A"
            
            d = sum(excitation_list[proc_i])
            idx = 3 + d
            for j in range(len(excitation_list[proc_i])):
                if excitation_list[proc_i][j] == True:
                    results_dict[process_name][f'beta_{j+1}'] = params[idx]
                    idx += 1
                else:
                    results_dict[process_name][f'beta_{j+1}'] = "N/A"
            # Add the IDA parameters
            k = len(IDA_config[proc_i])
            for j in range(k):
                results_dict[process_name][f'{IDA_config[proc_i][j]["name"]}'] = params[3 + 2*d + j]

        if baseline_i == "mu_sigmoid":
            results_dict[process_name] = {
                            'base': params[0],
                            'peak': params[1],
                            'b': params[2],
                            'c': params[3]
                        }
            # Add the alpha and beta parameters
            idx = 4
            for j in range(len(excitation_list[proc_i])):
                if excitation_list[proc_i][j] == True:
                    results_dict[process_name][f'alpha_{j+1}'] = params[idx]
                    idx += 1
                else:
                    results_dict[process_name][f'alpha_{j+1}'] = "N/A"
            
            d = sum(excitation_list[proc_i])
            idx = 4 + d
            for j in range(len(excitation_list[proc_i])):
                if excitation_list[proc_i][j] == True:
                    results_dict[process_name][f'beta_{j+1}'] = params[idx]
                    idx += 1
                else:
                    results_dict[process_name][f'beta_{j+1}'] = "N/A"
            # Add the IDA parameters
            k = len(IDA_config[proc_i])
            for j in range(k):
                results_dict[process_name][f'{IDA_config[proc_i][j]["name"]}'] = params[4 + 2*d + j]                

    # Create DataFrame and transpose to get processes as rows
    results_df = pd.DataFrame(results_dict).T
    if print_it:
        print(results_df.to_string(float_format=lambda x: f'{x:.4f}'))
    
    return results_df



# EXTRA HELPER FUNCTIONS ------------------------------------------------------------------------------
def process_data_to_excitation_data(process_data):
    return [list(row) for row in zip(*process_data)]

