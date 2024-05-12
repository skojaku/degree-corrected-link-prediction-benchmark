import argparse
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pickle as pkl
import os

parser = argparse.ArgumentParser(description="Process some integers.")

# Add arguments
parser.add_argument("-a", "--alpha", type=float, help="alpha", required=True)

parser.add_argument("-i", "--ini_string", type=str, help="initial string", required=True)

parser.add_argument("-st", "--start", type=int, help="starting state", required=True)

parser.add_argument("-s", "--sd", type=int, help="sd of proposal", required=True)

parser.add_argument("-e", "--ensembles", type=int, help="run number", required=True)


args = parser.parse_args()
alpha = args.alpha
ini_string = args.ini_string
start = args.start
sd = args.sd
e = args.ensembles


def mcmc_updater(curr_state1, curr_state2, likelihood, proposal_distribution):
    """ Propose a new state and compare the likelihoods
    
    Given the current state (initially random), 
      current likelihood, the likelihood function, and 
      the transition (proposal) distribution, `mcmc_updater` generates 
      a new proposal, evaluate its likelihood, compares that to the current 
      likelihood with a uniformly samples threshold, 
    then it returns new or current state in the MCMC chain.

    Args:
        curr_state (float): the current parameter/state value
        likelihood (function): a function handle to compute the likelihood
        proposal_distribution (function): a function handle to compute the 
          next proposal state

    Returns:
        (tuple): either the current state or the new state
          and its corresponding likelihood
    """
    # Generate a proposal state using the proposal distribution
    # Proposal state == new guess state to be compared to current
    proposal_state1,proposal_state2 = proposal_distribution(curr_state1,curr_state2)

    prop_likeli = likelihood(curr_state1,curr_state2,proposal_state1,proposal_state2)

    # Calculate the acceptance criterion
    r_f = numerator(proposal_state1,proposal_state2)/numerator(curr_state1,curr_state1)
    r_g = likelihood(proposal_state1,proposal_state2,curr_state1,curr_state2)/prop_likeli

    accept_crit = min(1,r_f * r_g)
    if accept_crit == 1:
        return proposal_state1,proposal_state2
    else:
        # Generate a random number between 0 and 1
        accept_threshold = np.random.uniform(0, 1)
        if accept_crit > accept_threshold:
            return proposal_state1,proposal_state2
        else:
            return curr_state1,curr_state2


def metropolis_hastings(
        likelihood, proposal_distribution, initial_state1, initial_state2,
        num_samples, stepsize=0.5, burnin=0.2):
    """ Compute the Markov Chain Monte Carlo

    Args:
        likelihood (function): a function handle to compute the likelihood
        proposal_distribution (function): a function handle to compute the 
          next proposal state
        initial_state (list): The initial conditions to start the chain
        num_samples (integer): The number of samples to compte, 
          or length of the chain
        burnin (float): a float value from 0 to 1.
          The percentage of chain considered to be the burnin length

    Returns:
        samples (list): The Markov Chain,
          samples from the posterior distribution
    """
    samples = []

    # The number of samples in the burn in phase
    idx_burnin = int(burnin * num_samples)

    # Set the current state to the initial state
    curr_state1,curr_state2 = initial_state1,initial_state2
    curr_likeli = likelihood(curr_state1,curr_state2,curr_state1,curr_state2)

    for i in range(num_samples):
        # The proposal distribution sampling and comparison
        #   occur within the mcmc_updater routine
        curr_state1, curr_state2 = mcmc_updater(
            curr_state1=curr_state1,
            curr_state2=curr_state2,
            likelihood=likelihood,
            proposal_distribution=proposal_distribution
        )

        # Append the current state to the list of samples
        if i >= idx_burnin:
            # Only append after the burnin to avoid including
            #   parts of the chain that are prior-dominated
            samples.append([curr_state1,curr_state2])

    return samples

def likelihood(conditioned_on1,conditioned_on2,x1,x2,sd=sd):
    return multivariate_normal([conditioned_on1, conditioned_on2], [[sd, 0], [0, sd]]).pdf([x1,x2])

def proposal_distribution(x1,x2, stepsize=1,sd=sd):
    # Select the proposed state (new guess) from a Gaussian distribution
    #  centered at the current state, within a Guassian of width `stepsize`
    return multivariate_normal([x1, x2], [[sd, 0], [0, sd]]).rvs()

def numerator(x1,x2,alpha=alpha,x_min=1,ini_string=ini_string):
    if x1 < 1 or x2 < 1:
        return 0
    if ini_string == "splus":
        return ( x1 * (alpha-1)/(x_min) ) * ( (x_min/x1)**(alpha) ) * ( x2 * (alpha-1)/(x_min) ) * ( (x_min/x2)**(alpha) ) 
    if ini_string == "sminus":
        return ( (alpha-1)/(x_min) ) * ( (x_min/x1)**(alpha) ) * ( (alpha-1)/(x_min) ) * ( (x_min/x2)**(alpha) )  

initial_state1,initial_state2 = start,start  # Trivial case, starting at the mode of the likelihood
num_samples = int(1e6)
burnin = 0.2

samples = metropolis_hastings(
    likelihood,
    proposal_distribution,
    initial_state1,
    initial_state2,
    num_samples,
    burnin=burnin
)

data_dir = "/data/sg/racball/link-prediction/notebooks/rachith/metropolis-hastings/"
with open(os.path.join(data_dir,f"{ini_string}_alpha{alpha}_start{start}_normalscale{sd}_ensembles{e}.pkl"),"wb") as f:
    pkl.dump(samples,f)

