import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
from ellipse import LsqEllipse
from math import comb
from matplotlib.patches import Ellipse
import scipy.optimize as sco
from scipy.optimize import fmin
from scipy.optimize import curve_fit
import scipy
import json
import glob
import os
import allantools
from tqdm.notebook import trange, tqdm
import warnings
warnings.filterwarnings("ignore")
rc('font',**{'family':'sans-serif','sans-serif':['Fira Sans'],'size':14,'style':'normal'})
rc('text', usetex=False)


BLUE = 'xkcd:pastel blue'
RED = 'xkcd:pastel red'
GREEN = 'xkcd:pastel green'
YELLOW ='xkcd:pastel yellow'
PURPLE = 'xkcd:pastel purple'
DBLUE = 'xkcd:light navy blue'
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[BLUE, RED, GREEN, YELLOW,PURPLE,DBLUE])




f0 = 429e12
h_const = 6.626e-34
og_directory = os.getcwd()

from scipy.optimize import curve_fit

def fit_sine(t, y, yerr=None):
    """
    Fit y(t) ≈ A * sin(2π f t + phi) + offset

    Parameters
    ----------
    t : array
        Time (or x) values.
    y : array
        Data values.
    yerr : array, optional
        1σ errors on y. If provided, weighted least-squares is used.

    Returns
    -------
    dict : best-fit parameters, 1σ errors, and model function.
    """
    def model(t, A, f, phi, offset):
        return A * np.sin(2*np.pi*f*t + phi) + offset

    # --- crude guesses ---
    guess_A = (np.max(y) - np.min(y)) / 2
    guess_offset = np.mean(y)
    
    # frequency guess from FFT
    freqs = np.fft.rfftfreq(len(t), d=(t[1]-t[0]))
    fft_mag = np.abs(np.fft.rfft(y - guess_offset))
    guess_f = freqs[np.argmax(fft_mag[1:]) + 1] if len(freqs) > 1 else 1.0
    
    guess_phi = 0
    p0 = [guess_A, guess_f, guess_phi, guess_offset]

    # --- fit ---
    popt, pcov = curve_fit(
        model, t, y, p0=p0,
        sigma=yerr, absolute_sigma=True if yerr is not None else False
    )
    perr = np.sqrt(np.diag(pcov))

    names = ["amplitude", "frequency", "phase", "offset"]

    return {
        name: val for name, val in zip(names, popt)
    } | {
        f"{name}_err": err for name, err in zip(names, perr)
    } | {
        "model": lambda t: model(t, *popt)
    }


def allan_errors(frac_frequencies, allan_dev, tau, ramsey_time, dead_time,
                     method="oadev", noise_process="whitefm"):

        #Getting our common varibles needed
        n = len(frac_frequencies)
        m = tau/(ramsey_time+dead_time)      
        
        #Calculate edf for chi-squared calculation based on noise process
        if method == "oadev":
            if noise_process == "whitepm":
                edf = (n + 1)*(n - 2*m)/(2*(n - m))
            elif noise_process == "whitefm":
                edf = (4*m**4/(4*m**2+5))*(3*(n-1)/(2*m) - 2*(n-2)/n)
            else:
                print("Unable to calculate error for given error type")
                
        if method == "totdev":
            if noise_process == "whitepm" or noise_process=="whitefm":
                b = 1.50
                c= 0.0
                big_t = len(frac_frequencies)*(ramsey_time+dead_time) #Total measurement time
                edf = b*(big_t/tau) - c
            else:
                print("Unable to calculate error for given error type")
                
        else:
            if noise_process == "whitepm" or noise_process == "whitefm":
                edf = (n + 1)*(n - 2*m)/(2*(n - m))
            else:
                print("Unable to calculate error for given error type")
        
        #Calculating Chi Squared Table Values
        lower_chi = scipy.stats.chi2.isf(0.16, df=edf)
        upper_chi = scipy.stats.chi2.isf(0.84, df=edf)
        #Making sure to use variance in our calculations and report our CI in terms of deviataion
        lower_cutoff = allan_dev - np.sqrt(allan_dev**2*(edf/lower_chi))
        upper_cutoff = np.sqrt(allan_dev**2*(edf/upper_chi)) - allan_dev
        
        return(lower_cutoff, upper_cutoff)    



def monoExp(x,t):
    return np.exp(-(1/t) * x) 

def monoExpN(x,t,c):
    return c*np.exp(-(1/t) * x) 


def get_parent_dir(directory):
    import os
    return os.path.dirname(directory)
def QPN(tau,N,C,t_ramsey,t_dead):
    QPN = np.sqrt(2)/(2*np.pi*f0*C*t_ramsey)*(np.sqrt((t_ramsey+t_dead)/(N*(tau))))
    return QPN
def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )
def likelihood(X,Y,P1,P2,n):
    N1 = X*n
    l1 = gaussian(N1,P1*n,np.sqrt(n*P1*(1-P1)))
    N2 = Y*n
    l2 = gaussian(N2,P2*n,np.sqrt(n*P2*(1-P2)))
    return l1*l2
        
    
def make_ellipse(phi,C1,xc,yc):
    phase_com = np.linspace(0,2*np.pi,200)
    P1 = 1/2*(1+np.cos(phase_com))
    P2 = 1/2*(1+np.cos(phase_com+phi))   
    return P1,P2

def likelihood_sum(phi, C1, C2, X, Y, xc, yc, n1, n2, phase_jitter_std = 0 ):
    phase_com = np.linspace(0, 2 * np.pi, 20000)

    # Effective contrasts from jitter
    C1_eff = np.abs(C1) * np.exp(-0.5 * phase_jitter_std**2)
    C2_eff = np.abs(C2) * np.exp(-0.5 * phase_jitter_std**2)

    # Raw P1, P2
    P1 = 0.5 * (1 + C1_eff * np.cos(phase_com))
    P2 = 0.5 * (1 + C2_eff * np.cos(phase_com + phi))

    # Clip to avoid exact 0 or 1
    eps = 1e-10
    P1 = np.clip(P1, eps, 1 - eps)
    P2 = np.clip(P2, eps, 1 - eps)

    # Repeat shapes
    X = np.repeat(X[:, None], len(phase_com), axis=1)
    Y = np.repeat(Y[:, None], len(phase_com), axis=1)
    P1 = np.repeat(P1[None, :], X.shape[0], axis=0)
    P2 = np.repeat(P2[None, :], Y.shape[0], axis=0)

    # QPN variance (in counts)
    sigma1_qpn = np.sqrt(n1 * P1 * (1 - P1))
    sigma2_qpn = np.sqrt(n2 * P2 * (1 - P2))

    # Jitter variance (in counts)
    # derivative dP/dphi = -0.5 * C_eff * sin(...)
    jitter1 = 0.25 * (C1_eff**2) * (np.sin(phase_com)**2) * (n1**2) * (phase_jitter_std**2)
    jitter2 = 0.25 * (C2_eff**2) * (np.sin(phase_com + phi)**2) * (n2**2) * (phase_jitter_std**2)

    # Match shapes for broadcasting
    jitter1 = np.repeat(jitter1[None, :], X.shape[0], axis=0)
    jitter2 = np.repeat(jitter2[None, :], Y.shape[0], axis=0)

    # Total sigma (counts)
    sigma1 = np.sqrt(sigma1_qpn**2 + jitter1)
    sigma2 = np.sqrt(sigma2_qpn**2 + jitter2)

    sigma1 = np.where(sigma1 == 0, 1e-12, sigma1)
    sigma2 = np.where(sigma2 == 0, 1e-12, sigma2)

    # Gaussian likelihoods
    likes1 = gaussian(X * n1, P1 * n1, sigma1)
    likes2 = gaussian(Y * n2, P2 * n2, sigma2)
    likes = likes1 * likes2

    # Log likelihood
    loglikes = np.log(np.sum(likes, axis=1) + 1e-300)
    total_likelihood = np.sum(loglikes)
    return total_likelihood


def inverse(x, c):
    return x**(-.5) * c
    


def MLE_optimized_err(data, n1, n2, offset, pll_noise = 0 , g=False, guess=None, cov=False):
    import numpy as np
    import scipy.optimize as sco

    P1 = data[:, 0]
    P2 = data[:, 1]

    # Objective function: negative log-likelihood
    def objective(args):
        [p1] = args
        return -likelihood_sum(p1, 1, 1, np.array(P1), np.array(P2), 0.5, 0.5, n1, n2, phase_jitter_std = pll_noise)

    # Initial guess
    if not g or guess is None:
        guess1 = np.array([offset])
    else:
        guess1 = guess + 1e-4  # Replace epsilon with 1e-4 if not defined

    # Optimization
    opt = sco.minimize(
        objective,
        guess1,
        method='Nelder-Mead',
        options={'disp': False, 'xatol': 1e-9}
    )

    # Wrap and clean phase estimate
    phi_MLE = np.abs(opt.x[0])
    if phi_MLE > np.pi:
        phi_MLE = 2 * np.pi - phi_MLE

    # Compute standard error (if requested)
    if cov:
        h = 1e-5
        try:
            f_plus  = objective([phi_MLE + h])
            f_minus = objective([phi_MLE - h])
            f0      = objective([phi_MLE])
            second_deriv = (f_plus - 2 * f0 + f_minus) / (h ** 2)

            if second_deriv <= 0 or np.isnan(second_deriv):
                phase_jitter_std = np.nan
                print("⚠️ Warning: curvature non-positive or NaN — error estimate invalid.")
            else:
                phase_jitter_std = np.sqrt(1 / second_deriv)
        except:
            phase_jitter_std = np.nan
            print("⚠️ Error computing second derivative — returning NaN for uncertainty.")

        return phi_MLE, phase_jitter_std

    return np.array([phi_MLE])

    
    
def adevffd(frac_freq_diff,t_ramsey,t_dead,N,C):
    t= np.linspace(1,150)
    r = 1/(t_ramsey+t_dead)
    t= np.linspace(1,len(frac_freq_diff)/r)
    #(t2, ad, ade, adn) = allantools.oadev(frac_freq_diff/f0, rate=r, data_type="freq", taus=t)
    (t2, ad, ade, adn) = allantools.adev(frac_freq_diff, rate=r, data_type="freq", taus=t)
    popt, pcov = sco.curve_fit(inverse, t2, ad,sigma = ade,p0 = [1e-17],absolute_sigma = True )
    return (popt[0],np.sqrt(pcov[0][0]))

def QPN_scale(C,phi):
    N =100 
    theta = np.linspace(0+10e-2,2*np.pi+10e-2,N)
    x = C/2*np.cos(theta)
    y = C/2*np.cos(theta+phi)
    var_x = (1/2-x)*(1/2+x)
    var_y = (1/2-y)*(1/2+y)
    integral = np.sum(1/(var_x/np.sin(theta)**2+var_x/np.sin(theta+phi)**2))/100    
    return 2/integral


def jackknife_ffd(data,t_ramsey,t_dead,n1,n2,guess = None, g = False,phases = False):
    phi_i = []
    C_i = []
    guesses =[]
    for i in (range(len(data))):
        jk_data = np.delete(data,i,axis = 0)        
        [opt_jk,guess1] = MLE_optimized(jk_data,n1,n2,g= g, guess = guess)
        phi_D = opt_jk[0]
        guesses.append(guess1)
        
        phi_i.append(phi_D)
        C_i.append(opt_jk[1])
    guesses = np.transpose(np.array(guesses))
    
    phi_i = np.array(phi_i)
    shots = len(data)
    phi_jk_mean = np.mean(phi_i)*np.ones(shots)
    phi_jk_i = (phi_jk_mean*shots)-(phi_i*(shots-1))
    frac_freq_diff = np.array(phi_jk_i/t_ramsey/(2*np.pi))/f0
    
    if phases==True:
        if g:
            return [frac_freq_diff,phi_i,C_i,guess1]
        else:
            return [frac_freq_diff,phi_i,C_i]
    
    return frac_freq_diff


def jackknife_phi(data,n1,n2,offset):
    phi_i = []
    for i in (range(len(data))):
        jk_data = np.delete(data,i,axis = 0)        
        opt_jk= MLE_optimized(jk_data,n1,n2,offset)
        phi_i.append(opt_jk)

    
    phi_i = np.array(phi_i)
    return np.std(phi_i)

def QPN_sim(p,n,C=1):
    return np.random.binomial(n,p)/n


def dd_window(size, n_pi_pulses):
    """
    Evenly spaced window alternating between 1 and -1.
    
    Parameters
    ----------
    size : int
        Total length of the array.
    n_switches : int
        Number of times to switch sign.
    """
    # total number of segments
    segments = n_pi_pulses + 1
    
    # index for each point
    idx = np.arange(size)
    
    # figure out which segment each index belongs to
    seg = (idx * segments) // size   # integer division
    
    # alternate signs: start at +1, flip each segment
    return (-1) ** seg


