from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from numpy import poly1d, polyfit
from scipy.stats import poisson
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

from sklearn.decomposition import NMF, PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE, Isomap
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)

def gen_self_connected_only(nrows, ncols):
    """
    Generates connection matrix of only self-connected nodes.
    This connectivity only allows infections within each node (compartment).
    
    A nrows x ncols 2D universe is assumed.
    
    Parameters
    ----------
    nrows   : int
            number of rows in 2D rectilinear universe.
    ncols   : int
            number of columns in 2D rectilinear universe.
    """
    num_nodes = nrows*ncols
    conn = np.zeros((num_nodes,num_nodes))
    for r in range(num_nodes):
        conn[r,r] = 1
    return conn

def gen_ring_connection(nrows, ncols, deg_vec=[1], p0=None):
    """
    Generates pair-wise connection matrix of nodes connected in a ring topology.
    
    A nrows x ncols 2D universe is connected in a row-major format: 
    the last column of each row is connected to the first column of the next row. 
    
    Parameters
    ----------
    nrows   : int
            number of rows in 2D rectilinear universe.
    ncols   : int
            number of columns in 2D rectilinear universe.
    deg_vec : array_like, optional
            List of incremental addresses of connected neighbors. 
            e.g. [1,2] will connect node n to (n+1 and n+2) 
    p0 :    {None, float}, optional
            Fraction of connections that will randomly rewired.
    """
    num_nodes = nrows*ncols
    conn = np.zeros((num_nodes,num_nodes))
    connections = []
    for r in range(num_nodes):
        for d in deg_vec:
            connections.append([r,(r+d)%num_nodes])
    if p0 is not None:
        replace = np.random.choice([0,1], p=[1-p0, p0], size=len(connections))
        replace_loc = np.where(replace==1)[0]
        for r in replace_loc:
            [n0,n1] = connections[r] 
            target_list = list(range(num_nodes))
            target_list.remove(n0)
            connections[r] = [n0, np.random.choice(target_list)]
    for vals in connections:
        conn[vals[0], vals[0]] = 1 
        conn[vals[1], vals[1]] = 1
        conn[vals[0], vals[1]] = 1 
        conn[vals[1], vals[0]] = 1 
    return conn

def gen_rand_connection(nrows, ncols, p0=0.01): 
    """
    Generates connectivity matrix between random nodes in a 2D universe
    whose shape is (nrows, ncols).
    
    Parameters
    ----------
    nrows   : int
            number of rows in 2D rectilinear universe.
    ncols   : int
            number of columns in 2D rectilinear universe.
    """
    num_nodes = nrows*ncols
    conn = np.random.choice([0,1], p=[1-p0, p0], size=num_nodes*num_nodes).reshape(num_nodes, -1)
    conn += conn.T
    conn //= 2
    for r in range(num_nodes):
        conn[r,r] = 1
    return conn

def gen_connection_from_neigh_list(nrows, ncols, neigh_list, periodic=True):
    """
    Generates connectivity matrix of a (nrows, ncols) 2D universe
    whose nodes are connected to their von Neumann 4-neighbors.
    
    Parameters
    ----------
    nrows   : int
            number of rows in 2D rectilinear universe.
    ncols   : int
            number of columns in 2D rectilinear universe.
    neigh_list: array_like
            List of (+row, +col) addresses of connected neighbors 
    periodic : bool
            Decides if nodes are connected like a torus:
            top-row connected to bottom-row, 
            leftmost-column connected to rightmost-column
    """
    conn = np.zeros((nrows*ncols, nrows*ncols))
    for r in range(nrows):
        for c in range(ncols):
            conn[r*ncols+c, r*ncols+c] = 1
            org = r*ncols + c
            dest = []
            for d_r,d_c in neigh_list: 
                rr = (r+d_r)%nrows
                cc = (c+d_c)%ncols
                if periodic:
                    dest.append(rr*ncols + cc)
                else:
                    if (rr == r+d_r)&(cc == c+d_c):
                        dest.append(rr*ncols + cc)
            for d in dest:
                conn[org, d] = 1
    return conn


def gen_von_neumann(nrows, ncols, periodic=True):
    """
    Generates connectivity matrix of a (nrows, ncols) 2D universe
    whose nodes are connected to their von Neumann 4-neighbors.
    
    Parameters
    ----------
    nrows   : int
            number of rows in 2D rectilinear universe.
    ncols   : int
            number of columns in 2D rectilinear universe.
    periodic : bool
            Decides if nodes are connected like a torus:
            top-row connected to bottom-row, 
            leftmost-column connected to rightmost-column
    """
    neigh_list = [[-1,0],[1,0],[0,-1],[0,1]]
    return gen_connection_from_neigh_list(nrows, ncols, neigh_list, periodic=periodic)
   
def gen_moore(nrows, ncols, periodic=True):
    """
    Generates connectivity matrix of a (nrows, ncols) 2D universe
    whose nodes are connected to their Moore 8-neighbors.
    
    Parameters
    ----------
    nrows   : int
            number of rows in 2D rectilinear universe.
    ncols   : int
            number of columns in 2D rectilinear universe.
    periodic : bool
            Decides if nodes are connected like a torus:
            top-row connected to bottom-row, 
            leftmost-column connected to rightmost-column
    """
    neigh_list = [[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1], [1,1], [1,-1]]
    return gen_connection_from_neigh_list(nrows, ncols, neigh_list, periodic=periodic)

def gen_uniform_pop(num_nodes, avg_pop=1000):
    """
    Generates a uniform list of populations across num_nodes.
    
    Parameters
    ----------
    num_nodes : int
    avg_pop   : int
                Population size per node
    """
    pop = avg_pop*np.ones(num_nodes)
    return pop

def gen_poisson_pop(num_nodes, avg_pop=1000):
    """
    Generates a uniform list of populations across num_nodes.
    
    Parameters
    ----------
    num_nodes : int
    avg_pop   : int
                Population size per node
    """
    pop = poisson.rvs(avg_pop, size=num_nodes)
    return pop

def my_single_imshow(to_plot, myfiglabel="generic_label", 
                     figsize=(6,6),
                     xlabel="",ylabel="",title=""):
    """
    Imshow that configures input array to aspect ratio 1,
    and adds colorbar to the right.
    
    Parameters
    ----------
    to_plot     : 2D numpy array
    figsize     : tuple (x,y) , optional
    myfiglabel  : string, optional
    xlabel      : string, optional
    ylabel      : string, optional
    title       : string, optional
    
    """
    aspect_r = to_plot.shape[1]/to_plot.shape[0]
    fig = plt.figure(myfiglabel, figsize)
    grid = ImageGrid(fig, 111, nrows_ncols=(1,1), 
                   axes_pad=0.1, cbar_mode='single')
    grid[0].set_xlabel(xlabel)
    grid[0].set_ylabel(ylabel)
    grid[0].set_title(title)
    im = grid[0].imshow(to_plot, aspect=aspect_r, cmap='inferno')
    plt.colorbar(im, cax=grid.cbar_axes[0])
    
    
def pca_and_view(input_arr, num_components=12, 
                 plot=True,
                 figsize=(15,10), 
                 xlabel1="nodes", 
                 ylabel1="components", 
                 title1="'Eigenvectors' of covariance matrix",
                 ylabel2="components", 
                 xlabel2="explained variance ratio", 
                 title2="'Eigenvalues'"):
    pca_model = PCA(n_components=num_components, random_state=0)
    transformed_input = pca_model.fit_transform(input_arr)
    if plot:
        fig,axes = plt.subplots(1,1,figsize=figsize, squeeze=True)
        to_plot = pca_model.components_
        aspect_r = to_plot.shape[1]/to_plot.shape[0]/3.
        #im = axes.imshow(to_plot, origin='left', aspect=aspect_r)
        im = axes.imshow(to_plot, origin='left')
        axes.set_xlabel(xlabel1)
        axes.set_ylabel(ylabel1)
        axes.set_title(title1)
        div = make_axes_locatable(axes)
        cax = div.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax0 = div.append_axes("left", size=2, pad=0.1)

        ax0.plot(pca_model.explained_variance_ratio_, np.arange(num_components), 'x-')
        ax0.set_xlabel(xlabel2)
        ax0.set_ylabel(ylabel2)
        ax0.set_title(title2)

    return (pca_model, transformed_input)

def view_pca_2(pca_model, universe_shape, show_shape=(3,3), punit=3): 
    num_comp_to_show = show_shape[0]*show_shape[1] 

    fig1 = plt.figure("PCA1", figsize=(3*show_shape[0], 2))
    plt.plot(pca_model.explained_variance_ratio_, 'k-x')
    plt.xlabel('PCA components')
    plt.ylabel('explained variance ratio')

    fig2 = plt.figure("PCA2", figsize=(punit*show_shape[0], punit*show_shape[1]))
    grid1 = ImageGrid(fig2, 111, nrows_ncols=show_shape, cbar_mode='each', 
                  label_mode="L", axes_pad=[0.5,0.4], cbar_pad=0, share_all=True,aspect=True)
    
    for n,g in enumerate(grid1):
        im = g.imshow(pca_model.components_[n].reshape(*universe_shape))
        g.set_title("Explained variance ratio: {:0.2f}".format(pca_model.explained_variance_ratio_[n]))
        g.cax.colorbar(im)
    
def view_components(model, universe_shape, show_shape=(3,3), punit=3): 
    num_comp_to_show = show_shape[0]*show_shape[1] 

    fig2 = plt.figure("model", figsize=(punit*show_shape[0], punit*show_shape[1]))
    grid1 = ImageGrid(fig2, 111, nrows_ncols=show_shape, cbar_mode='each', 
                  label_mode="L", axes_pad=[0.5,0.4], cbar_pad=0, share_all=True,aspect=True)
    
    for n,g in enumerate(grid1):
        im = g.imshow(model.components_[n].reshape(*universe_shape), cmap='PiYG')
        #g.set_title("Explained variance ratio: {:0.2f}".format(pca_model.explained_variance_ratio_[n]))
        g.cax.colorbar(im)
        
class sir_universe(object):
    def __init__(self, universe_sh=(5,5), alpha=0.1, beta=0.8, gamma=0.2,
                 conn=None, init_inf_frac=0.1, 
                 avg_pop_per_node=1000000, pop_type='poisson'):
        self.universe_sh = universe_sh
        self.num_nodes = universe_sh[0]*universe_sh[1]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.conn = conn
        self.beta_arr = beta*self.conn
        self.state = []
        self.avg_pop_per_node = avg_pop_per_node
        self.pop_type = pop_type
        self.init_inf_frac = init_inf_frac
        self.state_dict = {"S":0, "I":1, "R":2}
        self.inf = None
        self.sus = None
        self.rec = None
        self.pop = None
        self.np_state = None
        
    def gen_pop(self):
        if self.pop_type == 'poisson':
            self.pop = gen_poisson_pop(self.num_nodes, avg_pop=self.avg_pop_per_node)
        else:
            self.pop = gen_uniform_pop(self.num_nodes, avg_pop=self.avg_pop_per_node)
            
    def gen_infection(self, single_site=True):
        if single_site:
            infected_node = np.random.choice(self.num_nodes)
            self.inf = np.zeros(self.num_nodes)
            self.inf[infected_node] = np.random.rand()*self.pop[infected_node]
        else:
            seed1 = np.random.rand(self.num_nodes)
            seed2 = np.random.rand(self.num_nodes)
            self.inf = seed1*(seed2>(1.-self.init_inf_frac))*self.pop
            self.inf = poisson.rvs(self.inf).astype('float')
    
    def clip_states(self):
        for a in [self.sus, self.inf, self.rec]:
            np.clip(a, 0, self.pop, out=a)
        
    def initialize(self):
        self.gen_pop()
        self.gen_infection()
        self.sus = (self.pop - self.inf)
        self.rec = self.pop - self.sus - self.inf
        self.clip_states()
        self.norm = np.matmul(self.conn, self.pop)
        self.state = [[self.sus.copy(), self.inf.copy(), self.rec.copy()]]
        
    def iterate(self, num_iters, verbose=False):
        t0 = time.time()
        for n in range(num_iters):
            force = np.matmul(self.beta_arr, self.inf)/self.norm
            new_inf = poisson.rvs(force*self.sus).astype('float')
            new_rec = poisson.rvs(self.gamma*self.inf).astype('float')
            new_sus = poisson.rvs(self.alpha*self.rec).astype('float')
            np.clip(new_inf, 0, self.sus, out=new_inf)
            np.clip(new_rec, 0, self.inf, out=new_rec)
            np.clip(new_sus, 0, self.rec, out=new_sus)
            self.sus += -new_inf + new_sus
            self.inf += new_inf - new_rec
            self.rec = self.pop - self.sus - self.inf
            self.clip_states()
            self.state.append([self.sus.copy(), self.inf.copy(), self.rec.copy()])    
        self.np_state = np.asarray(self.state)  
        if verbose:
            print("{:d} iterations took {:0.3f}s".format(num_iters, time.time()-t0))
        
    def show_state_vs_time(self, logscale=False):
        tot_pop = self.pop.sum()
        tot_sus = self.np_state[:,0].sum(axis=1)/tot_pop
        tot_inf = self.np_state[:,1].sum(axis=1)/tot_pop 
        tot_rec = self.np_state[:,2].sum(axis=1)/tot_pop
        asymp_sus = self.gamma/self.beta
        asymp_inf = self.alpha*(self.beta-self.gamma)/self.beta/(self.alpha+self.gamma)
        asymp_rec = 1 - asymp_sus - asymp_inf
        fig, axes = plt.subplots(1,3, figsize=(20,5))
        axes[0].plot(tot_sus, color='r', label="susceptible")
        axes[0].plot(tot_inf, color='teal', label="infected")
        axes[0].plot(tot_rec, color='b', label="recovered")
        asymp_dict = {'$S_\inf$':[asymp_sus,'lightsalmon'], 
                      '$I_\inf$':[asymp_inf,'darkseagreen'], 
                      '$R_\inf$':[asymp_rec,'mediumslateblue']}
        for k,v in asymp_dict.items():
            if v[0] > 0:
                axes[0].axhline(y=v[0], lw=1.5, c=v[1], label=k)
        if logscale:
            axes[0].set_yscale('log')
            axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), 
                           ncol=3, fancybox=True, shadow=True)
        else:
            axes[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), 
                           ncol=3, fancybox=True, shadow=True)
        axes[0].set_xlabel('iteration')
        axes[0].set_ylabel('fraction')

        axes[1].plot(tot_sus,tot_inf)
        axes[1].set_xlabel('susceptible fraction')
        axes[1].set_ylabel('infected fraction')
        axes[1].set_yscale('log')
        axes[1].set_xscale('log')

        axes[2].plot(tot_sus,tot_rec)
        axes[2].set_xlabel('susceptible fraction')
        axes[2].set_ylabel('recovered fraction')
        axes[2].set_yscale('log')
        axes[2].set_xscale('log')
        
    def show_snapshots(self, figsize=(25,25), nrows_ncols=(8,8), single_cbar=False, view_state="I"):
        fig = plt.figure(figsize=figsize)

        if single_cbar:
            grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols,
                             cbar_mode='single', axes_pad=(0.1,0.3), cbar_pad=0.1, cbar_size="2%")
            loc = np.linspace(0, len(self.np_state)-1, len(grid)).astype(int)
            for n,g in zip(loc, grid):
                to_plot = self.np_state[n,self.state_dict[view_state]].reshape(*self.universe_sh) 
                im = g.imshow(to_plot, cmap='inferno', vmax=self.pop.max())
                g.set_title("Iteration {:d}".format(n))
                g.cax.colorbar(im)
        else:
            grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols,
                             cbar_mode='each', axes_pad=(0.4,0.3), cbar_pad=0., cbar_size="3%")
            loc = np.linspace(0, len(self.np_state)-1, len(grid)).astype(int)
            #loc = np.argsort(np_state[:,state_dict[view]].sum(axis=1))[-1::-1][:len(grid)] 
            #re_loc = np.argsort(loc)
            for n,g in zip(loc, grid):
                to_plot = self.np_state[n,self.state_dict[view_state]].reshape(*self.universe_sh) 
                im= g.imshow(to_plot, cmap='inferno')
                g.set_title("Iteration {:d}".format(n))
                g.cax.colorbar(im)
                
    def compute_r0(self, fit_t_range=[0,5], plot=True, ax=None):
        total_population = self.pop.sum()
        total_infected = (self.np_state[:,self.state_dict["I"]]).sum(axis=1)/total_population
        total_susceptible = (self.np_state[:,self.state_dict["S"]]).sum(axis=1)/total_population

        try:
            full_t = np.arange(0, len(total_infected))
            y = np.log(total_infected)[fit_t_range[0]:fit_t_range[1]]
            t = full_t[fit_t_range[0]:fit_t_range[1]]

            fit_params = polyfit(t, y, 1)
            func = poly1d(fit_params)
            doubling_time = np.log(2.)/fit_params[0]
            r0 = 1. + fit_params[0]/(self.gamma)
            fit_res = [np.exp(fit_params[0]*tt)*np.exp(fit_params[1]) for tt in t]

            if plot:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(total_infected, 'k', lw=0.5, label='simuations')
                ax.set_yscale('log')
                ax.set_ylabel('fraction of total population infected')
                ax.set_xlabel('iteration')

                ax.plot(t, total_infected[fit_t_range[0]:fit_t_range[1]], 'rx', label='used for fitting', markersize=6)
                ax.plot(t, fit_res, 'r', label='fit-results', lw=2)
                ax.legend(loc='upper right')
                ax.annotate("Doubling time: {:0.3f} iterations".format(doubling_time), 
                            xy=(t[1], fit_res[0]))
                ax.annotate("R_0: {:0.3f}".format(r0), 
                            xy=(t[2], 1.5*fit_res[2]))
            return (doubling_time, r0)
        except:
            print("Fit did not work!")