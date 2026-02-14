#Imports
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from math import ceil, log10



# Functions

def plot_diffusion(metadata):
    """
        Short description of what the function does.

        Parameters
        ----------
        metadata : dict
            Description of the input.

        Returns
        -------
        None
        """
    for node in metadata['nodes']:
        plt.plot(metadata['time_array'], metadata['data'][node, :], label=str(node))
        plt.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='small')
    plt.xlabel('Time (t)')
    plt.ylabel('$u_i(t)$')
    plt.title(metadata['title'])
    plt.show()


# Still not working properly...
def show_diffusion(metadata): 
    """
        Short description of what the function does.

        Parameters
        ----------
        metadata : dict
            Description of the input.

        Returns
        -------
        None
        """
    ceiling = 10**ceil(log10(metadata['data'][:,0].max())) # round up to the nearest power of 10
    title = metadata['title'].partition('diffusion')[0] + 'diffusion'
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(metadata['graph'])
    colorbar = ScalarMappable(cmap='turbo', norm=Normalize(vmin=0, vmax=ceiling))
    fig.colorbar(colorbar, ax=ax)

    for i in range(metadata['data'].shape[1]):
        ax.clear()
        nx.draw(metadata['graph'],
                pos=pos,
                with_labels=True,
                node_size=750,
                node_color=metadata['data'][:,i],
                vmin=0,
                vmax=ceiling,
                cmap='turbo',
                ax=ax)
        ax.set_title(f'{title} at t={metadata['time_array'][i]}')
        plt.draw()
        plt.pause(0.1)

    plt.ioff()    
    plt.show()

