from _static.common.helpers import setup_hardware_client
setup_hardware_client()

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from ipywidgets import interact, IntSlider
IntSlider = partial(IntSlider, continuous_update=False)
plt.style.use("_static/matplotlibrc")

import pynn_brainscales.brainscales2 as pynn
from pynn_brainscales.brainscales2 import Population
from pynn_brainscales.brainscales2.standardmodels.cells import SpikeSourceArray
from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
from pynn_brainscales.brainscales2.connectors import FixedNumberPreConnector
from pyNN.random import NumpyRNG  #  Important for random connectors
from neo.io import PickleIO

from _static.common.helpers import get_nightly_calibration

def plot_emulation_results(block, output_file="simulation_results.png"):
    for segment in block.segments:
        # Plot analog recordings (v)
        for idx, mem_v in enumerate(segment.irregularlysampledsignals):
            plt.figure(figsize=(10, 4))
            plt.plot(mem_v.times, mem_v, label=f"Neuron {idx} Membrane Potential")
            plt.xlabel("Time [ms]")
            plt.ylabel("Membrane Potential [mV]")
            plt.title(f"Neuron {idx} Membrane Potential")
            plt.grid(True)
            plt.legend()

        # Plot spikes
        if segment.spiketrains:
            plt.figure(figsize=(10, 4))
            for i, spiketrain in enumerate(segment.spiketrains):
                plt.plot(spiketrain.times, [i] * len(spiketrain.times), 'r|')
            plt.xlabel("Time [ms]")
            plt.ylabel("Neuron Index")
            plt.title("Spike Raster Plot")
            plt.grid(True)

# Setup
calib = get_nightly_calibration()
pynn.setup(initial_config=calib)

# Neuron Parameters
neuron_params = {
    'leak_v_leak': 400,
    'leak_i_bias': 200,
    'threshold_v_threshold': 100,
    'reset_v_reset': 300,
    'membrane_capacitance_capacitance': 60,
    'refractory_period_refractory_time': 40,
    'threshold_enable': True,
    'reset_i_bias': 1022,
    'reset_enable_multiplication': True,
    'excitatory_input_enable': True,
    'inhibitory_input_enable': True,
    'excitatory_input_i_bias_gm': 1022,
    'inhibitory_input_i_bias_gm': 1022,
    'excitatory_input_i_bias_tau': 200,
    'inhibitory_input_i_bias_tau': 200,
    'excitatory_input_i_shift_reference': 300,
    'inhibitory_input_i_shift_reference': 300
}

# Create Population
n_neurons = 10
pop = pynn.Population(n_neurons, pynn.cells.HXNeuron(**neuron_params))

# Record spikes from all neurons, and 'v' from max 2 neurons (MADC limit)
pop.record("spikes")
pop[0:2].record("v")

# Random Input Generation
np_rng = np.random.default_rng(42)
spike_sources = []
for i in range(n_neurons):
    times = sorted(np_rng.uniform(0.01, 0.2, size=5))
    src = pynn.Population(1, SpikeSourceArray(spike_times=times))
    spike_sources.append(src)
    pynn.Projection(src, pop[i:i+1],
                    pynn.AllToAllConnector(),
                    synapse_type=StaticSynapse(weight=63),
                    receptor_type="excitatory")

# Random Connectivity with PyNN RNG
pynn_rng = NumpyRNG(seed=42)
pynn.Projection(pop, pop,
                FixedNumberPreConnector(n=3, rng=pynn_rng),
                synapse_type=StaticSynapse(weight=40),
                receptor_type="excitatory")

# Run
print("Running experiment on BrainScaleS hardware...")
pynn.run(20.0)  # Run for 20 ms
data = pop.get_data()
plot_emulation_results(data, output_file="emulation_results.png")
pynn.end()
print("BrainScaleS hardware emulation ended.")
