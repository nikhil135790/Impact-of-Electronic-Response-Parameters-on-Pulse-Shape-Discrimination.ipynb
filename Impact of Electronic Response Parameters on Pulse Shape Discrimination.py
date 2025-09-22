#!/usr/bin/env python
# coding: utf-8

# In[4]:


import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc.hdf5"

event_idx = 1
prepad_ns = 45000
postpad_ns = 30000
n_resp = 8000

with h5py.File(filename, "r") as f:
    dt = 16
    values = f["pss/pss/waveform/values/flattened_data"][:]
    length = f["pss/pss/waveform/values/cumulative_length"][:]

    start_idx = length[event_idx - 1] if event_idx > 0 else 0
    end_idx = length[event_idx]
    charge = values[start_idx:end_idx]
    n_prepad = int(np.ceil(prepad_ns / dt))
    n_postpad = int(np.ceil(postpad_ns / dt))
    max_val = np.max(charge)

    charge_padded = np.concatenate([
        np.zeros(n_prepad), charge, np.full(n_postpad, max_val)
    ])
    t_padded = np.arange(len(charge_padded)) * dt
    x = np.gradient(charge_padded, t_padded)  # derivative = current

# --- Single plot with two y-axes ---
fig, ax1 = plt.subplots(figsize=(8,5))

# Plot charge (left axis)
ln1 = ax1.plot(t_padded, charge_padded, color="tab:blue", lw=1.5, label="Charge Signal")
ax1.set_xlim(44000, 48000)
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Charge (e)")
ax1.ticklabel_format(style='sci', axis='y', scilimits=(6,6))
ax1.grid(alpha=0.3)

# Add second axis for current
ax2 = ax1.twinx()
ln2 = ax2.plot(t_padded, x, color="tab:red", lw=1.5, label="Current Signal")
ax2.set_ylabel("Current (e/ns)")

# Combine legends
lns = ln1 + ln2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc="upper left", frameon=False)

plt.title("Charge and Current Waveforms")
plt.tight_layout()
plt.show()


# In[17]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

def asymmetric_triangle_filter(signal, rise_ns, fall_ns, dt):
    rise = int(rise_ns / dt)
    fall = int(fall_ns / dt)
    kernel = np.concatenate([
        np.linspace(0, 1, rise, endpoint=False),
        np.linspace(1, 0, fall, endpoint=True)
    ])
    kernel = kernel / np.sum(kernel)
    return np.convolve(signal, kernel, mode='same')

filename = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc.hdf5"

event_idx = 1
prepad_ns = 45000
postpad_ns = 30000
n_resp = 8000

def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2*L + G, 1), N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] + signal[k - 2*L - G] - signal[k - L - G]) 
    return S / L

def triangle_recursive(signal, L, G):
        S = trap_recursive(signal, L, G)
        N = len(S)
        T = np.zeros(N)
        M = 2*L + G  
        for k in range(M, N):
            T[k] = T[k-1] + S[k] - S[k - M]
        return T / M
    
with h5py.File(filename, "r") as f:
    dt = 16
    values = f["pss/pss/waveform/values/flattened_data"][:]
    length = f["pss/pss/waveform/values/cumulative_length"][:]

    start_idx = length[event_idx - 1] if event_idx > 0 else 0
    end_idx = length[event_idx]
    charge = values[start_idx:end_idx]
    n_prepad = int(np.ceil(prepad_ns / dt))
    n_postpad = int(np.ceil(postpad_ns / dt))
    max_val = np.max(charge)

    charge_padded = np.concatenate([np.zeros(n_prepad), charge, np.full(n_postpad, max_val)])
    t_padded = np.arange(len(charge_padded)) * dt


    t_resp = np.arange(n_resp) * dt
    
    tau = 40
    freq = 1e9/(2 * np.pi * tau)
    h_LP = np.sqrt(2) / tau * np.sin(t_resp / (np.sqrt(2)*tau)) * np.exp(-t_resp / (np.sqrt(2)*tau))
    
    tau1 = 3000000
    h = np.exp(-t_resp / tau1)
    h_total = np.convolve(h_LP, h)[:n_resp] * dt
    h_total = h_total / np.sum(h_total * dt)

    x = np.gradient(charge_padded, t_padded)
    
    y_shaped = np.convolve(x, h_total, mode='full')[:len(x)]
    
    amplitude = 0.0002
    noise = np.random.normal(0, amplitude, size=len(y_shaped))

    y_noisy = y_shaped + noise
    
    L = int(4000 / dt)
    G = int(2500 / dt)
    
    L_tri = 1
    G_tri = 0
    
    y_trap = trap_recursive(y_noisy, L, G)
    y_tri = triangle_recursive(y_noisy, L_tri, G_tri)
    z = np.gradient(y_noisy, dt) * dt
    A = np.max(y_tri)
    E = np.max(y_trap)


plt.figure()
plt.plot(t_padded, y_noisy, label = 'CSA Output')
plt.xlim(45000, 55000)
plt.plot(t_padded, y_trap, label = 'Trap Filter (A) Output')
plt.plot(t_padded, y_tri, label = 'Triangle Filter (E) Output')
plt.xlabel("Time (ns)")
plt.ylabel("CSA Waveforms (a.u)")
plt.title("CSA and Filter Outputs")
plt.legend()
plt.grid(True)
plt.show()


# In[18]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy.signal import fftconvolve

prepad_ns = 45000
postpad_ns = 20000
n_resp = 8000
dt = 16
t_resp = np.arange(n_resp) * dt
tau_rise = 47
tau1 = 3_000_000
h_LP = np.sin(t_resp / (np.sqrt(2)*tau_rise)) * np.exp(-t_resp / (np.sqrt(2)*tau_rise))
h_HP = np.exp(-t_resp / tau1)
h_total = np.convolve(h_LP, h_HP)[:n_resp] * dt
h_total = h_total / np.sum(h_total * dt)
L = int(4000 / dt)
G = int(2500 / dt)
L_tri = 1
G_tri = 0
n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))
noise_amplitude = 0.0002

@njit
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2*L + G, 1), N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] + signal[k - 2*L - G] - signal[k - L - G])
    return S / L

@njit
def triangle_recursive(signal, L, G):
    S = trap_recursive(signal, L, G)
    N = len(S)
    T = np.zeros(N)
    M = 2*L + G
    for k in range(M, N):
        T[k] = T[k-1] + S[k] - S[k - M]
    return T / M

def process_file(filename):
    with h5py.File(filename, "r") as f:
        values = f["pss/pss/waveform/values/flattened_data"][:]
        length = f["pss/pss/waveform/values/cumulative_length"][:]
        num_events = len(length)
        event_lens = []
        for i in range(num_events):
            start_idx = length[i-1] if i > 0 else 0
            end_idx = length[i]
            charge = values[start_idx:end_idx]
            event_lens.append(n_prepad + len(charge) + n_postpad)
        max_padded_len = max(event_lens)
        signals = np.zeros((num_events, max_padded_len))
        for i in range(num_events):
            start_idx = length[i-1] if i > 0 else 0
            end_idx = length[i]
            charge = values[start_idx:end_idx]
            max_val = np.max(charge)
            charge_padded = np.concatenate([np.zeros(n_prepad), charge, np.full(n_postpad, max_val)])
            signals[i, :len(charge_padded)] = charge_padded
        x = np.gradient(signals, dt, axis=1)
        def event_convolve(sig):
            return fftconvolve(sig, h_total, mode='full')[:sig.shape[0]]
        y_shaped = np.stack(Parallel(n_jobs=-1, prefer="threads")(delayed(event_convolve)(sig) for sig in x))
        np.random.seed(42)
        noise = np.random.normal(0, noise_amplitude, size=y_shaped.shape)
        y_noisy = y_shaped + noise
        A_over_E_values = []
        for i in range(num_events):
            y_trap = trap_recursive(y_noisy[i], L, G)
            y_tri = triangle_recursive(y_noisy[i], L_tri, G_tri)
            A = np.max(y_tri)
            E = np.max(y_trap)
            if E > 0:
                A_over_E_values.append(A / E)
        return np.array(A_over_E_values)

file_template = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_{}_sc.hdf5"
filepaths = [
    r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc.hdf5"
]
for i in range(2, 11):
    filepaths.append(file_template.format(i))

AE_values = []
for fp in filepaths:
    vals = process_file(fp)
    AE_values.append(vals)
AE_values = np.concatenate(AE_values)
AE_values = AE_values[~np.isnan(AE_values)]

plt.figure()
plt.hist(AE_values[(AE_values >= 0.02) & (AE_values <= 0.15)],bins=500, color='blue', alpha=0.75, edgecolor='red')
plt.xlabel('A/E')
plt.ylabel('Counts')
plt.title('A/E Distribution')
plt.grid(True)
plt.show()


# In[20]:


import h5py
import numpy as np
import matplotlib.pyplot as plt

# --- File paths ---
file_template = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_{}_sc.hdf5"
filepaths = [r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc (1).hdf5"]
for i in range(2, 11):
    filepaths.append(file_template.format(i))

# --- Energy peaks of interest (keV) ---
FEP = 2614.5
SEP = FEP - 511
DEP = FEP - 2 * 511
peak_centers = {"DEP": DEP, "SEP": SEP, "FEP": FEP}

# --- Collect all energies from all files ---
all_event_energies = []
for fp in filepaths:
    with h5py.File(fp, "r") as f:
        edep_flat = f["pss/truth/edep/flattened_data"][:]
        edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
        start = 0
        for end in edep_cumlen:
            e = np.sum(edep_flat[start:end])
            all_event_energies.append(e)
            start = end
all_event_energies = np.array(all_event_energies)

plt.figure(figsize=(9,6), dpi = 200)
plt.hist(all_event_energies, bins=250, range=(0, 4000),
         histtype="stepfilled", color="navy", alpha=0.8)

plt.xlabel("Event Energy (keV)")
plt.ylabel("Counts")
plt.title("Event Energy Spectrum")
plt.xlim(450, 3500)

# Softer annotations for DEP, SEP, FEP
for label, center in peak_centers.items():
    plt.axvline(center, color="darkgreen", linestyle="--", alpha=0.5, linewidth=1)
    plt.annotate(label,
                 xy=(center, plt.ylim()[1]*0.85),   # anchor point
                 xytext=(0, 30),                   # offset in pixels
                 textcoords="offset points",
                 ha="center", va="bottom",
                 fontsize=15, color="red",
                 arrowprops=dict(arrowstyle="-", color="darkgreen", lw=0.8, alpha=0.5))

plt.tight_layout()
plt.show()


# In[21]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.signal import fftconvolve
import os
from numba import njit

# -------------------------
# File setup
# -------------------------
file_template = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_{}_sc.hdf5"
filepaths = []
for i in range(1, 11):
    fname = file_template.format(i)
    if os.path.exists(fname):
        filepaths.append(fname)
    else:
        print(f"File missing: {fname} (skipped)")

# -------------------------
# Peak definitions
# -------------------------
FEP = 2614.5
SEP = FEP - 511
DEP = FEP - 2 * 511
peak_centers = [("DEP", DEP), ("SEP", SEP), ("FEP", FEP)]
window = 5.0

# -------------------------
# Signal shaping parameters
# -------------------------
prepad_ns = 45000
postpad_ns = 30000    # NOTE: matches script #2’s original
n_resp = 8000
dt = 16
tau = 47
tau1 = 3_000_000
amplitude = 0.0002

# Trap/triangle parameters
L = int(4000 / dt)
G = int(2500 / dt)
L_tri = 1
G_tri = 0

# -------------------------
# Filters
# -------------------------
t_resp = np.arange(n_resp) * dt
h_LP = np.sqrt(2) / tau * np.sin(t_resp / (np.sqrt(2) * tau)) * np.exp(-t_resp / (np.sqrt(2) * tau))
h = np.exp(-t_resp / tau1)
h_total = np.convolve(h_LP, h)[:n_resp] * dt
h_total = h_total / np.sum(h_total * dt)

n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))

# -------------------------
# Recursive filters
# -------------------------
@njit
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2 * L + G, 1), N):
        S[k] = S[k - 1] + (signal[k] - signal[k - L] + signal[k - 2 * L - G] - signal[k - L - G])
    return S / L

@njit
def triangle_recursive(signal, L, G):
    S = trap_recursive(signal, L, G)
    N = len(S)
    T = np.zeros(N)
    M = 2 * L + G
    for k in range(M, N):
        T[k] = T[k - 1] + S[k] - S[k - M]
    return T / M

# -------------------------
# Load all events with corrected indexing
# -------------------------
all_event_energies = []
all_event_waveforms_list = []
all_event_starts = []
all_event_ends = []

global_offset = 0

for filename in filepaths:
    print(f"Loading {filename} ...")
    with h5py.File(filename, "r") as f:
        edep_flat = f["pss/truth/edep/flattened_data"][:]
        edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
        values = f["pss/pss/waveform/values/flattened_data"][:]
        length = f["pss/pss/waveform/values/cumulative_length"][:]

        # event energies
        event_energies = []
        start = 0
        for end in edep_cumlen:
            event_energies.append(np.sum(edep_flat[start:end]))
            start = end
        all_event_energies.append(np.array(event_energies, dtype=np.float32))

        # corrected indexing
        starts = np.empty_like(length)
        starts[0] = 0
        starts[1:] = length[:-1]
        starts = starts + global_offset
        ends = length + global_offset

        all_event_starts.append(starts)
        all_event_ends.append(ends)
        all_event_waveforms_list.append(values)

        global_offset += values.shape[0]

# build global arrays
all_event_waveforms = np.concatenate(all_event_waveforms_list)
all_event_starts = np.concatenate(all_event_starts)
all_event_ends = np.concatenate(all_event_ends)
all_event_energies = np.concatenate(all_event_energies)

n_events = len(all_event_energies)


labels = ['DEP', 'SEP', 'FEP']
colors = {'DEP': 'royalblue', 'SEP': 'orange', 'FEP': 'crimson'}

relevant_mask = np.zeros_like(all_event_energies, dtype=bool)
for label, center in peak_centers:
    relevant_mask |= (np.abs(all_event_energies - center) <= window)
relevant_indices = np.where(relevant_mask)[0]
print(f"Processing {len(relevant_indices)} out of {n_events} events (pre-filtered by energy).")

def process_event(event_idx, starts, ends, values, n_prepad, n_postpad, dt, h_total, amplitude, L, G, L_tri, G_tri):
    start_idx = int(starts[event_idx])
    end_idx = int(ends[event_idx])
    if end_idx <= start_idx:
        return np.nan, np.nan
    charge = values[start_idx:end_idx]
    if charge.size == 0:
        return np.nan, np.nan

    max_val = np.max(charge)
    charge_padded = np.concatenate([np.zeros(n_prepad), charge, np.full(n_postpad, max_val)])
    x = np.gradient(charge_padded, dt)
    y_shaped = fftconvolve(x, h_total, mode='full')[:len(x)]
    y_noisy = y_shaped + np.random.normal(0, amplitude, size=len(y_shaped))

    y_trap = trap_recursive(y_noisy, L, G)
    y_tri = triangle_recursive(y_noisy, L_tri, G_tri)

    A = np.max(y_tri)
    E = np.max(y_trap)
    if E <= 0:
        return np.nan, np.nan
    return A, E

# run
results = Parallel(n_jobs=-1, prefer='threads')(
    delayed(process_event)(idx, all_event_starts, all_event_ends, all_event_waveforms,
                           n_prepad, n_postpad, dt, h_total, amplitude,
                           L, G, L_tri, G_tri)
    for idx in relevant_indices
)

A_all, E_all = zip(*results)
A_all = np.array(A_all)
E_all = np.array(E_all)
valid_mask = ~np.isnan(A_all) & ~np.isnan(E_all) & (E_all > 0)
A_all = A_all[valid_mask]
E_all = E_all[valid_mask]
selected_energies = all_event_energies[relevant_indices][valid_mask]

AE_all = A_all / E_all
valid_AE_mask = AE_all <= 0.15
AE_all = AE_all[valid_AE_mask]
selected_energies = selected_energies[valid_AE_mask]

AE_dict = {}
for label, center in peak_centers:
    mask = np.abs(selected_energies - center) <= window
    AE_dict[label] = AE_all[mask]


plt.figure(figsize=(9, 6))
if len(AE_all) > 0:
    bins = np.linspace(np.min(AE_all), np.max(AE_all), 200)
else:
    bins = 45
for label in labels:
    z = 0 if label == "FEP" else 1   
    plt.hist(
        AE_dict[label],
        bins=bins,
        alpha=0.7,
        label=label,
        color=colors[label],
        edgecolor='k',
        linewidth=1.0,
        density=True,
        zorder=z
    )
plt.xlabel("A/E")
plt.ylabel("Normalized Counts")
plt.title("A/E Distributions for DEP, SEP, FEP")
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()


# In[26]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.signal import fftconvolve
from joblib import Parallel, delayed
import time

prepad_ns = 45000
postpad_ns = 20000
n_resp = 8000
dt = 16
t_resp = np.arange(n_resp) * dt
tau_rise = 50
tau1 = 3_000_000
h_LP = np.sin(t_resp / (np.sqrt(2)*tau_rise)) * np.exp(-t_resp / (np.sqrt(2)*tau_rise))
h_HP = np.exp(-t_resp / tau1)
h_total = np.convolve(h_LP, h_HP)[:n_resp] * dt
h_total = h_total / np.sum(h_total * dt)
L = int(4000 / dt)
G = int(2500 / dt)
L_tri = 1
G_tri = 0
n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))
noise_amplitude = 0

@njit
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2*L + G, 1), N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] + signal[k - 2*L - G] - signal[k - L - G])
    return S / L

@njit
def triangle_recursive(signal, L, G):
    S = trap_recursive(signal, L, G)
    N = len(S)
    T = np.zeros(N)
    M = 2*L + G
    for k in range(M, N):
        T[k] = T[k-1] + S[k] - S[k - M]
    return T / M

def process_file(filename, stride=1):
    with h5py.File(filename, "r") as f:
        values = f["pss/pss/waveform/values/flattened_data"][:]
        length = f["pss/pss/waveform/values/cumulative_length"][:]
        edep_flat = f["pss/truth/edep/flattened_data"][:]
        edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
        num_events = len(length)
        event_indices = range(0, num_events, stride)

        A_over_E_values = []
        event_energies = []

        for i in event_indices:
            edep_start = edep_cumlen[i-1] if i > 0 else 0
            edep_end = edep_cumlen[i]
            energy = np.sum(edep_flat[edep_start:edep_end])
            start_idx = length[i-1] if i > 0 else 0
            end_idx = length[i]
            charge = values[start_idx:end_idx]
            max_val = np.max(charge)
            charge_padded = np.concatenate([np.zeros(n_prepad), charge, np.full(n_postpad, max_val)])
            x = np.gradient(charge_padded, dt)
            y_shaped = fftconvolve(x, h_total, mode='full')[:len(x)]
            y_noisy = y_shaped
            y_trap = trap_recursive(y_noisy, L, G)
            y_tri = triangle_recursive(y_noisy, L_tri, G_tri)
            A = np.max(y_tri)
            E = np.max(y_trap)
            if E > 0:
                A_over_E_values.append(A / E)
                event_energies.append(energy)
        return np.array(A_over_E_values), np.array(event_energies)

file_template = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_{}_sc.hdf5"
filepaths = [r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc.hdf5"]
for i in range(2, 11):
    filepaths.append(file_template.format(i))



results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
    delayed(process_file)(fp, stride=1) for fp in filepaths
)
AE_values = [res[0] for res in results]
energies = [res[1] for res in results]
AE_values = np.concatenate(AE_values)
energies = np.concatenate(energies)




valid = (~np.isnan(AE_values)) & (AE_values > 0) & (AE_values < 0.15)
A_over_E_clean = AE_values[valid]
energies = energies[valid]
energy_threshold = 1500
valid_energy = energies > energy_threshold
A_over_E_clean = A_over_E_clean[valid_energy]
energies = energies[valid_energy]

dep_signal_win = (1591, 1594)
sep_win = (2102, 2105)
fep_win = (2611, 2619)

dep_mask = (energies >= dep_signal_win[0]) & (energies <= dep_signal_win[1])
sep_mask = (energies >= sep_win[0]) & (energies <= sep_win[1])
fep_mask = (energies >= fep_win[0]) & (energies <= fep_win[1])

A_over_E_dep = A_over_E_clean[dep_mask]
A_over_E_sep = A_over_E_clean[sep_mask]
A_over_E_fep = A_over_E_clean[fep_mask]
A_over_E_bkg = np.concatenate([A_over_E_sep, A_over_E_fep])

cut_values = np.linspace(np.min(A_over_E_dep), np.max(A_over_E_dep), 300)
fom_list = []
signal_acc_list = []
bkg_acc_list = []

signal_threshold = 0.84

for cut in cut_values:
    sig_acc = np.mean(A_over_E_dep > cut)
    bkg_acc = np.mean(A_over_E_bkg > cut)
    if sig_acc < signal_threshold:
        fom = 0
    else:
        fom = sig_acc / np.sqrt(bkg_acc) if bkg_acc > 0 else 0
    fom_list.append(fom)
    signal_acc_list.append(sig_acc)
    bkg_acc_list.append(bkg_acc)

fom_array = np.array(fom_list)
best_idx = np.argmax(fom_array)
optimal_cut = cut_values[best_idx]
optimal_fom = fom_array[best_idx]
optimal_sig = signal_acc_list[best_idx]
optimal_bkg = bkg_acc_list[best_idx]


plt.figure()
plt.plot(cut_values, fom_array, label='FoM')
plt.axvline(optimal_cut, color='red', linestyle='--', label=f'Optimal cut ({optimal_cut:.5f})')
plt.xlabel("A/E Cut")
plt.ylabel("FoM")
plt.title("Optimization of A/E Cut")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# In[30]:


import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc.hdf5"

event_idx = 1
prepad_ns = 45000
postpad_ns = 30000
n_resp = 8000

with h5py.File(filename, "r") as f:
    dt = 16
    t0 = 0
    values = f["pss/pss/waveform/values/flattened_data"][:]
    length = f["pss/pss/waveform/values/cumulative_length"][:]

    start_idx = length[event_idx - 1] if event_idx > 0 else 0
    end_idx = length[event_idx]
    charge = values[start_idx:end_idx]
    n_prepad = int(np.ceil(prepad_ns / dt))
    n_postpad = int(np.ceil(postpad_ns / dt))
    max_val = np.max(charge)

    charge_padded = np.concatenate([np.zeros(n_prepad), charge, np.full(n_postpad, max_val)])
    t_padded = np.arange(len(charge_padded)) * dt

    t_resp = np.arange(n_resp) * dt

    amplitude = 0.00002
    x = np.gradient(charge_padded, t_padded)
    base_noise = np.random.normal(0, amplitude, size=len(x))
    
    plt.figure()

    freq_list_MHz = [3, 4, 5, 6, 7, 8]
    colors = plt.cm.viridis(np.linspace(0, 1, len(freq_list_MHz)))

    for i, freq_MHz in enumerate(freq_list_MHz):
        freq_Hz = freq_MHz * 1e6
        tau = 1e9 / (2 * np.pi * freq_Hz)
        tau1 = 3000000
        h_LP = np.sqrt(2) / tau * np.sin(t_resp / (np.sqrt(2) * tau)) * np.exp(-t_resp / (np.sqrt(2) * tau))
        h = np.exp(-t_resp / tau1)
        h_total = np.convolve(h_LP, h)[:n_resp] * dt
        h_total /= np.sum(h_total * dt)


        y_shaped = np.convolve(x, h_total, mode='full')[:len(x)]
        y_noisy = y_shaped + base_noise

        plt.plot(t_padded, y_noisy, label=f'{freq_MHz:.1f} MHz', color=colors[i])
        
    plt.xlim(46200, 46700)
    plt.ylim(0.28, 0.3)

    
    plt.xlabel("Time (ns)")
    plt.ylabel("CSA Waveform (a.u.)")
    plt.title("CSA Output for Different LP Filter Cutoff Frequencies")
    plt.legend(title="Cutoff Freq.", loc="center left", bbox_to_anchor=(1, 0.5))

    plt.yticks([0.280, 0.285, 0.290, 0.295, 0.300]) 
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# In[38]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import njit
from scipy.signal import fftconvolve

filename = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc (1).hdf5"

FEP = 2614.5
SEP = FEP - 511
DEP = FEP - 2*511
peak_dict = {"DEP": DEP, "SEP": SEP, "FEP": FEP}
window = 5  # keV

prepad_ns = 45000
postpad_ns = 30000
n_resp = 8000
dt = 16

@njit
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2*L + G, 1), N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] + signal[k - 2*L - G] - signal[k - L - G]) 
    return S / L

filtered_charges = {key: [] for key in peak_dict}
with h5py.File(filename, "r") as f:
    edep_flat = f["pss/truth/edep/flattened_data"][:]
    edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
    values = f["pss/pss/waveform/values/flattened_data"][:]
    length = f["pss/pss/waveform/values/cumulative_length"][:]
    start = 0
    for i, end in enumerate(edep_cumlen):
        energy = np.sum(edep_flat[start:end])
        for k, center in peak_dict.items():
            if abs(energy - center) <= window:
                s_idx = length[i-1] if i > 0 else 0
                e_idx = length[i]
                filtered_charges[k].append(values[s_idx:e_idx])
                break
        start = end

n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))
L = int(4000 / dt)
G = int(2500 / dt)

freq_list_MHz = np.linspace(1, 10, 21)
tau_list_ns = 1e9 / (2 * np.pi * freq_list_MHz * 1e6) 

tau1 = 3_000_000 

results_tau = {key: [] for key in peak_dict}
results_fc = {key: [] for key in peak_dict}

def overshoot_worker(charge, h_total, n_prepad, n_postpad, dt, L, G):
    max_val = np.max(charge)
    charge_padded = np.concatenate([np.zeros(n_prepad), charge, np.full(n_postpad, max_val)])
    t_padded = np.arange(len(charge_padded)) * dt
    x = np.gradient(charge_padded, t_padded)
    y_shaped = fftconvolve(x, h_total, mode='full')[:len(x)]
    y_noisy = y_shaped
    y_trap = trap_recursive(y_noisy, L, G)
    max_csa = np.max(y_noisy)
    max_trap = np.max(y_trap)
    overshoot = 100 * (max_csa - max_trap) / max_trap if max_trap != 0 else np.nan
    return overshoot

for i, (freq_MHz, tau) in enumerate(zip(freq_list_MHz, tau_list_ns)):
    print(f"Processing {freq_MHz:.2f} MHz ({i+1}/{len(freq_list_MHz)})")
    t_resp = np.arange(n_resp) * dt
    h_LP = np.sqrt(2) / tau * np.sin(t_resp / (np.sqrt(2) * tau)) * np.exp(-t_resp / (np.sqrt(2) * tau))
    h = np.exp(-t_resp / tau1)
    h_total = np.convolve(h_LP, h)[:n_resp] * dt
    h_total /= np.sum(h_total * dt)

    for k in peak_dict.keys():
        charges = filtered_charges[k]
        if not charges:  # No events for this peak
            results_tau[k].append(np.nan)
            results_fc[k].append(np.nan)
            continue
        overshoots = Parallel(n_jobs=-1, prefer='threads')(
            delayed(overshoot_worker)(charge, h_total, n_prepad, n_postpad, dt, L, G)
            for charge in charges
        )
        arr = np.array(overshoots)
        results_tau[k].append(np.nanmedian(arr))
        results_fc[k].append(np.nanmedian(arr))

plt.figure(figsize=(7,5))
for k in peak_dict.keys():
    plt.plot(freq_list_MHz, results_fc[k], marker='o', label=k)
plt.xlabel("LP Filter Cutoff Frequency (MHz)")
plt.ylabel("Median Overshoot (%)")
plt.title("CSA Overshoot vs LP Cutoff Frequency")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# In[39]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from numba import njit
from scipy.signal import fftconvolve
import os
import time
import matplotlib.ticker as ticker

# --- File paths ---
file_template = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_{}_sc.hdf5"
filepaths = [r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc (1).hdf5"]
for i in range(2, 3):
    fname = file_template.format(i)
    if os.path.exists(fname):
        filepaths.append(fname)
    else:
        print(f"File missing: {fname} (skipped)")

# --- Energy peaks of interest (keV) ---
FEP = 2614.5
SEP = FEP - 511
DEP = FEP - 2 * 511
peak_centers = [("DEP", DEP), ("SEP", SEP), ("FEP", FEP)]
window = 7.0 

prepad_ns = 45000
postpad_ns = 30000
n_resp = 8000
dt = 16  
tau1 = 3000000
amplitude = 0.0002

# Trap/triangle parameters
L = int(4000 / dt)
G = int(2500 / dt)
L_tri = 1
G_tri = 0

cutoff_freqs_MHz = np.arange(1.0, 10.1, 0.1)

@njit(cache=True, fastmath=True)
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    min_k = max(2*L + G, 1)
    for k in range(min_k, N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] + signal[k - 2*L - G] - signal[k - L - G])
    return S / L

@njit(cache=True, fastmath=True)
def triangle_recursive(signal, L, G):
    S = trap_recursive(signal, L, G)
    N = len(S)
    T = np.zeros(N)
    M = 2*L + G  
    for k in range(M, N):
        T[k] = T[k-1] + S[k] - S[k-M]
    return T / M

def process_event(event_idx, length, values, n_prepad, n_postpad, dt, h_total, amplitude, L, G, L_tri, G_tri):
    start_idx = length[event_idx - 1] if event_idx > 0 else 0
    end_idx = length[event_idx]
    if end_idx <= start_idx:
        return np.nan, np.nan
    charge = values[start_idx:end_idx]
    if charge.size == 0:
        return np.nan, np.nan
    max_val = np.max(charge)
    charge_padded = np.concatenate([np.zeros(n_prepad), charge, np.full(n_postpad, max_val)])
    t_padded = np.arange(len(charge_padded)) * dt
    x = np.gradient(charge_padded, t_padded)
    y_shaped = fftconvolve(x, h_total, mode='full')[:len(x)]
    noise = np.random.normal(0, amplitude, size=len(y_shaped))
    y_noisy = y_shaped + noise
    y_trap = trap_recursive(y_noisy, L, G)
    y_tri = triangle_recursive(y_noisy, L_tri, G_tri)
    A = np.max(y_tri)
    E = np.max(y_trap)
    if E <= 0:
        return np.nan, np.nan 
    return A, E

# --- Load all event energies/waveforms ---
all_event_energies = []
all_event_waveforms = []
all_event_lengths = []
for filename in filepaths:
    print(f"Loading {filename} ...")
    with h5py.File(filename, "r") as f:
        edep_flat = f["pss/truth/edep/flattened_data"][:]
        edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
        values = f["pss/pss/waveform/values/flattened_data"][:]
        length = f["pss/pss/waveform/values/cumulative_length"][:]
        event_energies = []
        start = 0
        for end in edep_cumlen:
            e = np.sum(edep_flat[start:end])
            event_energies.append(e)
            start = end
        all_event_energies.append(np.array(event_energies, dtype=np.float32))
        all_event_waveforms.append(values)
        all_event_lengths.append(length)

all_event_energies = np.concatenate(all_event_energies)
all_event_waveforms = np.concatenate(all_event_waveforms)
all_event_lengths = np.concatenate(all_event_lengths)

n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))
n_events = len(all_event_energies)

labels = ['DEP', 'SEP', 'FEP']
colors = {'DEP': 'royalblue', 'SEP': 'orange', 'FEP': 'crimson'}

relevant_mask = np.zeros_like(all_event_energies, dtype=bool)
for label, center in peak_centers:
    relevant_mask |= (np.abs(all_event_energies - center) <= window)
relevant_indices = np.where(relevant_mask)[0]
print(f"Processing {len(relevant_indices)} out of {n_events} events (pre-filtered by energy).")

# Only keep medians
stats_median = {label: [] for label in labels}

start = time.time()

for fc_MHz in cutoff_freqs_MHz:
    tau = 1 / (2 * np.pi * fc_MHz * 1e6)  # tau in seconds
    t_resp = np.arange(n_resp) * dt * 1e-9 
    h_LP = np.sqrt(2) / tau * np.sin(t_resp / (np.sqrt(2) * tau)) * np.exp(-t_resp / (np.sqrt(2) * tau))
    h = np.exp(-t_resp / tau1)
    h_total = np.convolve(h_LP, h)[:n_resp] * dt
    sum_kernel = np.sum(h_total * dt)
    if np.abs(sum_kernel) < 1e-12:
        print(f"Warning: kernel sum ~0 for cutoff {fc_MHz:.2f} MHz, skipping.")
        for label in labels:
            stats_median[label].append(np.nan)
        continue
    h_total = h_total / sum_kernel
    
    results = Parallel(n_jobs=-1, prefer='processes')(delayed(process_event)(
        idx, all_event_lengths, all_event_waveforms, n_prepad, n_postpad, dt,
        h_total, amplitude, L, G, L_tri, G_tri) for idx in relevant_indices)
    A_all, E_all = zip(*results)
    A_all = np.array(A_all)
    E_all = np.array(E_all)
    valid_mask = ~np.isnan(A_all) & ~np.isnan(E_all) & (E_all > 0)
    A_all = A_all[valid_mask]
    E_all = E_all[valid_mask]
    selected_energies = all_event_energies[relevant_indices][valid_mask]
    AE_all = A_all / E_all

    valid_AE_mask = AE_all <= 0.2
    AE_all = AE_all[valid_AE_mask]
    selected_energies = selected_energies[valid_AE_mask]

    for label, center in peak_centers:
        mask = np.abs(selected_energies - center) <= window
        arr = AE_all[mask]
        if len(arr) > 0:
            stats_median[label].append(np.median(arr))
        else:
            stats_median[label].append(np.nan)

print(f"\nTotal processing time: {time.time() - start:.2f} seconds")

plt.figure(figsize=(7,5))
for label in labels:
    plt.plot(cutoff_freqs_MHz, stats_median[label], marker='o', ms=3, 
             label=label, color=colors[label], alpha=0.75)
plt.xlabel('LP Filter Cutoff Frequency (MHz)')
plt.ylabel('A/E Median')
plt.title('A/E Median vs. Cutoff Frequency')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
ax = plt.gca()
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.show()


# In[ ]:


#LP Filter

import h5py
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy.signal import fftconvolve

prepad_ns = 45000
postpad_ns = 20000
n_resp = 8000
dt = 16
tau1 = 3_000_000
L = int(4000 / dt)
G = int(2500 / dt)
L_tri = 1
G_tri = 0
n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))
noise_amplitude = 0.0005

@njit
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2*L + G, 1), N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] + signal[k - 2*L - G] - signal[k - L - G])
    return S / L

@njit
def triangle_recursive(signal, L, G):
    S = trap_recursive(signal, L, G)
    N = len(S)
    T = np.zeros(N)
    M = 2*L + G
    for k in range(M, N):
        T[k] = T[k-1] + S[k] - S[k - M]
    return T / M

def process_file(filename, h_total, stride=1):
    with h5py.File(filename, "r") as f:
        values = f["pss/pss/waveform/values/flattened_data"][:]
        length = f["pss/pss/waveform/values/cumulative_length"][:]
        edep_flat = f["pss/truth/edep/flattened_data"][:]
        edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
        num_events = len(length)
        event_indices = range(0, num_events, stride)
        dep_win = (1589, 1595)
        sep_win = (2100, 2107)
        fep_win = (2609, 2621)
        A_over_E_values = []
        event_energies = []
        for i in event_indices:
            edep_start = edep_cumlen[i-1] if i > 0 else 0
            edep_end = edep_cumlen[i]
            energy = np.sum(edep_flat[edep_start:edep_end])
            if not ((dep_win[0] <= energy <= dep_win[1]) or 
                    (sep_win[0] <= energy <= sep_win[1]) or 
                    (fep_win[0] <= energy <= fep_win[1])):
                continue  # skip event if not in a peak window
            start_idx = length[i-1] if i > 0 else 0
            end_idx = length[i]
            charge = values[start_idx:end_idx]
            max_val = np.max(charge)
            charge_padded = np.concatenate([np.zeros(n_prepad), charge, np.full(n_postpad, max_val)])
            x = np.gradient(charge_padded, dt)
            y_shaped = fftconvolve(x, h_total, mode='full')[:len(x)]
            y_noisy = y_shaped
            y_trap = trap_recursive(y_noisy, L, G)
            y_tri = triangle_recursive(y_noisy, L_tri, G_tri)
            A = np.max(y_tri)
            E = np.max(y_trap)
            if E > 0:
                A_over_E_values.append(A / E)
                event_energies.append(energy)
        return np.array(A_over_E_values), np.array(event_energies)

def max_fom_vs_tau_rise(tau_rise, filepaths):
    t_resp = np.arange(n_resp) * dt
    h_LP = np.sin(t_resp / (np.sqrt(2)*tau_rise)) * np.exp(-t_resp / (np.sqrt(2)*tau_rise))
    h_HP = np.exp(-t_resp / tau1)
    h_total = np.convolve(h_LP, h_HP)[:n_resp] * dt
    h_total = h_total / np.sum(h_total * dt)
    AE_values = []
    energies = []
    for fp in filepaths:
        vals, event_energies = process_file(fp, h_total, stride=1)
        AE_values.append(vals)
        energies.append(event_energies)
    AE_values = np.concatenate(AE_values)
    energies = np.concatenate(energies)
    valid = (~np.isnan(AE_values)) & (AE_values > 0) & (AE_values < 0.12)
    A_over_E_clean = AE_values[valid]
    energies = energies[valid]
    energy_threshold = 1500
    valid_energy = energies > energy_threshold
    A_over_E_clean = A_over_E_clean[valid_energy]
    energies = energies[valid_energy]
    dep_mask = (energies >= 1589) & (energies <= 1595)
    sep_mask = (energies >= 2100) & (energies <= 2107)
    fep_mask = (energies >= 2609) & (energies <= 2621)
    A_over_E_fep = A_over_E_clean[fep_mask]
    A_over_E_dep = A_over_E_clean[dep_mask]
    A_over_E_sep = A_over_E_clean[sep_mask]
    A_over_E_bkg = np.concatenate([A_over_E_sep, A_over_E_fep])
    cut_values = np.linspace(np.min(A_over_E_dep), np.max(A_over_E_dep), 300)
    fom_list = []
    sig_acc_list = []
    bkg_acc_list = []
    signal_threshold = 0.85
    for cut in cut_values:
        sig_acc = np.mean(A_over_E_dep > cut)
        bkg_acc = np.mean(A_over_E_bkg > cut)
        if sig_acc < signal_threshold:
            fom = 0
        else:
            fom = sig_acc / np.sqrt(bkg_acc) if bkg_acc > 0 else 0
        fom_list.append(fom)
        sig_acc_list.append(sig_acc)
        bkg_acc_list.append(bkg_acc)
    fom_array = np.array(fom_list)
    sig_acc_array = np.array(sig_acc_list)
    best_idx = np.argmax(fom_array)
    best_sig_idx = np.argmax(sig_acc_array)
    optimal_cut = cut_values[best_idx]
    optimal_fom = fom_array[best_idx]
    max_sig_acc = sig_acc_array[best_sig_idx]
    cut_at_max_sig_acc = cut_values[best_sig_idx]
    return optimal_fom, optimal_cut, max_sig_acc, cut_at_max_sig_acc, sig_acc_list[best_idx], bkg_acc_list[best_idx]

file_template = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_{}_sc.hdf5"
filepaths = [r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc (1).hdf5"]
for i in range(2, 2):
    filepaths.append(file_template.format(i))

lp_cutoff_freqs_MHz =  np.arange(1.0, 10.1, 0.1)  # 81 points from 1 MHz to 10 MHz
lp_cutoff_freqs_Hz = lp_cutoff_freqs_MHz * 1e6

# Convert to tau_rise in nanoseconds: tau = 1 / (2πf)
tau_rise_values = (1 / (2 * np.pi * lp_cutoff_freqs_Hz)) * 1e9  # in ns

# Run optimization over these tau_rise values
results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
    delayed(max_fom_vs_tau_rise)(tau_rise, filepaths) for tau_rise in tau_rise_values
)

# Unpack results
fom_values, cut_values, max_sig_acc_values, cut_at_max_sig_acc_values, sig_accs_at_best, bkg_accs_at_best = zip(*results)

# Plot FoM vs LP cutoff frequency
plt.figure()
plt.plot(lp_cutoff_freqs_MHz, fom_values, marker='o', ms=3)
plt.xlabel("LP Filter Cutoff Frequency (MHz)")
plt.ylabel("Maximum FoM")
plt.title("Maximum FoM vs LP Filter Cutoff Frequency")
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy.signal import fftconvolve

prepad_ns = 45000
postpad_ns = 20000
n_resp = 8000
dt = 16
tau1 = 3_000_000
L = int(4000 / dt)
G = int(2500 / dt)
G_tri = 0
n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))

@njit
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2*L + G, 1), N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] + signal[k - 2*L - G] - signal[k - L - G])
    return S / L

@njit
def triangle_recursive(signal, L, G):
    S = trap_recursive(signal, L, G)
    N = len(S)
    T = np.zeros(N)
    M = 2*L + G
    for k in range(M, N):
        T[k] = T[k-1] + S[k] - S[k - M]
    return T / M

def process_file(filename, h_total, L_tri, noise_amplitude, stride=1):
    with h5py.File(filename, "r") as f:
        values = f["pss/pss/waveform/values/flattened_data"][:]
        length = f["pss/pss/waveform/values/cumulative_length"][:]
        edep_flat = f["pss/truth/edep/flattened_data"][:]
        edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
        num_events = len(length)
        event_indices = range(0, num_events, stride)
        dep_win = (1589, 1595)
        sep_win = (2100, 2107)
        fep_win = (2609, 2621)
        A_over_E_values = []
        event_energies = []
        for i in event_indices:
            edep_start = edep_cumlen[i-1] if i > 0 else 0
            edep_end = edep_cumlen[i]
            energy = np.sum(edep_flat[edep_start:edep_end])
            if not ((dep_win[0] <= energy <= dep_win[1]) or 
                    (sep_win[0] <= energy <= sep_win[1]) or 
                    (fep_win[0] <= energy <= fep_win[1])):
                continue  # skip event if not in a peak window
            start_idx = length[i-1] if i > 0 else 0
            end_idx = length[i]
            charge = values[start_idx:end_idx]
            max_val = np.max(charge)
            charge_padded = np.concatenate([np.zeros(n_prepad), charge, np.full(n_postpad, max_val)])
            x = np.gradient(charge_padded, dt)
            y_shaped = fftconvolve(x, h_total, mode='full')[:len(x)]
            # Inject Gaussian noise here:
            if noise_amplitude > 0:
                y_shaped = y_shaped + np.random.normal(0, noise_amplitude, size=y_shaped.shape)
            y_trap = trap_recursive(y_shaped, L, G)
            y_tri = triangle_recursive(y_shaped, L_tri, G_tri)
            A = np.max(y_tri)
            E = np.max(y_trap)
            if E > 0:
                A_over_E_values.append(A / E)
                event_energies.append(energy)
        return np.array(A_over_E_values), np.array(event_energies)

def max_fom_vs_Ltri(L_tri, filepaths, noise_amplitude):
    # LP filter shaping time is fixed at 50 ns
    tau_rise = 50
    t_resp = np.arange(n_resp) * dt
    h_LP = np.sin(t_resp / (np.sqrt(2)*tau_rise)) * np.exp(-t_resp / (np.sqrt(2)*tau_rise))
    h_HP = np.exp(-t_resp / tau1)
    h_total = np.convolve(h_LP, h_HP)[:n_resp] * dt
    h_total = h_total / np.sum(h_total * dt)
    AE_values = []
    energies = []
    for fp in filepaths:
        vals, event_energies = process_file(fp, h_total, L_tri, noise_amplitude, stride=1)
        AE_values.append(vals)
        energies.append(event_energies)
    AE_values = np.concatenate(AE_values)
    energies = np.concatenate(energies)
    valid = (~np.isnan(AE_values)) & (AE_values > 0) & (AE_values < 1)
    A_over_E_clean = AE_values[valid]
    energies = energies[valid]
    energy_threshold = 1500
    valid_energy = energies > energy_threshold
    A_over_E_clean = A_over_E_clean[valid_energy]
    energies = energies[valid_energy]
    dep_mask = (energies >= 1589) & (energies <= 1595)
    sep_mask = (energies >= 2100) & (energies <= 2107)
    fep_mask = (energies >= 2609) & (energies <= 2621)
    A_over_E_fep = A_over_E_clean[fep_mask]
    A_over_E_dep = A_over_E_clean[dep_mask]
    A_over_E_sep = A_over_E_clean[sep_mask]
    A_over_E_bkg = np.concatenate([A_over_E_sep, A_over_E_fep])
    cut_values = np.linspace(np.min(A_over_E_dep), np.max(A_over_E_dep), 300)
    fom_list = []
    sig_acc_list = []
    bkg_acc_list = []
    signal_threshold = 0.85
    for cut in cut_values:
        sig_acc = np.mean(A_over_E_dep > cut)
        bkg_acc = np.mean(A_over_E_bkg > cut)
        if sig_acc < signal_threshold:
            fom = 0
        else:
            fom = sig_acc / np.sqrt(bkg_acc) if bkg_acc > 0 else 0
        fom_list.append(fom)
        sig_acc_list.append(sig_acc)
        bkg_acc_list.append(bkg_acc)
    fom_array = np.array(fom_list)
    sig_acc_array = np.array(sig_acc_list)
    best_idx = np.argmax(fom_array)
    return fom_array[best_idx]

file_template = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_{}_sc.hdf5"
filepaths = [r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc (1).hdf5"]
for i in range(2, 2):
    filepaths.append(file_template.format(i))

L_tri_values = np.arange(1, 16)  # 1 to 10
L_tri_ns = L_tri_values * dt     # For plotting: rise time in ns
noise_levels = np.array([0, 5e-6, 5e-5, 5e-4, 5e-3, 5e-2]) # e.g., 5e-5 to 0.005 (you can tweak the range or count if needed)

plt.figure(figsize=(8,6))

# Store best results for each noise amplitude
best_FoMs = []
best_risetimes = []

for noise_amplitude in noise_levels:
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(max_fom_vs_Ltri)(L_tri, filepaths, noise_amplitude) for L_tri in L_tri_values
    )
    plt.plot(L_tri_ns, results, marker='o', ms = 3, label=f'Noise: {noise_amplitude:.1e}', alpha=0.6)
    
    idx_best = np.argmax(results)
    best_FoMs.append(results[idx_best])
    best_risetimes.append(L_tri_ns[idx_best])

plt.xlabel("Triangle Filter Rise Time (ns)")
plt.ylabel("Maximum FoM")
plt.title("FoM vs Triangle Filter Rise Time")
plt.legend(title="Noise Amplitude", fontsize=10)
plt.grid()
plt.tight_layout()
plt.show()


# In[ ]:


#Ringing Response

import h5py
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy.signal import fftconvolve

# --- Constants ---
prepad_ns = 45000
postpad_ns = 20000
n_resp = 8000
dt = 16
tau1 = 3_000_000
L = int(4000 / dt)
G = int(2500 / dt)
L_tri = 1
G_tri = 0
n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))

@njit
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2*L + G, 1), N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] + signal[k - 2*L - G] - signal[k - L - G])
    return S / L

@njit
def triangle_recursive(signal, L, G):
    S = trap_recursive(signal, L, G)
    N = len(S)
    T = np.zeros(N)
    M = 2*L + G
    for k in range(M, N):
        T[k] = T[k-1] + S[k] - S[k - M]
    return T / M

def build_composite_response(dt, n_resp, tau1, tau2, freq, amp):
    t_resp = np.arange(n_resp) * dt
    h_HP = np.exp(-t_resp / tau1) + amp * np.exp(-t_resp / tau2) * np.sin(2 * np.pi * freq * t_resp * 1e-9)
    return h_HP

def process_file(filename, h_total, noise_amplitude=0.0002, stride=1):
    with h5py.File(filename, "r") as f:
        values = f["pss/pss/waveform/values/flattened_data"][:]
        length = f["pss/pss/waveform/values/cumulative_length"][:]
        edep_flat = f["pss/truth/edep/flattened_data"][:]
        edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
        num_events = len(length)
        event_indices = range(0, num_events, stride)
        dep_win = (1591, 1594)
        sep_win = (2102, 2105)
        fep_win = (2611, 2619)
        A_over_E_values = []
        event_energies = []
        for i in event_indices:
            edep_start = edep_cumlen[i-1] if i > 0 else 0
            edep_end = edep_cumlen[i]
            energy = np.sum(edep_flat[edep_start:edep_end])
            if not ((dep_win[0] <= energy <= dep_win[1]) or 
                    (sep_win[0] <= energy <= sep_win[1]) or 
                    (fep_win[0] <= energy <= fep_win[1])):
                continue
            start_idx = length[i-1] if i > 0 else 0
            end_idx = length[i]
            charge = values[start_idx:end_idx]
            max_val = np.max(charge)
            charge_padded = np.concatenate([np.zeros(n_prepad), charge, np.full(n_postpad, max_val)])
            x = np.gradient(charge_padded, dt)
            y_shaped = fftconvolve(x, h_total, mode='full')[:len(x)]
            y_noisy = y_shaped
            y_trap = trap_recursive(y_noisy, L, G)
            y_tri = triangle_recursive(y_noisy, L_tri, G_tri)
            A = np.max(y_tri)
            E = np.max(y_trap)
            if E > 0:
                A_over_E_values.append(A / E)
                event_energies.append(energy)
        return np.array(A_over_E_values), np.array(event_energies)

def max_fom_vs_ring_freq(ring_freq, filepaths, tau1=3_000_000, tau2=50_000, amp=0.009):
    tau_rise = 50 
    t_resp = np.arange(n_resp) * dt
    h_LP = np.sin(t_resp / (np.sqrt(2)*tau_rise)) * np.exp(-t_resp / (np.sqrt(2)*tau_rise))
    h_HP = build_composite_response(dt, n_resp, tau1, tau2, ring_freq, amp)
    h_total = np.convolve(h_LP, h_HP)[:n_resp] * dt
    h_total = h_total / np.sum(np.abs(h_total))
    AE_values = []
    energies = []
    for fp in filepaths:
        vals, event_energies = process_file(fp, h_total, noise_amplitude=0.0002, stride=1)
        AE_values.append(vals)
        energies.append(event_energies)
    AE_values = np.concatenate(AE_values)
    energies = np.concatenate(energies)
    valid = (~np.isnan(AE_values)) & (AE_values > 0) & (AE_values < 0.15)
    A_over_E_clean = AE_values[valid]
    energies = energies[valid]
    energy_threshold = 1500
    valid_energy = energies > energy_threshold
    A_over_E_clean = A_over_E_clean[valid_energy]
    energies = energies[valid_energy]
    dep_mask = (energies >= 1591) & (energies <= 1594)
    sep_mask = (energies >= 2102) & (energies <= 2105)
    fep_mask = (energies >= 2611) & (energies <= 2619)
    A_over_E_fep = A_over_E_clean[fep_mask]
    A_over_E_dep = A_over_E_clean[dep_mask]
    A_over_E_sep = A_over_E_clean[sep_mask]
    A_over_E_bkg = np.concatenate([A_over_E_sep, A_over_E_fep])
    cut_values = np.linspace(np.min(A_over_E_dep), np.max(A_over_E_dep), 300)
    fom_list = []
    sig_acc_list = []
    bkg_acc_list = []
    signal_threshold = 0.84
    for cut in cut_values:
        sig_acc = np.mean(A_over_E_dep > cut)
        bkg_acc = np.mean(A_over_E_bkg > cut)
        if sig_acc < signal_threshold:
            fom = 0
        else:
            fom = sig_acc / np.sqrt(bkg_acc) if bkg_acc > 0 else 0
        fom_list.append(fom)
        sig_acc_list.append(sig_acc)
        bkg_acc_list.append(bkg_acc)
    fom_array = np.array(fom_list)
    sig_acc_array = np.array(sig_acc_list)
    best_idx = np.argmax(fom_array)
    best_sig_idx = np.argmax(sig_acc_array)
    optimal_cut = cut_values[best_idx]
    optimal_fom = fom_array[best_idx]
    max_sig_acc = sig_acc_array[best_sig_idx]
    cut_at_max_sig_acc = cut_values[best_sig_idx]
    return optimal_fom, optimal_cut, max_sig_acc, cut_at_max_sig_acc

file_template = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_{}_sc.hdf5"
filepaths = [r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc (1).hdf5"]
for i in range(2, 11):
    filepaths.append(file_template.format(i))

ring_freqs = np.arange(0, 3.6e6, 0.025e6) 

results = Parallel(n_jobs=-1, backend='loky', verbose=10)(delayed(max_fom_vs_ring_freq)(freq, filepaths) for freq in ring_freqs)
fom_values, cut_values, max_sig_acc_values, cut_at_max_sig_acc_values = zip(*results)

plt.figure()
plt.plot(ring_freqs * 1e-6, fom_values, marker='o', ms = 3)
plt.xlabel("Ringing Frequency (MHz)")
plt.ylabel("Maximum FoM")
plt.title("Maximum FoM vs Ringing Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[41]:


# LP Filter: Signal Efficiency at Fixed Background
import h5py
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy.signal import fftconvolve


prepad_ns = 45000
postpad_ns = 20000
n_resp = 8000
dt = 16
tau1 = 3_000_000
L = int(4000 / dt)   # trapezoid rise (samples)
G = int(2500 / dt)   # trapezoid flat-top (samples)
L_tri = 1
G_tri = 0
n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))
noise_amplitude = 0.0005


@njit
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2*L + G, 1), N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] +
                         signal[k - 2*L - G] - signal[k - L - G])
    return S / L

@njit
def triangle_recursive(signal, L, G):
    S = trap_recursive(signal, L, G)
    N = len(S)
    T = np.zeros(N)
    M = 2*L + G
    for k in range(M, N):
        T[k] = T[k-1] + S[k] - S[k - M]
    return T / M


def process_file(filename, h_total, stride=1):
    with h5py.File(filename, "r") as f:
        values = f["pss/pss/waveform/values/flattened_data"][:]
        length = f["pss/pss/waveform/values/cumulative_length"][:]
        edep_flat = f["pss/truth/edep/flattened_data"][:]
        edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
        num_events = len(length)
        event_indices = range(0, num_events, stride)

        dep_win = (1589, 1595)
        sep_win = (2100, 2107)
        fep_win = (2609, 2621)

        A_over_E_values = []
        event_energies = []

        for i in event_indices:
            edep_start = edep_cumlen[i-1] if i > 0 else 0
            edep_end = edep_cumlen[i]
            energy = np.sum(edep_flat[edep_start:edep_end])
            if not ((dep_win[0] <= energy <= dep_win[1]) or 
                    (sep_win[0] <= energy <= sep_win[1]) or 
                    (fep_win[0] <= energy <= fep_win[1])):
                continue  # skip event if not in peak window

            start_idx = length[i-1] if i > 0 else 0
            end_idx = length[i]
            charge = values[start_idx:end_idx]
            max_val = np.max(charge)
            charge_padded = np.concatenate([np.zeros(n_prepad),
                                            charge,
                                            np.full(n_postpad, max_val)])
            x = np.gradient(charge_padded, dt)
            y_shaped = fftconvolve(x, h_total, mode='full')[:len(x)]
            y_noisy = y_shaped
            y_trap = trap_recursive(y_noisy, L, G)
            y_tri = triangle_recursive(y_noisy, L_tri, G_tri)
            A = np.max(y_tri)
            E = np.max(y_trap)
            if E > 0:
                A_over_E_values.append(A / E)
                event_energies.append(energy)

        return np.array(A_over_E_values), np.array(event_energies)


def sig_eff_at_fixed_bkg(tau_rise, filepaths, bkg_target=0.075):
    # Construct LP + HP response
    t_resp = np.arange(n_resp) * dt
    h_LP = np.sin(t_resp / (np.sqrt(2)*tau_rise)) * np.exp(-t_resp / (np.sqrt(2)*tau_rise))
    h_HP = np.exp(-t_resp / tau1)
    h_total = np.convolve(h_LP, h_HP)[:n_resp] * dt
    h_total = h_total / np.sum(h_total * dt)

    AE_values, energies = [], []
    for fp in filepaths:
        vals, event_energies = process_file(fp, h_total, stride=1)
        AE_values.append(vals)
        energies.append(event_energies)
    AE_values = np.concatenate(AE_values)
    energies = np.concatenate(energies)

    valid = (~np.isnan(AE_values)) & (AE_values > 0) & (AE_values < 0.12)
    A_over_E_clean = AE_values[valid]
    energies = energies[valid]
    valid_energy = energies > 1500
    A_over_E_clean = A_over_E_clean[valid_energy]
    energies = energies[valid_energy]

    dep_mask = (energies >= 1589) & (energies <= 1595)
    sep_mask = (energies >= 2100) & (energies <= 2107)
    fep_mask = (energies >= 2609) & (energies <= 2621)

    A_over_E_fep = A_over_E_clean[fep_mask]
    A_over_E_dep = A_over_E_clean[dep_mask]
    A_over_E_sep = A_over_E_clean[sep_mask]
    A_over_E_bkg = np.concatenate([A_over_E_sep, A_over_E_fep])

    cut_values = np.linspace(np.min(A_over_E_dep), np.max(A_over_E_dep), 300)
    sig_acc_list, bkg_acc_list = [], []
    for cut in cut_values:
        sig_acc = np.mean(A_over_E_dep > cut)
        bkg_acc = np.mean(A_over_E_bkg > cut)
        sig_acc_list.append(sig_acc)
        bkg_acc_list.append(bkg_acc)

    sig_acc_list = np.array(sig_acc_list)
    bkg_acc_list = np.array(bkg_acc_list)

    if bkg_target < bkg_acc_list.min(): 
        bkg_target = bkg_acc_list.min()
    if bkg_target > bkg_acc_list.max():
        bkg_target = bkg_acc_list.max()

    # Interpolate
    sig_eff = float(np.interp(bkg_target, bkg_acc_list[::-1], sig_acc_list[::-1]))
    return sig_eff

file_template = r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_{}_sc.hdf5"
filepaths = [r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc (1).hdf5"]
for i in range(2, 2):  # add more files if needed
    filepaths.append(file_template.format(i))

lp_cutoff_freqs_MHz = np.arange(1.0, 10.1, 0.1)  # 1–10 MHz
lp_cutoff_freqs_Hz = lp_cutoff_freqs_MHz * 1e6
tau_rise_values = (1 / (2 * np.pi * lp_cutoff_freqs_Hz)) * 1e9  # ns

# Run in parallel
results_sig_eff = Parallel(n_jobs=-1, backend='loky', verbose=10)(
    delayed(sig_eff_at_fixed_bkg)(tau_rise, filepaths, bkg_target=0.075) 
    for tau_rise in tau_rise_values
)


results_sig_eff = np.array(results_sig_eff).astype(float)

plt.figure()
plt.plot(lp_cutoff_freqs_MHz, results_sig_eff*100, marker='o', ms=3)
plt.xlabel("LP Filter Cutoff Frequency (MHz)")
plt.ylabel("Signal Acceptance %")
plt.title("Signal Acceptance vs LP Cutoff Frequency (Fixed Background = 7.5%)")
plt.grid()
plt.tight_layout()
plt.show()


# In[4]:


# baseline_drift_scan.py

import h5py
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy.signal import fftconvolve

# --- constants ---
prepad_ns = 45000
postpad_ns = 20000
n_resp = 8000
dt = 16
tau1 = 3_000_000
L = int(4000 / dt)
G = int(2500 / dt)
L_tri, G_tri = 1, 0
n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))

# --- filters ---
@njit
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2*L + G, 1), N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] +
                         signal[k - 2*L - G] - signal[k - L - G])
    return S / L

@njit
def triangle_recursive(signal, L, G):
    S = trap_recursive(signal, L, G)
    N = len(S)
    T = np.zeros(N)
    M = 2*L + G
    for k in range(M, N):
        T[k] = T[k-1] + S[k] - S[k - M]
    return T / M

# --- helper: add baseline drift ---
def add_baseline_drift(signal, frac=0.0):
    return signal + frac * np.max(signal)

# --- process file with drift ---
def process_file_with_drift(filename, h_total, drift_frac, stride=1):
    with h5py.File(filename, "r") as f:
        values = f["pss/pss/waveform/values/flattened_data"][:]
        length = f["pss/pss/waveform/values/cumulative_length"][:]
        edep_flat = f["pss/truth/edep/flattened_data"][:]
        edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
        num_events = len(length)

        dep_win = (1589, 1595)
        sep_win = (2100, 2107)
        fep_win = (2609, 2621)

        A_over_E_values, event_energies = [], []
        for i in range(0, num_events, stride):
            edep_start = edep_cumlen[i-1] if i > 0 else 0
            edep_end = edep_cumlen[i]
            energy = np.sum(edep_flat[edep_start:edep_end])
            if not ((dep_win[0] <= energy <= dep_win[1]) or 
                    (sep_win[0] <= energy <= sep_win[1]) or 
                    (fep_win[0] <= energy <= fep_win[1])):
                continue

            start_idx = length[i-1] if i > 0 else 0
            end_idx = length[i]
            charge = values[start_idx:end_idx]

            # add drift
            charge = add_baseline_drift(charge, drift_frac)

            charge_padded = np.concatenate([np.zeros(n_prepad),
                                            charge,
                                            np.full(n_postpad, np.max(charge))])
            x = np.gradient(charge_padded, dt)
            y_shaped = fftconvolve(x, h_total, mode='full')[:len(x)]
            y_trap = trap_recursive(y_shaped, L, G)
            y_tri = triangle_recursive(y_shaped, L_tri, G_tri)

            A, E = np.max(y_tri), np.max(y_trap)
            if E > 0:
                A_over_E_values.append(A / E)
                event_energies.append(energy)

        return np.array(A_over_E_values), np.array(event_energies)

# --- FoM ---
def fom_vs_drift(drift_frac, filepaths, h_total):
    AE_values, energies = [], []
    for fp in filepaths:
        vals, evs = process_file_with_drift(fp, h_total, drift_frac)
        AE_values.append(vals)
        energies.append(evs)
    AE_values = np.concatenate(AE_values)
    energies = np.concatenate(energies)

    valid = (~np.isnan(AE_values)) & (AE_values > 0) & (AE_values < 0.12)
    A_over_E_clean, energies = AE_values[valid], energies[valid]
    dep_mask = (energies >= 1589) & (energies <= 1595)
    sep_mask = (energies >= 2100) & (energies <= 2107)
    fep_mask = (energies >= 2609) & (energies <= 2621)

    dep, bkg = A_over_E_clean[dep_mask], np.concatenate([A_over_E_clean[sep_mask],
                                                         A_over_E_clean[fep_mask]])

    cut_values = np.linspace(np.min(dep), np.max(dep), 300)
    foms = []
    for cut in cut_values:
        sig_acc = np.mean(dep > cut)
        bkg_acc = np.mean(bkg > cut)
        foms.append(sig_acc / np.sqrt(bkg_acc) if bkg_acc > 0 else 0)
    return np.max(foms)

# --- fixed LP filter at 5 MHz ---
fc = 5e6
tau_rise = (1 / (2 * np.pi * fc)) * 1e9
t_resp = np.arange(n_resp) * dt
h_LP = np.sin(t_resp / (np.sqrt(2)*tau_rise)) * np.exp(-t_resp / (np.sqrt(2)*tau_rise))
h_HP = np.exp(-t_resp / tau1)
h_total = np.convolve(h_LP, h_HP)[:n_resp] * dt
h_total = h_total / np.sum(h_total * dt)

# --- files ---
filepaths = [r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc (1).hdf5"]

# --- scan drift fractions ---
drift_fracs = np.linspace(-0.1, 0.1, 21)
results = Parallel(n_jobs=-1)(
    delayed(fom_vs_drift)(d, filepaths, h_total) for d in drift_fracs
)

plt.plot(drift_fracs*100, results, marker='o')
plt.xlabel("Baseline Drift (% of pulse height)")
plt.ylabel("FoM")
plt.title("FoM vs Baseline Drift")
plt.grid()
plt.show()


# In[6]:


# pileup_scan.py

import h5py
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from joblib import Parallel, delayed
from scipy.signal import fftconvolve

# --- constants ---
prepad_ns = 45000
postpad_ns = 20000
n_resp = 8000
dt = 16
tau1 = 3_000_000
L = int(4000 / dt)
G = int(2500 / dt)
L_tri, G_tri = 1, 0
n_prepad = int(np.ceil(prepad_ns / dt))
n_postpad = int(np.ceil(postpad_ns / dt))

@njit
def trap_recursive(signal, L, G):
    N = len(signal)
    S = np.zeros(N)
    for k in range(max(2*L + G, 1), N):
        S[k] = S[k-1] + (signal[k] - signal[k-L] +
                         signal[k - 2*L - G] - signal[k - L - G])
    return S / L

@njit
def triangle_recursive(signal, L, G):
    S = trap_recursive(signal, L, G)
    N = len(S)
    T = np.zeros(N)
    M = 2*L + G
    for k in range(M, N):
        T[k] = T[k-1] + S[k] - S[k - M]
    return T / M

def add_pileup(sig1, sig2, delay_samples, alpha=1.0):
    # Pad sig2 with delay
    sig2_shift = np.pad(sig2, (delay_samples, 0), mode="constant")
    
    # Match lengths
    N = max(len(sig1), len(sig2_shift))
    sig1_padded = np.pad(sig1, (0, N - len(sig1)), mode="constant")
    sig2_padded = np.pad(sig2_shift, (0, N - len(sig2_shift)), mode="constant")
    
    return sig1_padded + alpha * sig2_padded
def process_file_with_pileup(filename, h_total, delay_samples, alpha=1.0):
    with h5py.File(filename, "r") as f:
        values = f["pss/pss/waveform/values/flattened_data"][:]
        length = f["pss/pss/waveform/values/cumulative_length"][:]
        edep_flat = f["pss/truth/edep/flattened_data"][:]
        edep_cumlen = f["pss/truth/edep/cumulative_length"][:]
        num_events = len(length)

        dep_win = (1589, 1595)
        sep_win = (2100, 2107)
        fep_win = (2609, 2621)

        A_over_E_values, event_energies = [], []
        for i in range(num_events-1):
            sig1 = values[(length[i-1] if i > 0 else 0):length[i]]
            sig2 = values[length[i]:length[i+1]]

            sig_pu = add_pileup(sig1, sig2, delay_samples, alpha)

            sig_pu = np.concatenate([np.zeros(n_prepad),
                                     sig_pu,
                                     np.full(n_postpad, np.max(sig_pu))])
            x = np.gradient(sig_pu, dt)
            y_shaped = fftconvolve(x, h_total, mode='full')[:len(x)]
            y_trap = trap_recursive(y_shaped, L, G)
            y_tri = triangle_recursive(y_shaped, L_tri, G_tri)

            A, E = np.max(y_tri), np.max(y_trap)
            if E > 0:
                A_over_E_values.append(A / E)
                event_energies.append(np.sum(edep_flat[edep_cumlen[i-1] if i > 0 else 0:edep_cumlen[i]]))

        return np.array(A_over_E_values), np.array(event_energies)

def fom_vs_pileup(delay_samples, filepaths, h_total, alpha=1.0):
    AE_values, energies = [], []
    for fp in filepaths:
        vals, evs = process_file_with_pileup(fp, h_total, delay_samples, alpha)
        AE_values.append(vals)
        energies.append(evs)
    AE_values = np.concatenate(AE_values)
    energies = np.concatenate(energies)

    valid = (~np.isnan(AE_values)) & (AE_values > 0) & (AE_values < 0.12)
    A_over_E_clean, energies = AE_values[valid], energies[valid]
    dep_mask = (energies >= 1589) & (energies <= 1595)
    sep_mask = (energies >= 2100) & (energies <= 2107)
    fep_mask = (energies >= 2609) & (energies <= 2621)

    dep, bkg = A_over_E_clean[dep_mask], np.concatenate([A_over_E_clean[sep_mask],
                                                         A_over_E_clean[fep_mask]])

    cut_values = np.linspace(np.min(dep), np.max(dep), 300)
    foms = []
    for cut in cut_values:
        sig_acc = np.mean(dep > cut)
        bkg_acc = np.mean(bkg > cut)
        foms.append(sig_acc / np.sqrt(bkg_acc) if bkg_acc > 0 else 0)
    return np.max(foms)

# --- fixed LP filter at 5 MHz ---
fc = 5e6
tau_rise = (1 / (2 * np.pi * fc)) * 1e9
t_resp = np.arange(n_resp) * dt
h_LP = np.sin(t_resp / (np.sqrt(2)*tau_rise)) * np.exp(-t_resp / (np.sqrt(2)*tau_rise))
h_HP = np.exp(-t_resp / tau1)
h_total = np.convolve(h_LP, h_HP)[:n_resp] * dt
h_total = h_total / np.sum(h_total * dt)

filepaths = [r"C:\Users\nikhi\Downloads\V09372A_ideal_pulses_1_sc (1).hdf5"]

# delays in samples (convert to ns via dt)
delays = [0, 50, 100, 200, 500, 1000]  # e.g. 0–16 µs
results = Parallel(n_jobs=-1)(
    delayed(fom_vs_pileup)(d, filepaths, h_total, alpha=1.0) for d in delays
)

plt.plot(np.array(delays)*dt, results, marker='o')
plt.xlabel("Pile-Up Delay (ns)")
plt.ylabel("FoM")
plt.title("FoM vs Pile-Up Delay (LP=5 MHz, α=1.0)")
plt.grid()
plt.show()


# In[ ]:




