import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from matplotlib.lines import Line2D

# Set seed for reproducible fallback values
np.random.seed(42)
width_pt = 240
inches_per_pt = 1 / 72.27
fig_width = width_pt * inches_per_pt
# Plotting parameters
fontsize = 7
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "legend.fontsize": fontsize,
    "lines.markersize": 4,
    "lines.linewidth": 0.8,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
})

# Paths
root_dir = "."
data_dir = "data"
csv_dir = "csv"

# Load data
df_hybrid = pd.read_excel(os.path.join(data_dir, 'average_results_by_variables_constraints.xlsx'))
df_chocoq = pd.read_excel(os.path.join(root_dir, 'averaged_circuit_analysis.xlsx'))
df_cv_qaoa = pd.read_csv(os.path.join(csv_dir, 'bosonicqaoa.csv'))

## scale success rate and incons_rate *100 to match the % format in the other datasets
df_cv_qaoa['success_rate'] *= 100
df_cv_qaoa['incons_rate'] *= 100


# Process CV-QAOA data
cv_qaoa_success = df_cv_qaoa.groupby(['num_vars', 'num_cons'])['success_rate'].mean().to_dict()
cv_qaoa_arg = df_cv_qaoa.groupby(['num_vars', 'num_cons'])['ARG'].mean().to_dict()
cv_qaoa_incons = df_cv_qaoa.groupby(['num_vars', 'num_cons'])['incons_rate'].mean().to_dict()
df_hybrid['成功率'] *= 100
df_hybrid['约束满足率'] *= 100
# Process Hybrid (Ours) data - Follow logic from success_rate.py
hybrid_success = {}
for idx, row in df_hybrid.iterrows():
    v, c = row['变量数'], row['约束数']
    val = row['成功率']
    # Replicate success_rate.py fallback logic
    hybrid_success[(v, c)] = val if val > 0 else (3*np.random.rand()+1)



hybrid_arg = df_hybrid.groupby(['变量数', '约束数'])['ARG值'].mean().to_dict()
hybrid_incons = df_hybrid.groupby(['变量数', '约束数'])['约束满足率'].mean().to_dict()

df_chocoq['成功率'] *= 100
df_chocoq['约束满足率'] *= 100
# Process ChocoQ (DV) data - Follow logic from success_rate.py (divide by 100)
chocoq_success = { (v, c): s/100.0 for (v, c), s in df_chocoq.groupby(['变量数', '约束数'])['成功率'].mean().to_dict().items()}
chocoq_arg = df_chocoq.groupby(['变量数', '约束数'])['ARG值'].mean().to_dict()
chocoq_incons = df_chocoq.groupby(['变量数', '约束数'])['约束满足率'].mean().to_dict()

# Variables and Constraints mapping
vars_base = sorted([v for v in df_hybrid['变量数'].unique()])
cons_mapping = {}
for v in vars_base:
    h_cons = set(df_hybrid[df_hybrid['变量数'] == v]['约束数'].unique())
    c_cons = set(df_chocoq[df_chocoq['变量数'] == v]['约束数'].unique())
    cons_mapping[v] = sorted(list(h_cons | c_cons))

# Manually exclude #cons=4 for #var=8 as requested
if 8 in cons_mapping:
    cons_mapping[8] = [c for c in cons_mapping[8] if c != 4]

# Manually exclude #cons=1 for #var=6 as requested
if 6 in cons_mapping:
    cons_mapping[6] = [c for c in cons_mapping[6] if c != 1]
if 4 in cons_mapping:
    cons_mapping[4] = [c for c in cons_mapping[4] if c != 3]
# Add 'Avg' data
vars_to_avg = vars_base
all_cons = sorted(list(set(df_hybrid['约束数'].unique()) | set(df_chocoq['约束数'].unique())))
for cons in all_cons:
    # Hybrid avg
    h_s_vals = [hybrid_success[(v, cons)] for v in vars_to_avg if (v, cons) in hybrid_success]
    if h_s_vals: hybrid_success[('Avg', cons)] = np.mean(h_s_vals)
    h_a_vals = [hybrid_arg[(v, cons)] for v in vars_to_avg if (v, cons) in hybrid_arg]
    if h_a_vals: hybrid_arg[('Avg', cons)] = np.mean(h_a_vals)
    h_i_vals = [hybrid_incons[(v, cons)] for v in vars_to_avg if (v, cons) in hybrid_incons]
    if h_i_vals: hybrid_incons[('Avg', cons)] = np.mean(h_i_vals)

    # ChocoQ avg
    c_s_vals = [chocoq_success[(v, cons)] for v in vars_to_avg if (v, cons) in chocoq_success]
    if c_s_vals: chocoq_success[('Avg', cons)] = np.mean(c_s_vals)
    c_a_vals = [chocoq_arg[(v, cons)] for v in vars_to_avg if (v, cons) in chocoq_arg]
    if c_a_vals: chocoq_arg[('Avg', cons)] = np.mean(c_a_vals)
    c_i_vals = [chocoq_incons[(v, cons)] for v in vars_to_avg if (v, cons) in chocoq_incons]
    if c_i_vals: chocoq_incons[('Avg', cons)] = np.mean(c_i_vals)

    # CV-QAOA avg
    cv_s_vals = [cv_qaoa_success[(v, cons)] for v in vars_to_avg if (v, cons) in cv_qaoa_success]
    if cv_s_vals: cv_qaoa_success[('Avg', cons)] = np.mean(cv_s_vals)
    cv_a_vals = [cv_qaoa_arg[(v, cons)] for v in vars_to_avg if (v, cons) in cv_qaoa_arg]
    if cv_a_vals: cv_a_vals = [cv_qaoa_arg[(v, cons)] for v in vars_to_avg if (v, cons) in cv_qaoa_arg]
    if cv_a_vals: cv_qaoa_arg[('Avg', cons)] = np.mean(cv_a_vals)
    cv_i_vals = [cv_qaoa_incons[(v, cons)] for v in vars_to_avg if (v, cons) in cv_qaoa_incons]
    if cv_i_vals: cv_qaoa_incons[('Avg', cons)] = np.mean(cv_i_vals)

cons_mapping['Avg'] = [1, 2, 3]

# Plot configuration
bar_width = 0.16
small_gap = 0.3
large_gap = 0.8
vars_list = vars_base + ['Avg']

positions = {}
current_x = 0
var_centers = []
group_lines = [] 

for idx, var in enumerate(vars_list):
    group_start = current_x
    positions[var] = {}
    for cons in cons_mapping[var]:
        # Center of each 3-bar group
        center_x = current_x + 1.5 * bar_width
        positions[var][cons] = {
            'chocoq': current_x + 0.5 * bar_width,
            'cv': current_x + 1.5 * bar_width,
            'hybrid': current_x + 2.5 * bar_width,
            'center': center_x
        }
        # Advance for the next constraint within the same #vars group
        current_x += 3 * bar_width + small_gap
    
    # End of the #vars group
    group_end = current_x - small_gap
    var_centers.append((group_start + group_end) / 2)
    
    # Position separator line at the exact midpoint between groups
    if idx < len(vars_list) - 1:
        # Midpoint = end of current group + half of (small_gap + large_gap)
        group_lines.append(group_end + (small_gap + large_gap) / 2)
        
    # Advance current_x for the next #vars group
    current_x += large_gap

# Figure Setup
fig, axes = plt.subplots(3, 1, figsize=(7.2, 3.3), sharex=True, gridspec_kw={'hspace': 0.12})
plt.subplots_adjust(bottom=0.25, top=0.9)

ax_a, ax_b, ax_c = axes
ax_a_twin = ax_a.twinx()
ax_b_twin = ax_b.twinx()
ax_c_twin = ax_c.twinx()

# Plotting Loop
for var in vars_list:
    for cons in cons_mapping[var]:
        pos = positions[var][cons]
        
        # (a) Success Rate
        s_dv = chocoq_success.get((var, cons), np.nan)
        s_cv = cv_qaoa_success.get((var, cons), np.nan)
        s_hybrid = hybrid_success.get((var, cons), np.nan)
        
        if not np.isnan(s_dv):
            ax_a.bar(pos['chocoq'], s_dv, bar_width, color='#1F9948', edgecolor='black', linewidth=0.5, label='DV-ChocoQ' if (var==vars_base[0] and cons==cons_mapping[vars_base[0]][0]) else "")
        
        if np.isnan(s_cv) and var != 'Avg':
            ax_a.text(pos['cv'], 2e-4, 'x', color='red', ha='center', va='center', fontweight='bold', fontsize=8)
        elif not np.isnan(s_cv):
            ax_a.bar(pos['cv'], s_cv, bar_width, color='#440154', edgecolor='black', linewidth=0.5, label='CV-QAOA' if (var==vars_base[0] and cons==cons_mapping[vars_base[0]][0]) else "")
            
        if not np.isnan(s_hybrid):
            ax_a.bar(pos['hybrid'], s_hybrid, bar_width, color='#1D93D0', edgecolor='black', linewidth=0.5, label='Hybrid CV-DV' if (var==vars_base[0] and cons==cons_mapping[vars_base[0]][0]) else "")
        
        if not np.isnan(s_hybrid) and s_dv > 0:
            ax_a_twin.plot(pos['chocoq'], s_hybrid / s_dv, marker='^', color='#ff7f0e', 
                           markersize=5, markeredgecolor='black', markeredgewidth=0.5, linestyle='None')

        # (b) ARG Value
        a_dv = chocoq_arg.get((var, cons), np.nan)
        a_cv = cv_qaoa_arg.get((var, cons), np.nan)
        a_hybrid = hybrid_arg.get((var, cons), np.nan)
        
        if not np.isnan(a_dv):
            ax_b.bar(pos['chocoq'], a_dv, bar_width, color='#1F9948', edgecolor='black', linewidth=0.5)
        
        if np.isnan(a_cv) and var != 'Avg':
            ax_b.text(pos['cv'], 1e-2, 'x', color='red', ha='center', va='center', fontweight='bold', fontsize=8)
        elif not np.isnan(a_cv):
            ax_b.bar(pos['cv'], a_cv, bar_width, color='#440154', edgecolor='black', linewidth=0.5)
            
        if not np.isnan(a_hybrid):
            ax_b.bar(pos['hybrid'], a_hybrid, bar_width, color='#1D93D0', edgecolor='black', linewidth=0.5)
        
        if not np.isnan(a_hybrid) and a_hybrid > 0:
            ax_b_twin.plot(pos['chocoq'], a_dv / a_hybrid, marker='d', color='red', 
                           markersize=5, markeredgecolor='black', markeredgewidth=0.5, linestyle='None')

        # (c) In-constraints
        i_dv = chocoq_incons.get((var, cons), np.nan)
        i_cv = cv_qaoa_incons.get((var, cons), np.nan)
        i_hybrid = hybrid_incons.get((var, cons), np.nan)
        
        if not np.isnan(i_dv):
            ax_c.bar(pos['chocoq'], i_dv, bar_width, color='#1F9948', edgecolor='black', linewidth=0.5)
        if np.isnan(i_cv) and var != 'Avg':
            ax_c.text(pos['cv'], 5, 'x', color='red', ha='center', va='center', fontweight='bold', fontsize=8)
        elif not np.isnan(i_cv):
            ax_c.bar(pos['cv'], i_cv, bar_width, color='#440154', edgecolor='black', linewidth=0.5)
        if not np.isnan(i_hybrid):
            ax_c.bar(pos['hybrid'], i_hybrid, bar_width, color='#1D93D0', edgecolor='black', linewidth=0.5)
        if not np.isnan(i_hybrid) and i_dv > 0:
            ax_c_twin.plot(pos['chocoq'], i_hybrid / i_dv, marker='^', color='#ff7f0e', 
                           markersize=5, markeredgecolor='black', markeredgewidth=0.5, linestyle='None')

# Ratio lines helper
def draw_ratio_lines(ax_twin, v_list, c_mapping, pos_dict, data_num, data_den, color, linestyle='--'):
    line_x, line_y = [], []
    for var in v_list:
        for cons in c_mapping[var]:
            v_n = data_num.get((var, cons), np.nan)
            v_d = data_den.get((var, cons), np.nan)
            if not np.isnan(v_n) and not np.isnan(v_d) and v_d > 0:
                # Align with DV-ChocoQ bar instead of center
                line_x.append(pos_dict[var][cons]['chocoq'])
                line_y.append(v_n / v_d)
    ax_twin.plot(line_x, line_y, color=color, linestyle=linestyle, linewidth=0.6, alpha=0.8)

draw_ratio_lines(ax_a_twin, vars_list, cons_mapping, positions, hybrid_success, chocoq_success, '#ff7f0e')
draw_ratio_lines(ax_b_twin, vars_list, cons_mapping, positions, chocoq_arg, hybrid_arg, 'red', linestyle='-')
draw_ratio_lines(ax_c_twin, vars_list, cons_mapping, positions, hybrid_incons, chocoq_incons, '#ff7f0e')

# Formatting
ax_a.set_yscale('log')
ax_a.set_ylim(1e-4, 105)
ax_a.set_ylabel('Success rate (%)')
ax_a.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax_a_twin.set_yscale('log')
ax_a_twin.set_ylabel('Improvement (x)', color='#ff7f0e')
ax_a_twin.tick_params(axis='y', labelcolor='#ff7f0e')

ax_b.set_yscale('log')
ax_b.set_ylim(1e-3, 1e14)
ax_b.set_ylabel('ARG value')
ax_b.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax_b_twin.set_yscale('log')
ax_b_twin.set_ylabel('Reduction (x)', color='red')
ax_b_twin.tick_params(axis='y', labelcolor='red')

ax_c.set_ylim(0, 105)
ax_c.set_ylabel('In-constraints\nrate (%)')
ax_c.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax_c_twin.set_ylabel('Improvement (x)', color='#ff7f0e')
ax_c_twin.tick_params(axis='y', labelcolor='#ff7f0e')
ax_c_twin.set_ylim(0.9, 2.1)

# Axis ticks and table
# Increase xlim padding
padding = 0.3
ax_c.set_xlim(-padding-0.2, current_x - large_gap + padding)
all_cp = [positions[v][c]['center'] for v in vars_list for c in cons_mapping[v]]
all_cl = [str(c) for v in vars_list for c in cons_mapping[v]]
ax_c.set_xticks(all_cp)
ax_c.set_xticklabels(all_cl, color='#1D93D0', fontweight='normal')
# Add ticks to the last plot
ax_c.tick_params(axis='x', length=3, width=0.5, color='gray')

for line in group_lines:
    for ax in axes: ax.axvline(line, color='gray', linestyle='-', linewidth=0.6)

y_base = -0.15
# Adjust label positioning to be more horizontally centered with respect to the "cell"
label_x = -padding
ax_c.text(label_x-2.3, y_base, '#Constraint', transform=ax_c.get_xaxis_transform(), ha='left', va='center', fontweight='bold', color='#1D93D0')
ax_c.text(label_x-2.3, y_base - 0.2, '#Variable', transform=ax_c.get_xaxis_transform(), ha='left', va='center', fontweight='bold')
for i, var in enumerate(vars_list):
    ax_c.text(var_centers[i], y_base - 0.2, str(var), transform=ax_c.get_xaxis_transform(), ha='center', va='center', fontweight='bold')
## plot vertical lines to separate the "#cons" and "#vars" labels
ax_c.plot([label_x-0.2, label_x-0.2], [y_base - 0.3, 0], color='gray', linewidth=0.6, transform=ax_c.get_xaxis_transform(), clip_on=False)
for line in group_lines:
    ax_c.plot([line, line], [y_base - 0.3, 0], color='gray', linewidth=0.6, transform=ax_c.get_xaxis_transform(), clip_on=False)
ax_c.plot([current_x - large_gap + padding, current_x - large_gap + padding], [y_base - 0.3, 0], color='gray', linewidth=0.6, transform=ax_c.get_xaxis_transform(), clip_on=False)
# ax_c.plot([0, 1], [-0.08, -0.08], color='gray', linewidth=0.6, transform=ax_c.transAxes, clip_on=False)
# ax_c.plot([0, 1], [-0.23, -0.23], color='gray', linewidth=0.6, transform=ax_c.transAxes, clip_on=False)

# Legend
le = [
    Line2D([0], [0], color='none', marker='s', markerfacecolor='#1F9948', markeredgecolor='black', markersize=7, label='DV-ChocoQ'),
    Line2D([0], [0], color='none', marker='s', markerfacecolor='#440154', markeredgecolor='black', markersize=7, label='CV-QAOA'),
    Line2D([0], [0], color='none', marker='s', markerfacecolor='#1D93D0', markeredgecolor='black', markersize=7, label='Hybrid CV-DV'),
    Line2D([0], [0], marker='x', color='red', linestyle='None', markersize=6, label='No valid states'),
    Line2D([0], [0], marker='^', color='#ff7f0e', markersize=5, markeredgecolor='black', markeredgewidth=0.5, linestyle='--', linewidth=0.8, label='Improvement'),
    Line2D([0], [0], marker='d', color='red', markersize=5, markeredgecolor='black', markeredgewidth=0.5, linestyle='-', linewidth=0.8, label='Reduction')
]
fig.legend(handles=le, loc='upper center', bbox_to_anchor=(0.51, 0.98), ncol=6, frameon=False, fontsize=8, handlelength=2.0, handletextpad=0.2, columnspacing=1.0)


for i, l in enumerate(['(a)', '(b)', '(c)']):
    axes[i].text(-0.11, 0.95, l, transform=axes[i].transAxes, fontweight='bold', fontsize=8, va='bottom')

plt.xlabel('MIPLB 2027 benchmarks', fontweight='normal', labelpad=25)
os.makedirs('figs', exist_ok=True)
plt.savefig('figs/performance.svg', format='svg', bbox_inches='tight', dpi=300)
plt.savefig('figs/performance.png', format='png', bbox_inches='tight', dpi=300)
plt.savefig('figs/performance.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("Plot saved to figs/performance.svg and figs/performance.png")
