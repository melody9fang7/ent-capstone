import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# these are from NSQIP I just didnt want to keep reloading the whole file

REVAL_BREAKPOINTS = {
    '38542': [2007, 2009, 2010],
    '42415': [2007, 2010, 2012],
    '42420': [2007, 2010, 2012],
    '42440': [2007, 2010, 2012],
    '60220': [2007, 2010, 2012],
    '60240': [2007, 2010, 2012],
}

REVAL_DIRECTIONS = {
    '38542': {2007: 'increase', 2009: 'increase', 2010: 'increase'},
    '42415': {2007: 'increase', 2010: 'increase', 2012: 'decrease'},
    '42420': {2007: 'increase', 2010: 'increase', 2012: 'decrease'},
    '42440': {2007: 'increase', 2010: 'increase', 2012: 'decrease'},
    '60220': {2007: 'increase', 2010: 'increase', 2012: 'decrease'},
    '60240': {2007: 'increase', 2010: 'increase', 2012: 'decrease'},
}


MIN_RVU_CHANGE_PCT = 0.05
YEAR_START = 2005
YEAR_END = 2022

CPT_GROUPS = {
    '38542': 'Neck Dissection',
    '42415': 'Salivary Gland',
    '42420': 'Salivary Gland',
    '42440': 'Salivary Gland',
    '60220': 'Thyroid',
    '60240': 'Thyroid',
}

TARGET_CPTS = ['38542', '42415', '42420', '42440', '60220', '60240']


# LOAD

def load_ent_codes(filepath):
    df = pd.read_csv(filepath)
    codes = set()
    for val in df['CPT Code']:
        try:
            if pd.notna(val):
                codes.add(str(int(float(str(val).strip()))))
        except:
            continue
    print(f"Loaded {len(codes)} ENT CPT codes")
    return codes


def load_mnpb(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded {len(df):,} MNPB rows")
    
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    df['HCPCS'] = df['HCPCS'].astype(str).str.strip()
    df['MODIFIER'] = df['MODIFIER'].astype(str).str.strip()
    
    for col in ['ALLOWED SERVICES', 'ALLOWED CHARGES', 'PAYMENT']:
        df[col] = df[col].astype(str).str.replace('$', '', regex=False)
        df[col] = df[col].str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[(df['YEAR'] >= YEAR_START) & (df['YEAR'] <= YEAR_END)]
    return df


# CALCULATE

def calculate_volumes(df_all, ent_codes):
    # Denominator: total Part B services (TOTAL modifier)
    total_all = df_all[(df_all['MODIFIER'] == 'TOTAL') & (df_all['ALLOWED SERVICES'].notna())]
    total_per_year = total_all.groupby('YEAR')['ALLOWED SERVICES'].sum().reset_index(name='total_services')
    
    # Numerator: ENT services (TOTAL modifier)
    df_ent = df_all[
        (df_all['HCPCS'].isin(ent_codes)) & 
        (df_all['MODIFIER'] == 'TOTAL') &
        (df_all['ALLOWED SERVICES'].notna())
    ]
    
    yearly = df_ent.groupby(['HCPCS', 'YEAR']).agg(
        services=('ALLOWED SERVICES', 'sum'),
        total_payment=('PAYMENT', 'sum'),
        total_charges=('ALLOWED CHARGES', 'sum')
    ).reset_index()
    
    yearly = yearly.merge(total_per_year, on='YEAR')
    yearly['volume_pct'] = yearly['services'] / yearly['total_services'] * 100
    yearly['avg_payment'] = yearly['total_payment'] / yearly['services']
    
    print(f"ENT CPTs with MNPB data: {yearly['HCPCS'].nunique()}")
    return yearly


# PLOT

def plot_mnpb_volumes(yearly, cpt_list, reval_map, direction_map, filename='mnpb_volume_selected.svg'):
    """
    MNPB volume trends with NSQIP revaluation year markers overlaid.
    """
    cpts_to_plot = [c for c in cpt_list if c in yearly['HCPCS'].values]
    if len(cpts_to_plot) == 0:
        print("No matching CPTs")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, cpt in enumerate(cpts_to_plot):
        ax = axes[idx]
        cpt_data = yearly[yearly['HCPCS'] == cpt].sort_values('YEAR')
        group = CPT_GROUPS.get(cpt, '')
        break_years = reval_map.get(cpt, [])
        
        if len(cpt_data) < 3:
            continue
        
        # Plot volume
        ax.plot(cpt_data['YEAR'], cpt_data['volume_pct'], 'o-', 
               color='steelblue', alpha=0.8, markersize=10, linewidth=2.5, 
               zorder=3)
        
        # Overlay breakpoints
        for by in break_years:
            direction = direction_map.get(cpt, {}).get(by, None)
            if direction == 'increase':
                color = 'green'
            elif direction == 'decrease':
                color = 'red'
            else:
                color = 'gray'
            ax.axvline(x=by, color=color, linestyle='--', linewidth=2, alpha=0.7, zorder=1)
        
        # Y-axis scaling
        y_min, y_max = cpt_data['volume_pct'].min(), cpt_data['volume_pct'].max()
        y_range = y_max - y_min
        padding = max(y_range * 0.15, 0.0005)
        ax.set_ylim(max(0, y_min - padding), y_max + padding)
        
        ax.set_xlabel('Year', fontsize=16, fontweight='bold')
        ax.set_ylabel('% of All Part B Services', fontsize=16, fontweight='bold')
        ax.set_title(f'CPT {cpt}: {group} (MNPB)', fontsize=20, fontweight='bold')
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        x_min, x_max = int(cpt_data['YEAR'].min()), int(cpt_data['YEAR'].max())
        tick_step = max(1, (x_max - x_min) // 5)
        ax.set_xticks(range(x_min, x_max + 1, tick_step))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
        
        total = cpt_data['services'].sum()
        ax.text(0.98, 0.96, f'Total: {total:,.0f} services', transform=ax.transAxes, va='top', ha='right', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    
    # hiding unused panels
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    # legend at the bottom of the figure
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='steelblue', linewidth=2.5, label='MNPB Volume'),
        Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='wRVU Increase'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='wRVU Decrease'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=14, 
              frameon=True, bbox_to_anchor=(0.5, 0.001))
    
    plt.suptitle('Medicare Part B Volume with NSQIP wRVU Revaluation Breakpoints\n'
                '(MNPB TOTAL Modifier, Standardized by All Part B Services)',
                fontsize=22, fontweight='bold')
    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(filename, dpi=200, facecolor='white', format='svg')
    plt.show()
    print(f"Saved: {filename}")


# MAIN

def main():
    print("MNPB VOLUME ANALYSIS — ENT Procedures")
    
    ent_codes = load_ent_codes('mnpb/ENT_CPT_CODES.csv')
    df_all = load_mnpb('mnpb/MNPB_MASTER_FINAL.csv')
    
    yearly = calculate_volumes(df_all, ent_codes)
    
    plot_mnpb_volumes(yearly, TARGET_CPTS, REVAL_BREAKPOINTS, REVAL_DIRECTIONS)

if __name__ == "__main__":
    main()