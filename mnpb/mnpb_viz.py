import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# these are from NSQIP I just didnt want to keep reloading the whole file

REVAL_BREAKPOINTS = {
    '15731': [2010], '21034': [2010], '21044': [2010], '21045': [2007, 2010], '21199': [2010], '21395': [2010],
    '21556': [2007], '21557': [2007], '21685': [2010], '31254': [2018], '31255': [2018], '31256': [2018], 
    '31267': [2018], '31287': [2018], '31288': [2018], '31360': [2007, 2010], '31365': [2007, 2010], '31367': [2010], 
    '31368': [2009, 2010], '31370': [2007, 2010], '31375': [2010], '31380': [2011], '31382': [2010], '31390': [2010], 
    '31395': [2010], '31580': [2011, 2017], '31587': [2010], '31610': [2018], '35701': [2007, 2010], '35800': [2007, 2010, 2012], 
    '37565': [2007, 2010], '37600': [2007, 2012], '38305': [2007, 2010], '38542': [2007, 2009, 2010], '38700': [2007, 2010], '38720': [2007, 2010], 
    '38724': [2007, 2010], '40810': [2007, 2010], '40816': [2007, 2010], '40819': [2010], '41120': [2007, 2010], '41130': [2007, 2010], 
    '41135': [2007, 2010], '41140': [2010], '41145': [2010], '41150': [2007, 2010], '41153': [2007, 2010], '41155': [2007, 2010], 
    '42120': [2007, 2010], '42145': [2007, 2010], '42415': [2007, 2010, 2012], '42420': [2007, 2010, 2012], '42440': [2007, 2010, 2012], 
    '42842': [2010], '42844': [2010], '42845': [2007, 2010], '42890': [2007, 2010], '42892': [2010], '42894': [2010], '43116': [2010], 
    '43410': [2007, 2010], '43420': [2007, 2010], '60220': [2007, 2010, 2012], '60240': [2007, 2010, 2012], '60252': [2007, 2010], '60254': [2007, 2010], 
    '60260': [2007, 2010], '60270': [2007, 2010], '60271': [2007, 2010], '69801': [2012]
}

REVAL_DIRECTIONS = {
    '15731': {2010: 'increase'}, '21034': {2010: 'increase'}, '21044': {2010: 'increase'}, '21045': {2007: 'increase', 2010: 'increase'}, 
    '21199': {2010: 'increase'}, '21395': {2010: 'increase'}, '21556': {2007: 'increase'}, '21557': {2007: 'increase'}, '21685': {2010: 'increase'}, 
    '31254': {2018: 'decrease'}, '31255': {2018: 'decrease'}, '31256': {2018: 'decrease'}, '31267': {2018: 'decrease'}, '31287': {2018: 'decrease'}, 
    '31288': {2018: 'decrease'}, '31360': {2007: 'increase', 2010: 'increase'}, '31365': {2007: 'increase', 2010: 'increase'}, 
    '31367': {2010: 'increase'}, '31368': {2009: 'increase', 2010: 'increase'}, '31370': {2007: 'increase', 2010: 'increase'}, 
    '31375': {2010: 'increase'}, '31380': {2011: 'increase'}, '31382': {2010: 'increase'}, '31390': {2010: 'increase'}, '31395': {2010: 'increase'}, 
    '31580': {2011: 'increase', 2017: 'decrease'}, '31587': {2010: 'increase'}, '31610': {2018: 'increase'}, '35701': {2007: 'increase', 2010: 'increase'}, 
    '35800': {2007: 'increase', 2010: 'increase', 2012: 'increase'}, '37565': {2007: 'increase', 2010: 'increase'}, 
    '37600': {2007: 'increase', 2012: 'increase'}, '38305': {2007: 'increase', 2010: 'increase'}, 
    '38542': {2007: 'increase', 2009: 'increase', 2010: 'increase'}, '38700': {2007: 'increase', 2010: 'increase'}, 
    '38720': {2007: 'increase', 2010: 'increase'}, '38724': {2007: 'increase', 2010: 'increase'}, '40810': {2007: 'increase', 2010: 'increase'}, 
    '40816': {2007: 'increase', 2010: 'increase'}, '40819': {2010: 'increase'}, '41120': {2007: 'increase', 2010: 'increase'}, 
    '41130': {2007: 'increase', 2010: 'increase'}, '41135': {2007: 'increase', 2010: 'increase'}, '41140': {2010: 'increase'}, 
    '41145': {2010: 'increase'}, '41150': {2007: 'increase', 2010: 'increase'}, '41153': {2007: 'increase', 2010: 'increase'}, 
    '41155': {2007: 'increase', 2010: 'increase'}, '42120': {2007: 'increase', 2010: 'increase'}, '42145': {2007: 'increase', 2010: 'increase'}, 
    '42415': {2007: 'increase', 2010: 'increase', 2012: 'decrease'}, '42420': {2007: 'increase', 2010: 'increase', 2012: 'decrease'}, 
    '42440': {2007: 'increase', 2010: 'increase', 2012: 'decrease'}, '42842': {2010: 'increase'}, '42844': {2010: 'increase'}, 
    '42845': {2007: 'increase', 2010: 'increase'}, '42890': {2007: 'increase', 2010: 'increase'}, '42892': {2010: 'increase'}, 
    '42894': {2010: 'increase'}, '43116': {2010: 'increase'}, '43410': {2007: 'increase', 2010: 'increase'}, '43420': {2007: 'increase', 2010: 'increase'}, 
    '60220': {2007: 'increase', 2010: 'increase', 2012: 'decrease'}, '60240': {2007: 'increase', 2010: 'increase', 2012: 'decrease'}, 
    '60252': {2007: 'increase', 2010: 'increase'}, '60254': {2007: 'increase', 2010: 'increase'}, '60260': {2007: 'increase', 2010: 'increase'}, 
    '60270': {2007: 'increase', 2010: 'increase'}, '60271': {2007: 'increase', 2010: 'increase'}, '69801': {2012: 'decrease'}
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

def plot_payment_vs_volume_deval(yearly, cpt_list, reval_map, direction_map,
                                   filename='deval_payment_vs_volume.svg'):
    """
    Dual-axis: Volume and average payment for devaluation CPTs.
    """
    cpts_to_plot = [c for c in cpt_list if c in yearly['HCPCS'].values]
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    axes = axes.flatten()
    
    for idx, cpt in enumerate(cpts_to_plot):
        ax1 = axes[idx]
        cpt_data = yearly[yearly['HCPCS'] == cpt].dropna(subset=['avg_payment', 'volume_pct']).sort_values('YEAR')
        break_years = reval_map.get(cpt, [])
        
        if len(cpt_data) < 3:
            continue
        
        # Volume (left axis)
        ax1.plot(cpt_data['YEAR'], cpt_data['volume_pct'], 'o-', 
                color='steelblue', linewidth=3, markersize=10,
                markerfacecolor='white', markeredgewidth=2, label='Volume')
        ax1.set_ylabel('% of Part B Services', fontsize=14, fontweight='bold', color='steelblue')
        ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=12)
        
        # Payment (right axis)
        ax2 = ax1.twinx()
        ax2.plot(cpt_data['YEAR'], cpt_data['avg_payment'], 's--', 
                color='#c0392b', linewidth=3, markersize=10,
                markerfacecolor='white', markeredgewidth=2, label='Avg Payment')
        ax2.set_ylabel('Average Payment ($)', fontsize=14, fontweight='bold', color='#c0392b')
        ax2.tick_params(axis='y', labelcolor='#c0392b', labelsize=12)
        
        # Breakpoints
        for by in break_years:
            direction = direction_map.get(cpt, {}).get(by, None)
            color = 'green' if direction == 'increase' else 'red'
            ax1.axvline(x=by, color=color, linestyle='--', linewidth=2, alpha=0.7)
        
        # Compute correlation between payment and volume
        corr = cpt_data['volume_pct'].corr(cpt_data['avg_payment'])
        
        ax1.set_title(f'CPT {cpt}\nPayment-Volume r = {corr:+.3f}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=12)
        
        # Integer x-axis
        x_min, x_max = int(cpt_data['YEAR'].min()), int(cpt_data['YEAR'].max())
        ax1.set_xticks(range(x_min, x_max + 1, max(1, (x_max-x_min)//4)))
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
    
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Volume vs Payment — Devaluation CPTs\n(Blue = Volume, Red = Payment, Green/Red lines = wRVU breakpoints)',
                fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=250, bbox_inches='tight', facecolor='white', format='svg')
    plt.show()
    print(f"Saved: {filename}")

def cross_correlation_deval(yearly, cpt, max_lag=4):
    """Cross-correlation between payment changes and volume changes."""
    cpt_data = yearly[yearly['HCPCS'] == cpt].sort_values('YEAR')
    
    pay_diff = cpt_data['avg_payment'].diff().dropna().values
    vol_diff = cpt_data['volume_pct'].diff().dropna().values
    
    n = min(len(pay_diff), len(vol_diff))
    pay_diff, vol_diff = pay_diff[:n], vol_diff[:n]
    
    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(vol_diff[-lag:], pay_diff[:lag])[0, 1]
        elif lag == 0:
            corr = np.corrcoef(vol_diff, pay_diff)[0, 1]
        else:
            corr = np.corrcoef(vol_diff[:-lag], pay_diff[lag:])[0, 1]
        correlations[lag] = corr
    
    return correlations

def devaluation_summary_table(yearly, deval_cpts, reval_map, magnitude_map):
    """Summary table for devaluation CPTs."""
    rows = []
    for cpt in deval_cpts:
        cpt_data = yearly[yearly['HCPCS'] == cpt].sort_values('YEAR')
        break_years = reval_map.get(cpt, [])
        
        # Find the decrease year
        dec_year = None
        for by in break_years:
            if direction_map.get(cpt, {}).get(by) == 'decrease':
                dec_year = by
                break
        
        if dec_year is None:
            continue
        
        pre = cpt_data[cpt_data['YEAR'] < dec_year]
        post = cpt_data[cpt_data['YEAR'] >= dec_year]
        
        pre_vol = pre['volume_pct'].mean()
        post_vol = post['volume_pct'].mean()
        pre_pay = pre['avg_payment'].mean()
        post_pay = post['avg_payment'].mean()
        
        rows.append({
            'CPT': cpt,
            'Deval_Year': dec_year,
            'Magnitude_%': magnitude_map.get(cpt, {}).get(dec_year, 0),
            'Pre_Vol_%': round(pre_vol, 4),
            'Post_Vol_%': round(post_vol, 4),
            'Vol_Change_pp': round(post_vol - pre_vol, 4),
            'Pre_Pay_$': round(pre_pay, 0),
            'Post_Pay_$': round(post_pay, 0),
            'Pay_Change_%': round((post_pay - pre_pay) / pre_pay * 100, 1),
        })
    
    return pd.DataFrame(rows)

def main():
    print("MNPB VOLUME ANALYSIS — ENT Procedures")
    
    ent_codes = load_ent_codes('mnpb/ENT_CPT_CODES.csv')
    df_all = load_mnpb('mnpb/MNPB_MASTER_FINAL.csv')
    
    yearly = calculate_volumes(df_all, ent_codes)
    
    plot_mnpb_volumes(yearly, TARGET_CPTS, REVAL_BREAKPOINTS, REVAL_DIRECTIONS)

    deval_cpts = ['60240', '60220', '42440', '42420', '69801']

if __name__ == "__main__":
    main()