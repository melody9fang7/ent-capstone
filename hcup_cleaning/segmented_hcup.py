import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from filtering import standardize_cpt
import warnings
warnings.filterwarnings('ignore')
 
YEAR_START = 2008
YEAR_END = 2017

def load_optime_data(filepath):
    df = pd.read_csv(filepath, low_memory=False)
    print(f"Loaded operative time data: {len(df):,} rows")

    df["ORTIME"] = pd.to_numeric(df["ORTIME"], errors="coerce")
    df["AYEAR"] = pd.to_numeric(df["AYEAR"], errors="coerce")
    df["CPT1"] = standardize_cpt(df["CPT1"])
    df = df.dropna(subset=["ORTIME", "AYEAR"])
    df = df[df["ORTIME"] > 0]
    df = df[(df["AYEAR"] >= YEAR_START) & (df["AYEAR"] <= YEAR_END)]

    return df
 
def load_revaluation_map(filepath):
    reval = pd.read_csv(filepath)
    reval["CPT Code"] = standardize_cpt(reval["CPT Code"])
    reval["Most Recent RUC Review"] = reval["Most Recent RUC Review"].astype(str).str[:4]
    reval["Most Recent RUC Review"] = pd.to_numeric(reval["Most Recent RUC Review"], errors="coerce")
    reval = reval.dropna(subset=["Most Recent RUC Review"])
    reval_map = {}

    for _, row in reval.iterrows():
        cpt = row["CPT Code"]
        year = int(row["Most Recent RUC Review"])
        reval_map[cpt] = [year]

    print(f"Loaded {len(reval_map)} CPT revaluation years")

    return reval_map
 
 
def fit_segmented(data, break_years, outcome_col):
    data = data.sort_values("YEAR").copy()
    X = data[["YEAR"]].copy()
    X["const"] = 1

    for by in break_years:
        data[f"TIME_SINCE_{by}"] = np.maximum(0, data["YEAR"] - by)
        X[f"TIME_SINCE_{by}"] = data[f"TIME_SINCE_{by}"]

    model = sm.OLS(data[outcome_col], X).fit()
    slopes = [model.params["YEAR"]]
    slope_changes = [model.params.get(f"TIME_SINCE_{by}", 0) for by in break_years]

    for sc in slope_changes:
        slopes.append(slopes[-1] + sc)

    return model, slopes, slope_changes
 
 
def evaluate_breakpoints(data, cpt, break_years, outcome_col, outcome_name):
    if not break_years or len(data) < 10:
        return None
    
    X_simple = sm.add_constant(data["YEAR"])
    simple_model = sm.OLS(data[outcome_col], X_simple).fit()

    try:
        seg_model, slopes, slope_changes = fit_segmented(data, break_years, outcome_col)
    except:
        return None
    
    rss_simple = np.sum(simple_model.resid ** 2)
    rss_seg = np.sum(seg_model.resid ** 2)
    rss_reduction_pct = (rss_simple - rss_seg) / rss_simple * 100 if rss_simple > 0 else 0
    df_diff = len(seg_model.params) - len(simple_model.params)

    if df_diff > 0:
        f_stat = ((rss_simple - rss_seg) / df_diff) / (rss_seg / (len(data) - len(seg_model.params)))
        f_pvalue = 1 - stats.f.cdf(f_stat, df_diff, len(data) - len(seg_model.params))
    else:
        f_pvalue = np.nan

    slope_pvalues = {}
    for by in break_years:
        slope_pvalues[by] = seg_model.pvalues.get(f"TIME_SINCE_{by}", 1.0)
    
    return {
        'CPT': cpt, 'Break_Years': break_years, 'Outcome': outcome_name,
        'n': len(data), 'R2_Simple': simple_model.rsquared, 'R2_Segmented': seg_model.rsquared,
        'RSS_Reduction_Pct': rss_reduction_pct, 'F_Pvalue': f_pvalue,
        'Breakpoints_Significant': f_pvalue < 0.05,
        'Pre_Slope': slopes[0], 'Segment_Slopes': slopes[1:],
        'Slope_Changes': dict(zip(break_years, slope_changes)),
        'Slope_Pvalues': slope_pvalues
    }
 
def get_optime_data(df, cpt):
    data = df[df["CPT1"] == cpt].copy()
    if len(data) < 10:
        return None
    return data[["AYEAR", "ORTIME"]].rename(columns={"AYEAR": "YEAR", "ORTIME": "VALUE"})

 
def plot_results(data_dict, results_df, reval_map, outcome_name, ylabel, filename):
    """Create segmented regression plots"""

    sig_cpts = results_df[results_df['Breakpoints_Significant'] == True]['CPT'].tolist()
    if not sig_cpts:
        sig_cpts = list(reval_map.keys())[:9]
    cpts_to_plot = sig_cpts
    n_plots = len(cpts_to_plot)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
 
    for idx, cpt in enumerate(cpts_to_plot):
        ax = axes[idx]
        data = data_dict.get(cpt)
        if data is None or len(data) < 6:
            ax.text(0.5, 0.5, f'CPT {cpt}\nInsufficient data', ha='center', va='center')
            ax.set_title(f'CPT {cpt}')
            continue

        break_years = reval_map.get(cpt, [])
        yearly_means = data.groupby('YEAR')['VALUE'].mean()

        try:
            model, slopes, _ = fit_segmented(data, break_years, 'VALUE')
            years_range = np.arange(data['YEAR'].min(), data['YEAR'].max() + 1)
            X_pred = pd.DataFrame({'YEAR': years_range})
            X_pred['const'] = 1

            for by in break_years:
                X_pred[f'TIME_SINCE_{by}'] = np.maximum(0, years_range - by)
            predictions = model.predict(X_pred)
            ax.plot(yearly_means.index, yearly_means.values, 'o-', color='steelblue',
                   alpha=0.7, markersize=4, linewidth=1.5, label='Observed')
            ax.plot(years_range, predictions, 'r-', linewidth=2, label='Segmented')

            for by in break_years:
                ax.axvline(x=by, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        except:
            ax.plot(yearly_means.index, yearly_means.values, 'o-', color='steelblue', alpha=0.7)
        row = results_df[results_df['CPT'] == cpt]
       
        if len(row) > 0:
            r2 = row.iloc[0]['R2_Segmented']
            f_p = row.iloc[0]['F_Pvalue']
            ax.text(0.98, 0.98, f'R²={r2:.3f}\np={f_p:.4f}', transform=ax.transAxes,
                   va='top', ha='right', fontsize=7, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.set_title(f'CPT {cpt} (n={len(data):,})')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
 
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Segmented Regression: {outcome_name}\n', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
 
def plot_specific_cpts(data_dict, results_df, reval_map, outcome_name, ylabel, filename, cpt_list):
    cpts_to_plot = [cpt for cpt in cpt_list if cpt in data_dict]

    if len(cpts_to_plot) == 0:
        print(f"None of the specified CPTs found in data: {cpt_list}")
        return
    
    n_plots = len(cpts_to_plot)
    n_cols = 2
    n_rows = (n_plots + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()
 
    for idx, cpt in enumerate(cpts_to_plot):
        ax = axes[idx]
        data = data_dict.get(cpt)

        if data is None or len(data) < 6:
            ax.text(0.5, 0.5, f'CPT {cpt}\nInsufficient data', ha='center', va='center')
            ax.set_title(f'CPT {cpt}')
            continue

        break_years = reval_map.get(cpt, [])
        yearly_means = data.groupby('YEAR')['VALUE'].mean()
        ax.plot(yearly_means.index, yearly_means.values, 'o-', color='steelblue',
               alpha=0.7, markersize=4, linewidth=1.5, label='Observed')
        
        try:
            model, slopes, _ = fit_segmented(data, break_years, 'VALUE')
            years_range = np.arange(data['YEAR'].min(), data['YEAR'].max() + 1)
            X_pred = pd.DataFrame({'YEAR': years_range})
            X_pred['const'] = 1
            for by in break_years:
                X_pred[f'TIME_SINCE_{by}'] = np.maximum(0, years_range - by)
            predictions = model.predict(X_pred)
            ax.plot(years_range, predictions, 'r-', linewidth=2, label='Segmented')
        except:
            pass

        for by in break_years:
            ax.axvline(x=by, color="black", linestyle='--', linewidth=1.5, alpha=0.7)

        row = results_df[results_df['CPT'] == cpt] if len(results_df) > 0 else None
        if row is not None and len(row) > 0:
            r2 = row.iloc[0]['R2_Segmented']
            f_p = row.iloc[0]['F_Pvalue']
            sig = '*' if row.iloc[0]['Breakpoints_Significant'] else ''
            ax.text(0.98, 0.98, f'F-test p={f_p:.4f}{sig}', transform=ax.transAxes,
                   va='top', ha='right', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.set_title(f'CPT {cpt} (n={len(data):,})')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
 
    for idx in range(len(cpts_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Segmented Regression: {outcome_name} (Selected CPTs)', fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight', format='png')
    plt.show()
    print(f"Saved: {filename}")
 
def main():
    print("HCUP SEGMENTED REGRESSION")
    df_optime = load_optime_data("HCUP_filtered_172_cleaned.csv")
    reval_map = load_revaluation_map("filtered_sina2.csv")
    optime_results = []
    optime_data_dict = {}

    for cpt, break_years in reval_map.items():
        if cpt not in df_optime["CPT1"].unique():
            continue
        data = get_optime_data(df_optime, cpt)
        if data is not None:
            optime_data_dict[cpt] = data
            result = evaluate_breakpoints(data, cpt, break_years, "VALUE", "Operative Time")
            if result:
                optime_results.append(result)
    
    optime_df = pd.DataFrame(optime_results)

    print(optime_df)

    optime_df.to_csv("optime_segmented_results_hcup.csv", index=False)
    plot_results(optime_data_dict, optime_df, reval_map, "Operative Time Response", "Operative Time (minutes)", "segmented_optime_hcup.svg")

    print("\nSaved:")
    print("optime_segmented_results_hcup.csv")
    print("segmented_optime_hcup.png")
    
    target_cpts = ['38542', '42415', '42420', '42440']
    plot_specific_cpts(optime_data_dict, optime_df, reval_map, "Operative Time Response", "Operative Time (minutes)", "segmented_optime_selected_cpts.svg", target_cpts)
 
 
if __name__ == "__main__":
    main()
 