import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
from patsy import bs

def standardize_cpt(series: pd.Series) -> pd.Series:
    def _to_str(x):
        if pd.isna(x):
            return None
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        return str(x).strip()
 
    return series.apply(_to_str)


def prep_hcup(hcup_file, reval_file):
    # load hcup
    df = pd.read_csv(hcup_file, low_memory=False)
    reval = pd.read_csv(reval_file)

    # clean hcup columns
    df["CPT1"] = standardize_cpt(df["CPT1"])
    df["ORTIME"] = pd.to_numeric(df["ORTIME"], errors="coerce")
    df["AYEAR"] = pd.to_numeric(df["AYEAR"], errors="coerce")
    df = df[df["ORTIME"] > 0]
    #df = df[df["ORTIME"] >= 10].copy()  # minimum 10 minutes


    reval["CPT Code"] = standardize_cpt(reval["CPT Code"])
    reval["Most Recent RUC Review"] = (reval["Most Recent RUC Review"].astype(str).str[:4])
    reval["Most Recent RUC Review"] = pd.to_numeric(reval["Most Recent RUC Review"], errors="coerce")

    # for better analysis
    reval = reval[(reval["Most Recent RUC Review"] >= 2010) &(reval["Most Recent RUC Review"] <= 2015)]

    # merge reval year onto hcup
    df = df.merge(reval[["CPT Code", "Most Recent RUC Review"]], left_on="CPT1",
                  right_on="CPT Code", how="left")
    
    df = df[df["CPT1"] != "69220"].copy() # median time is around 1 min, drop it.

    # 0 if procedure was before reval year, 1 if it was after (or same)
    df["POST_REVAL"] = (df["AYEAR"] >= df["Most Recent RUC Review"] ).astype(int)

    # only cpt codes that have the reval year in correct range
    valid_cpts = df.groupby("CPT1")["POST_REVAL"].nunique()
    valid_cpts = valid_cpts[valid_cpts == 2].index
    df = df[df["CPT1"].isin(valid_cpts)].copy()

    # eligible cpts, counts before and after reval year
    counts = df.groupby(["CPT1", "POST_REVAL"]).size()

    eligible = (counts[counts >= 50].reset_index().groupby("CPT1").size())

    eligible = eligible[eligible == 2].index

    df = df[df["CPT1"].isin(eligible)].copy()
    # center year
    df["YEAR_CENTERED"] = df["AYEAR"] - df["Most Recent RUC Review"]

    return df


def run_simple_mixed_model(df):
    model = smf.mixedlm(
        "np.log(ORTIME) ~ YEAR_CENTERED + POST_REVAL + YEAR_CENTERED:POST_REVAL",
        data = df,
        groups = df["CPT1"]
    )

    result = model.fit(method="lbfgs", maxiter=1000)
    print(result.summary())

    return result

def predict_simple(result, pred_df):
    pred = result.predict(pred_df).copy()
    preds = []

    for i in range(len(pred_df)):
        cpt = pred_df.iloc[i]["CPT1"] 
        re = result.random_effects[cpt]
        rand = re["Group"] 

        preds.append(pred.iloc[i] + rand)

    return np.exp(np.array(preds))

def run_advanced_mixed_model(df):
    model = smf.mixedlm(
        "np.log(ORTIME) ~ bs(YEAR_CENTERED, df=3) * POST_REVAL",
        data = df,
        groups = df["CPT1"],
    )

    result = model.fit(method = "lbfgs", maxiter = 1000)
    print(result.summary())

    return result

def predict_advanced(result, pred_df):
    pred = result.predict(pred_df).copy()

    preds = []

    for i in range(len(pred_df)):
        cpt = pred_df.iloc[i]["CPT1"]
        re = result.random_effects[cpt]
        rand = re["Group"]

        preds.append(pred.iloc[i] + rand)

    return np.exp(np.array(preds))


def plot_mixed_model(df, result, type = "simple"):
    cpts = sorted(df["CPT1"].unique())
    n = len(cpts)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()


    for ax, cpt in zip(axes, cpts):
        sub = df[df["CPT1"] == cpt].copy() # sub data for this cpt code

        yearly = (sub.groupby("AYEAR")["ORTIME"].mean().reset_index()) # yearly mean ORTIME

        reval_year = int(sub["Most Recent RUC Review"].iloc[0]) # reval year for this cpt code

        years = np.arange(2008, 2018) # range of years to predict for (2008-2017)

        pred_df = pd.DataFrame({"AYEAR": years, "YEAR_CENTERED": years - reval_year,
                                "POST_REVAL": (years >= reval_year).astype(int), "CPT1": cpt}) # dataframe for prediction line

        if type == "simple":
            pred_df["PRED_ORTIME"] = predict_simple(result, pred_df)
        else:
            pred_df["PRED_ORTIME"] = predict_advanced(result, pred_df)

        # plot raw data, yearly means, prediction line, and reval year
        ax.scatter(sub["AYEAR"], sub["ORTIME"], color = "gray", alpha = 0.3, s = 10, label = "Raw Observations")
        ax.scatter(yearly["AYEAR"], yearly["ORTIME"], color = "black", alpha = 0.8, s = 45, label = "Yearly Mean")
        ax.plot(pred_df["AYEAR"], pred_df["PRED_ORTIME"], color = "crimson", linewidth = 2.5, label = "Prediction")
        ax.axvline(reval_year, color = "black", linestyle = "--", linewidth = 1.5, alpha = 0.8, label = "Reval Year")

        ax.set_title(f"CPT {cpt}", fontsize = 13)
        ax.set_ylabel("Operative Time (Minutes)", fontsize = 11)
        ax.set_xlabel("Year", fontsize = 11)

        ax.grid(True, alpha = 0.25)
        lower = sub["ORTIME"].quantile(0.01)
        upper = sub["ORTIME"].quantile(0.90)
        ax.set_ylim(lower - 1, upper * 2)

        ax.set_xticks([2008, 2010, 2012, 2014, 2016, 2017])
    
    for i in range(len(cpts), len(axes)):
        axes[i].set_visible(False)
    
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc = "upper center", ncol = 4, fontsize = 10, frameon = True, bbox_to_anchor = (0.5, 0.94))

    fig.suptitle("Mixed Effects Model for Operative Time by CPT", fontsize=18)
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.88)
    if type == "simple":
        plt.savefig("mixed_effects_model_simple.png", dpi = 300, bbox_inches = "tight")
    else:
        plt.savefig("mixed_effects_model_advanced.png", dpi = 300, bbox_inches = "tight")
    plt.show()


def qq_plot(result):
    import statsmodels.api as sm
    sm.qqplot(result.resid, line="45")
    plt.title("QQ Plot of Residuals")
    plt.show()

def main():
    hcup_file = "HCUP_filtered_172_cleaned.csv" # have cleaned dataset (this already had solo cases)
    reval_file = "filtered_sina2.csv" # cpt list

    df = prep_hcup(hcup_file, reval_file)

    result_advanced = run_advanced_mixed_model(df)
    result_simple = run_simple_mixed_model(df)

    plot_mixed_model(df, result_simple, type = "simple")
    plot_mixed_model(df, result_advanced, type = "advanced")

    qq_plot(result_simple)
    qq_plot(result_advanced)


if __name__ == "__main__":
    main()
