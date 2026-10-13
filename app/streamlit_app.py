"""
23CSE301 Machine Learning Capstone - interactive app (bonus GUI).

Run locally:   streamlit run app/streamlit_app.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as pl  # noqa: E402

PALETTE = px.colors.qualitative.Safe  # colour-blind friendly
st.set_page_config(page_title="ML Capstone - FC24 & Hotel Bookings", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="First launch: training the models (about 2 minutes)...")
def load_models():
    if not (pl.CLF_PATH.exists() and pl.REG_PATH.exists()):
        pl.train_all()
    return joblib.load(pl.CLF_PATH), joblib.load(pl.REG_PATH)


@st.cache_data
def load_players():
    return pl.load_players()


@st.cache_data
def load_meta():
    load_models()
    return json.loads(pl.META_PATH.read_text()) if pl.META_PATH.exists() else {}


@st.cache_resource
def cluster_base():
    df, X, scaler, X_all, idx = pl.clustering_data()
    Xs = X_all[idx]
    pca = PCA(n_components=2, random_state=pl.RANDOM_STATE)
    Xp = pca.fit_transform(Xs)
    return {"df": df, "X_cols": X.columns.tolist(), "scaler": scaler, "Xs": Xs, "Xp": Xp,
            "var": pca.explained_variance_ratio_, "sample": df.iloc[idx].reset_index(drop=True)}


def _metrics(Xs, labels):
    return {"Silhouette": silhouette_score(Xs, labels),
            "Davies-Bouldin": davies_bouldin_score(Xs, labels),
            "Calinski-Harabasz": calinski_harabasz_score(Xs, labels)}


@st.cache_resource(show_spinner="Fitting clustering model...")
def fit_clusters(algo: str, k: int):
    Xs = cluster_base()["Xs"]
    if algo == "K-Means":
        model = KMeans(n_clusters=k, n_init=10, random_state=pl.RANDOM_STATE).fit(Xs)
        labels = model.labels_
    else:
        model = None
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xs)
    return model, labels, _metrics(Xs, labels)


@st.cache_data(show_spinner="Computing elbow curve...")
def elbow_curve():
    Xs = cluster_base()["Xs"]
    rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=pl.RANDOM_STATE).fit(Xs)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(Xs, km.labels_)})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Drawing dendrogram...")
def dendrogram_figure():
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    Z = linkage(cluster_base()["Xs"][:800], method="ward")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax)
    ax.set(title="Dendrogram (Ward linkage, 800-point sample)",
           xlabel="Cluster size / sample index", ylabel="Distance")
    fig.tight_layout()
    return fig


def fmt_eur(v: float) -> str:
    return f"€{v / 1e6:,.1f}M" if v >= 1e6 else f"€{v / 1e3:,.0f}K"


def idx_of(options, value, default=0):
    return list(options).index(value) if value in options else default


# --------------------------------------------------------------------------- #
# Page: Home
# --------------------------------------------------------------------------- #
def page_home():
    st.title("📊 Machine Learning Capstone")
    st.caption("23CSE301 Machine Learning · Regression · Classification · Clustering")
    meta = load_meta()
    _, _, km_metrics = fit_clusters("K-Means", 4)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Regression")
        st.write("Predict a football player's **market value** from FC 24 attributes. "
                 "Final model: tuned Gradient Boosting.")
        if "regressor" in meta:
            st.metric("Test R² (log value)", f"{meta['regressor']['R2']:.4f}")
            st.caption(f"RMSE {meta['regressor']['RMSE']:.3f} · MAE {meta['regressor']['MAE']:.3f}")
    with c2:
        st.subheader("Classification")
        st.write("Predict whether a **hotel booking will be cancelled**. Final model: Random Forest.")
        if "classifier" in meta:
            st.metric("Test ROC-AUC", f"{meta['classifier']['ROC-AUC']:.3f}")
            st.caption(f"Accuracy {meta['classifier']['Accuracy']:.3f} · "
                       f"Recall {meta['classifier']['Recall']:.3f}")
    with c3:
        st.subheader("Clustering")
        st.write("Group FC 24 players by their attributes **without using position labels**. "
                 "K-Means vs Agglomerative.")
        st.metric("K-Means (k=4) silhouette", f"{km_metrics['Silhouette']:.3f}")
        st.caption(f"Davies-Bouldin {km_metrics['Davies-Bouldin']:.2f} · "
                   f"Calinski-Harabasz {km_metrics['Calinski-Harabasz']:,.0f}")

    st.info("Use the sidebar to open each track's interactive page. "
            "The classifier shown here is a lighter Random Forest (100 trees) than the notebook's "
            "200-tree model so the app stays small enough to deploy; ROC-AUC is 0.911 vs 0.912.")

    for label, fname in [("Regression results (10 models)", "regression_results.csv"),
                         ("Classification results (10 models)", "classification_results.csv"),
                         ("Clustering results", "clustering_results.csv")]:
        path = pl.MODEL_DIR / fname
        if path.exists():
            with st.expander(label):
                st.dataframe(pd.read_csv(path, index_col=0).round(4), width="stretch")


# --------------------------------------------------------------------------- #
# Page: Classification - hotel cancellation
# --------------------------------------------------------------------------- #
def page_hotel():
    clf = load_models()[0]
    opt = clf["options"]
    st.title("🏨 Hotel Booking Cancellation Predictor")
    st.write("Enter the details of a booking and the Random Forest estimates the chance it gets cancelled.")

    with st.form("hotel_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Stay**")
            hotel_type = st.selectbox("Hotel type", opt["hotel_type"])
            city = st.selectbox("City", opt["city"], index=idx_of(opt["city"], "Chennai"))
            arrival = st.date_input("Arrival date", value=date(2024, 7, 15))
            lead_time = st.slider("Lead time (days from booking to arrival)", 0, 737, 100)
            weekend = st.number_input("Weekend nights", 0, 20, 1)
            week = st.number_input("Week nights", 0, 50, 2)
            adr = st.number_input("Average daily rate (ADR)", 0.0, 1000.0, 100.0, step=5.0)
        with c2:
            st.markdown("**Guest**")
            adults = st.number_input("Adults", 1, 10, 2)
            children = st.number_input("Children", 0, 10, 0)
            babies = st.number_input("Babies", 0, 5, 0)
            customer_type = st.selectbox("Customer type", opt["customer_type"],
                                         index=idx_of(opt["customer_type"], "Transient"))
            repeated = st.checkbox("Repeated guest")
            prev_canc = st.number_input("Previous cancellations", 0, 30, 0)
            prev_ok = st.number_input("Previous bookings not cancelled", 0, 80, 0)
        with c3:
            st.markdown("**Booking**")
            market = st.selectbox("Market segment", opt["market_segment"],
                                  index=idx_of(opt["market_segment"], "Online TA"))
            channel = st.selectbox("Distribution channel", opt["distribution_channel"],
                                   index=idx_of(opt["distribution_channel"], "TA/TO"))
            deposit = st.selectbox("Deposit type", opt["deposit_type"],
                                   index=idx_of(opt["deposit_type"], "No Deposit"))
            meal = st.selectbox("Meal plan", opt["meal"], index=idx_of(opt["meal"], "BB"))
            reserved = st.selectbox("Reserved room type", opt["reserved_room_type"],
                                    index=idx_of(opt["reserved_room_type"], "A"))
            assigned = st.selectbox("Assigned room type", opt["assigned_room_type"],
                                    index=idx_of(opt["assigned_room_type"], "A"))
            changes = st.number_input("Booking changes", 0, 25, 0)
            waiting = st.number_input("Days on waiting list", 0, 400, 0)
            parking = st.number_input("Parking spaces required", 0, 8, 0)
            requests = st.number_input("Special requests", 0, 5, 0)
        submitted = st.form_submit_button("Predict cancellation risk", type="primary")

    if submitted:
        row = {
            "lead_time": lead_time,
            "arrival_date_month": arrival.strftime("%B"),
            "arrival_date_week_number": arrival.isocalendar()[1],
            "arrival_date_day_of_month": arrival.day,
            "stays_in_weekend_nights": weekend, "stays_in_week_nights": week,
            "adults": adults, "children": float(children), "babies": babies,
            "meal": meal, "market_segment": market, "distribution_channel": channel,
            "is_repeated_guest": int(repeated), "previous_cancellations": prev_canc,
            "previous_bookings_not_canceled": prev_ok,
            "reserved_room_type": reserved, "assigned_room_type": assigned,
            "booking_changes": changes, "deposit_type": deposit,
            "days_in_waiting_list": waiting, "customer_type": customer_type, "adr": adr,
            "required_car_parking_spaces": parking, "total_of_special_requests": requests,
            "city": city, "hotel_type": hotel_type,
            "total_nights": weekend + week, "total_guests": adults + children + babies,
        }
        X = pd.DataFrame([row])[clf["columns"]]
        proba = float(clf["model"].predict_proba(clf["preprocessor"].transform(X))[0, 1])

        st.divider()
        left, right = st.columns([1, 2])
        with left:
            st.metric("Cancellation probability", f"{proba:.0%}")
            if proba >= 0.6:
                st.error("High risk")
            elif proba >= 0.3:
                st.warning("Medium risk")
            else:
                st.success("Low risk")
        with right:
            st.progress(proba)
            if proba >= 0.6:
                st.write("Consider asking for a deposit, sending a reminder, or planning to resell the room.")
            elif proba >= 0.3:
                st.write("Worth a reminder or confirmation message closer to arrival.")
            else:
                st.write("This booking looks likely to go ahead.")
        if deposit == "Non Refund":
            st.caption("Note: in this dataset 'Non Refund' bookings are almost always cancelled "
                       "(a data quirk discussed in the notebook), so this choice pushes the risk up sharply.")

    with st.expander("What drives the model? (top 12 feature importances)"):
        imp = (pd.Series(clf["model"].feature_importances_, index=clf["feature_names"])
               .sort_values(ascending=False).head(12).iloc[::-1])
        fig = px.bar(imp, orientation="h", color_discrete_sequence=PALETTE,
                     labels={"value": "Importance", "index": "Feature"},
                     title="Random Forest - Top 12 Feature Importances")
        fig.update_layout(showlegend=False, yaxis_title="Feature")
        st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Page: Regression - player value
# --------------------------------------------------------------------------- #
def page_value():
    bundle = load_models()[1]
    players = load_players()
    st.title("⚽ Player Market Value Estimator")
    st.write("Pick a player as a starting point, change their attributes, and see how the "
             "tuned Gradient Boosting model re-values them. All other attributes stay as in the dataset.")

    q = st.text_input("Search player by name (leave empty for the top-rated players)")
    pool = players[players["short_name"].str.contains(q, case=False, na=False, regex=False)] if q else players
    pool = pool.sort_values("overall", ascending=False).head(50)
    if pool.empty:
        st.warning("No player found with that name.")
        return
    labels = {i: f"{r.short_name} - OVR {r.overall}, {r.player_positions}, age {r.age}"
              for i, r in pool.iterrows()}
    sel = st.selectbox("Player", list(labels), format_func=labels.get)
    base = players.loc[sel]
    k = f"p{sel}"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Profile**")
        overall = st.slider("Overall", 40, 95, int(base.overall), key=f"ovr{k}")
        potential = st.slider("Potential", 40, 95, int(base.potential), key=f"pot{k}")
        age = st.slider("Age", 16, 45, int(base.age), key=f"age{k}")
        wage = st.number_input("Weekly wage (EUR)", 0, 700_000, int(base.wage_eur), step=1000, key=f"wage{k}")
        rep = st.slider("International reputation", 1, 5, int(base.international_reputation), key=f"rep{k}")
        skill = st.slider("Skill moves", 1, 5, int(base.skill_moves), key=f"sk{k}")
        weak = st.slider("Weak foot", 1, 5, int(base.weak_foot), key=f"wf{k}")
    with c2:
        st.markdown("**Core attributes**")
        vals = {a: st.slider(a.capitalize(), 0, 99, int(base[a]), key=f"{a}{k}")
                for a in ["pace", "shooting", "passing", "dribbling", "defending", "physic"]}
    with c3:
        st.markdown("**Role**")
        groups = sorted(players["position_group"].unique())
        pos = st.selectbox("Position group", groups, index=idx_of(groups, base.position_group), key=f"pos{k}")
        foot = st.selectbox("Preferred foot", ["Right", "Left"], index=idx_of(["Right", "Left"], base.preferred_foot), key=f"ft{k}")
        wr_att, wr_def = base.work_rate.split("/")
        levels = ["High", "Medium", "Low"]
        att = st.selectbox("Attacking work rate", levels, index=idx_of(levels, wr_att), key=f"wa{k}")
        dfn = st.selectbox("Defensive work rate", levels, index=idx_of(levels, wr_def), key=f"wd{k}")

    edited = base.copy()
    edited["overall"], edited["potential"], edited["age"] = overall, potential, age
    edited["wage_eur"], edited["international_reputation"] = wage, rep
    edited["skill_moves"], edited["weak_foot"] = skill, weak
    for a, v in vals.items():
        edited[a] = v
    edited["position_group"], edited["preferred_foot"] = pos, foot
    edited["work_rate"] = f"{att}/{dfn}"

    original = pl.predict_player_values(bundle, base.to_frame().T.infer_objects())[0]
    estimate = pl.predict_player_values(bundle, edited.to_frame().T.infer_objects())[0]

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Value in dataset", fmt_eur(base.value_eur))
    m2.metric("Model estimate (unedited player)", fmt_eur(original))
    m3.metric("Model estimate (with your edits)", fmt_eur(estimate),
              delta=f"{(estimate - original) / original:+.0%} vs unedited")
    st.caption("The model predicts log(value), so it is most accurate for mid-range players "
               "(median error about 3% on the whole dataset) and tends to under-estimate a few superstars.")

    sweep = pd.DataFrame([edited] * 56).reset_index(drop=True)
    sweep["overall"] = np.arange(40, 96)
    curve = pd.DataFrame({"Overall rating": sweep["overall"],
                          "Estimated value (EUR millions)": pl.predict_player_values(bundle, sweep.infer_objects()) / 1e6})
    fig = px.line(curve, x="Overall rating", y="Estimated value (EUR millions)",
                  color_discrete_sequence=PALETTE, title="Estimated value vs overall rating (other inputs as above)")
    fig.add_vline(x=overall, line_dash="dash", annotation_text="current")
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Page: Clustering
# --------------------------------------------------------------------------- #
def page_clusters():
    base = cluster_base()
    st.title("🧩 Player Clusters")
    st.write("Unsupervised grouping of 5,000 FC 24 players (49 standardised attributes). "
             "Position labels are **not** used for fitting - they only appear in the validation view.")

    c1, c2 = st.columns([1, 2])
    algo = c1.radio("Algorithm", ["K-Means", "Agglomerative (Ward)"])
    k = c2.slider("Number of clusters (k)", 2, 8, 4)
    algo_key = "K-Means" if algo == "K-Means" else "Agglomerative"
    model, labels, met = fit_clusters(algo_key, k)

    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette (higher is better)", f"{met['Silhouette']:.3f}")
    m2.metric("Davies-Bouldin (lower is better)", f"{met['Davies-Bouldin']:.3f}")
    m3.metric("Calinski-Harabasz (higher is better)", f"{met['Calinski-Harabasz']:,.0f}")

    sample = base["sample"]
    tab_pca, tab_prof, tab_sel, tab_assign = st.tabs(
        ["PCA view", "Cluster profiles", "Choosing k", "Where does a player fall?"])

    with tab_pca:
        color_by = st.radio("Colour points by", ["Cluster", "True position (validation only)"], horizontal=True)
        plot = pd.DataFrame({"PC1": base["Xp"][:, 0], "PC2": base["Xp"][:, 1],
                             "Cluster": labels.astype(str), "Position": sample["position_group"],
                             "Player": sample["short_name"]})
        col = "Cluster" if color_by == "Cluster" else "Position"
        fig = px.scatter(plot, x="PC1", y="PC2", color=col, hover_data=["Player", "Position"],
                         color_discrete_sequence=PALETTE, opacity=0.7,
                         category_orders={"Cluster": [str(i) for i in range(k)]},
                         title=f"{algo} clusters (k={k}) on PCA components")
        fig.update_layout(xaxis_title=f"PC1 ({base['var'][0]:.1%} of variance)",
                          yaxis_title=f"PC2 ({base['var'][1]:.1%} of variance)", legend_title=col)
        fig.update_traces(marker_size=5)
        st.plotly_chart(fig, width="stretch")

    with tab_prof:
        tmp = sample.assign(Cluster=labels)
        rows = []
        for c in range(k):
            sub = tmp[tmp.Cluster == c]
            top = sub["position_group"].value_counts()
            rows.append({"Cluster": c, "Players": len(sub),
                         "Most common position": f"{top.index[0]} ({top.iloc[0] / len(sub):.0%})"})
        st.markdown("**Cluster size and dominant position** (labels used only to interpret, after fitting)")
        st.dataframe(pd.DataFrame(rows).set_index("Cluster"), width="stretch")
        attrs = ["overall", "pace", "shooting", "passing", "dribbling", "defending", "physic", "goalkeeping_reflexes"]
        prof = tmp.groupby("Cluster")[attrs].mean().round(1)
        st.markdown("**Average attributes per cluster**")
        st.dataframe(prof.style.background_gradient(cmap="Blues", axis=0), width="stretch")
        ct = pd.crosstab(tmp["Cluster"], tmp["position_group"])
        st.markdown("**Cluster vs true position (counts)**")
        st.dataframe(ct, width="stretch")

    with tab_sel:
        el = elbow_curve()
        a, b = st.columns(2)
        f1 = px.line(el, x="k", y="Inertia", markers=True, color_discrete_sequence=PALETTE,
                     title="K-Means elbow curve")
        f2 = px.line(el, x="k", y="Silhouette", markers=True, color_discrete_sequence=PALETTE,
                     title="Silhouette score vs k")
        a.plotly_chart(f1, width="stretch")
        b.plotly_chart(f2, width="stretch")
        with st.expander("Dendrogram (hierarchical clustering)"):
            st.pyplot(dendrogram_figure())

    with tab_assign:
        if model is None:
            st.info("Switch the algorithm to K-Means: agglomerative clustering has no predict step, "
                    "so it cannot place a new player into a cluster.")
        else:
            players = base["df"]
            q = st.text_input("Search player", key="cl_q")
            pool = players[players["short_name"].str.contains(q, case=False, na=False, regex=False)] if q else players
            pool = pool.sort_values("overall", ascending=False).head(50)
            if pool.empty:
                st.warning("No player found with that name.")
            else:
                lab = {i: f"{r.short_name} - OVR {r.overall}, {r.player_positions}" for i, r in pool.iterrows()}
                sel = st.selectbox("Player", list(lab), format_func=lab.get, key="cl_sel")
                x = base["scaler"].transform(players.loc[[sel], base["X_cols"]])
                c = int(model.predict(x)[0])
                st.success(f"**{players.loc[sel, 'short_name']}** falls in K-Means cluster **{c}** (k={k}).")
                st.caption(f"Listed position group: {players.loc[sel, 'position_group']}")


# --------------------------------------------------------------------------- #
PAGES = {"🏠 Home": page_home, "🏨 Hotel cancellation (Classification)": page_hotel,
         "⚽ Player value (Regression)": page_value, "🧩 Player clusters (Clustering)": page_clusters}
choice = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.caption("Capstone project · 23CSE301")
PAGES[choice]()
