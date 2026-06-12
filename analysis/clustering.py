import os

import json
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from feature_extract import extract_features


ROOT_DIR = "/content/drive/MyDrive/建築パース解析"
ANALYSIS_DIR = os.path.join(ROOT_DIR, "analysis")

LATEST_DIR = os.path.join(ANALYSIS_DIR, "latest")
RUNS_DIR = os.path.join(ANALYSIS_DIR, "runs")

def create_run_dirs():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{now}"

    os.makedirs(LATEST_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    return run_id, run_dir

def find_latest_queue_csv():
    csv_files = [
        f for f in os.listdir(ANALYSIS_DIR)
        if f.startswith("analysis_queue_") and f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError("analysis_queue_ で始まるCSVが見つかりません。")

    csv_files.sort(
        key=lambda f: os.path.getmtime(os.path.join(ANALYSIS_DIR, f)),
        reverse=True
    )

    return os.path.join(ANALYSIS_DIR, csv_files[0])


def find_image_path(file_name):
    for root, dirs, files in os.walk(ROOT_DIR):
        if file_name in files:
            return os.path.join(root, file_name)

    return None


def load_queue():
    queue_path = find_latest_queue_csv()
    print("使用CSV:", os.path.basename(queue_path))

    df = pd.read_csv(queue_path)

    required_cols = ["id", "fileName", "mainLabel"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
      raise ValueError(f"必要な列がありません: {missing}")

    return df


def build_feature_table(df):
    rows = []

    for _, row in df.iterrows():
        file_name = str(row["fileName"])
        image_path = find_image_path(file_name)

        if image_path is None:
            print("画像が見つかりません:", file_name)
            continue

        features = extract_features(image_path)

        features.update({
            "id": row["id"],
            "fileName": row["fileName"],
            "mainLabel": row["mainLabel"],
            "imagePath": image_path
        })

        rows.append(features)

    if not rows:
        raise ValueError("解析できる画像がありませんでした。")

    return pd.DataFrame(rows)


def run_clustering(color_df):
    feature_cols = [
        "mean_r",
        "mean_g",
        "mean_b",
        "mean_hue",
        "mean_brightness",
        "mean_saturation",
        "white_ratio",
        "black_ratio",
        "color_count",
        "contrast",
        "edge_ratio",
        "aspect_ratio",
        "green_ratio",
        "sky_like_ratio",
        "warm_ratio",
        "cool_ratio",
    ]

    missing = [c for c in feature_cols if c not in color_df.columns]

    if missing:
        raise ValueError(f"color_dfに不足している列があります: {missing}")

    X = color_df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_samples = len(color_df)
    n_clusters = min(3, n_samples)

    if n_clusters < 2:
        raise ValueError("クラスタリングには最低2枚以上の画像が必要です。")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    color_df["cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)

    color_df["PC1"] = pca_result[:, 0]
    color_df["PC2"] = pca_result[:, 1]

# =========================
# PCA 主成分の意味分析
# =========================

pca_components = pd.DataFrame(
    pca.components_,
    columns=feature_cols,
    index=['PC1', 'PC2']
)

def get_top_pca_features(pc_name, top_n=5):
    series = pca_components.loc[pc_name]

    result = (
        series
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )

    return [
        {
            'feature': feature,
            'weight': round(float(series[feature]), 4),
            'direction': 'positive' if series[feature] >= 0 else 'negative'
        }
        for feature in result
    ]

pc1_top_features = get_top_pca_features('PC1')
pc2_top_features = get_top_pca_features('PC2')

    print("クラスタ数:", n_clusters)
    print("PC1寄与率:", round(pca.explained_variance_ratio_[0], 4))
    print("PC2寄与率:", round(pca.explained_variance_ratio_[1], 4))

    return color_df


def save_result_csv(color_df, run_dir):
    latest_path = os.path.join(LATEST_DIR, "color_analysis_result.csv")
    run_path = os.path.join(run_dir, "result.csv")

    save_df = color_df.drop(columns=["imagePath"], errors="ignore")

    save_df.to_csv(latest_path, index=False, encoding="utf-8-sig")
    save_df.to_csv(run_path, index=False, encoding="utf-8-sig")

    print("結果CSVを保存しました:")
    print(latest_path)
    print(run_path)

def save_cluster_plot(color_df, run_dir):
    latest_path = os.path.join(LATEST_DIR, "color_cluster_plot.png")
    run_path = os.path.join(run_dir, "cluster_plot.png")

    plt.figure(figsize=(8, 6))

    for cluster_id in sorted(color_df["cluster"].unique()):
        group = color_df[color_df["cluster"] == cluster_id]

        plt.scatter(
            group["PC1"],
            group["PC2"],
            label=f"Cluster {cluster_id}",
            s=90
        )

        for _, row in group.iterrows():
            plt.text(
                row["PC1"],
                row["PC2"],
                str(row["id"]),
                fontsize=9
            )

    plt.title("Color Feature Clustering")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.savefig(latest_path, dpi=200)
    plt.savefig(run_path, dpi=200)
    plt.show()

    print("クラスタ図を保存しました:")
    print(latest_path)
    print(run_path)

def save_summary(color_df, run_id, run_dir):
    summary = {
        "run_id": run_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_count": int(len(color_df)),
        "cluster_count": int(color_df["cluster"].nunique()),
        'pc1_explained_ratio': round(float(pca.explained_variance_ratio_[0]), 4),
        'pc2_explained_ratio': round(float(pca.explained_variance_ratio_[1]), 4),
        'pc1_top_features': pc1_top_features,
        'pc2_top_features': pc2_top_features,
        "features": [
            "mean_r",
            "mean_g",
            "mean_b",
            "mean_hue",
            "mean_brightness",
            "mean_saturation",
            "white_ratio",
            "black_ratio",
            "color_count",
            "contrast",
            "edge_ratio",
            "aspect_ratio",
            "green_ratio",
            "sky_like_ratio",
            "warm_ratio",
            "cool_ratio"
        ]
    }

    latest_path = os.path.join(LATEST_DIR, "analysis_summary.json")
    run_path = os.path.join(run_dir, "summary.json")

    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("summary.json を保存しました:")
    print(latest_path)
    print(run_path)

def main():
    print("Perspective Lab Analysis Start")

    run_id, run_dir = create_run_dirs()

    df = load_queue()
    color_df = build_feature_table(df)
    color_df = run_clustering(color_df)

    save_result_csv(color_df, run_dir)
    save_cluster_plot(color_df, run_dir)
    save_summary(color_df, run_id, run_dir)

    print("Perspective Lab Analysis Complete")

if __name__ == "__main__":
    main()
