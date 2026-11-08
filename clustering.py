"""
Atividade - Aprendizagem Não Supervisionada (GEX1225 - Inteligência Artificial, UFFS).

Agrupamento do conjunto de dados de gostos musicais e de jogos com:
  1. K-Means (sklearn.cluster.KMeans)
  2. Agrupamento Hierárquico Aglomerativo (sklearn.cluster.AgglomerativeClustering)

Etapas:
  - Varredura de parâmetros (variações) de cada algoritmo, com métricas internas
    de qualidade (silhueta, Davies-Bouldin, Calinski-Harabasz, inércia) e
    estabilidade (ARI entre sementes).
  - Execução da configuração escolhida para cada algoritmo.
  - Armazenamento do grupo de cada amostra e dos centroides de cada grupo.
  - Geração de figuras e tabelas usadas no relatório.
  - Respostas às perguntas da atividade (grupos dos integrantes, recomendações).

Uso:
  python clustering.py --integrantes "Henrique Rodrigues"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.neighbors import NearestCentroid

BASE = Path(__file__).resolve().parent
DATASET = BASE / "MusicasJogosDataset_20232_20262.csv"
SAIDA = BASE / "resultados"
FIG = SAIDA / "figuras"

SEED = 42
K_RANGE = range(2, 11)

# Configurações escolhidas após a análise das variações (ver relatório).
# Podem ser sobrescritas via linha de comando.
K_ESCOLHIDO_KMEANS = 3
K_ESCOLHIDO_AGLO = 3
LINKAGE_ESCOLHIDO = "ward"


# --------------------------------------------------------------------------
# Dados
# --------------------------------------------------------------------------
def carregar_dados() -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_csv(DATASET, index_col=0, encoding="utf-8")
    df.index = df.index.astype(str).str.strip()
    df.index.name = "Pessoa"
    df.columns = df.columns.str.strip()
    df = df.astype(int)
    colunas = list(df.columns)
    idx_fps = colunas.index("FPS")
    musicas = colunas[:idx_fps]
    jogos = colunas[idx_fps:]
    return df, musicas, jogos


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def metricas(X: np.ndarray, rotulos: np.ndarray, D_jac: np.ndarray) -> dict:
    n_grupos = len(np.unique(rotulos))
    tamanhos = np.bincount(rotulos)
    res = {
        "n_grupos": n_grupos,
        "menor_grupo": int(tamanhos.min()),
        "maior_grupo": int(tamanhos.max()),
    }
    if 1 < n_grupos < len(X):
        res["silhueta_euclid"] = silhouette_score(X, rotulos, metric="euclidean")
        res["silhueta_jaccard"] = silhouette_score(D_jac, rotulos, metric="precomputed")
        res["davies_bouldin"] = davies_bouldin_score(X, rotulos)
        res["calinski_harabasz"] = calinski_harabasz_score(X, rotulos)
    return res


# --------------------------------------------------------------------------
# Variações - K-Means
# --------------------------------------------------------------------------
def variacoes_kmeans(X: np.ndarray, D_jac: np.ndarray) -> pd.DataFrame:
    linhas = []
    for init in ("k-means++", "random"):
        for k in K_RANGE:
            modelo = KMeans(n_clusters=k, init=init, n_init=50, random_state=SEED).fit(X)
            m = metricas(X, modelo.labels_, D_jac)
            # Estabilidade: ARI médio entre 10 execuções com sementes distintas
            # e n_init=1 (sensibilidade à inicialização).
            rot = [
                KMeans(n_clusters=k, init=init, n_init=1, random_state=s).fit_predict(X)
                for s in range(10)
            ]
            aris = [
                adjusted_rand_score(rot[i], rot[j])
                for i in range(len(rot))
                for j in range(i + 1, len(rot))
            ]
            linhas.append(
                {
                    "init": init,
                    "k": k,
                    "inercia": modelo.inertia_,
                    "iteracoes": modelo.n_iter_,
                    "ari_estabilidade_n_init1": float(np.mean(aris)),
                    **m,
                }
            )
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Variações - Aglomerativo
# --------------------------------------------------------------------------
def modelo_aglomerativo(X, D_jac, k, linkage_, metrica):
    if linkage_ == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    if metrica == "jaccard":
        return AgglomerativeClustering(
            n_clusters=k, linkage=linkage_, metric="precomputed"
        ).fit(D_jac)
    return AgglomerativeClustering(n_clusters=k, linkage=linkage_, metric=metrica).fit(X)


def variacoes_aglomerativo(X: np.ndarray, D_jac: np.ndarray) -> pd.DataFrame:
    combinacoes = [("ward", "euclidean")] + [
        (lk, mt)
        for lk in ("complete", "average", "single")
        for mt in ("euclidean", "manhattan", "cosine", "jaccard")
    ]
    linhas = []
    for lk, mt in combinacoes:
        for k in K_RANGE:
            modelo = modelo_aglomerativo(X, D_jac, k, lk, mt)
            linhas.append({"linkage": lk, "metric": mt, "k": k, **metricas(X, modelo.labels_, D_jac)})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Execução das configurações escolhidas
# --------------------------------------------------------------------------
def executar_escolhidos(X, k_km, k_ag, linkage_):
    km = KMeans(n_clusters=k_km, init="k-means++", n_init=50, random_state=SEED).fit(X)
    ag = AgglomerativeClustering(n_clusters=k_ag, linkage=linkage_).fit(X)
    # O AgglomerativeClustering não gera centroides; eles são obtidos com o
    # NearestCentroid (média das amostras de cada grupo).
    nc = NearestCentroid().fit(X, ag.labels_)
    return km, km.cluster_centers_, ag, nc.centroids_


def renumerar_por_tamanho(rotulos: np.ndarray, centroides: np.ndarray):
    """Renumera os grupos de 0..k-1 em ordem decrescente de tamanho."""
    ordem = np.argsort(-np.bincount(rotulos), kind="stable")
    mapa = {antigo: novo for novo, antigo in enumerate(ordem)}
    return np.vectorize(mapa.get)(rotulos), centroides[ordem]


# --------------------------------------------------------------------------
# Interpretação
# --------------------------------------------------------------------------
def extremos(centroide: pd.Series, media_geral: pd.Series, n: int = 5) -> dict:
    desvio = (centroide - media_geral).sort_values()
    return {
        "mais_altos": [(c, round(centroide[c], 3), round(desvio[c], 3)) for c in desvio.index[::-1][:n]],
        "mais_baixos": [(c, round(centroide[c], 3), round(desvio[c], 3)) for c in desvio.index[:n]],
    }


def recomendar(cent: pd.DataFrame, media: pd.Series, chave: str, alto: bool, candidatos: list[str]):
    """Escolhe o grupo com valor mais alto (ou mais baixo) de `chave` no centroide
    e recomenda os candidatos com maior valor absoluto e maior desvio em relação
    à média geral nesse grupo."""
    g = cent[chave].idxmax() if alto else cent[chave].idxmin()
    linha = cent.loc[g, candidatos]
    desvio = (linha - media[candidatos]).sort_values(ascending=False)
    return {
        "grupo": int(g),
        "valor_chave_no_grupo": round(float(cent.loc[g, chave]), 3),
        "valor_chave_por_grupo": {int(i): round(float(v), 3) for i, v in cent[chave].items()},
        "top_absoluto": [(c, round(float(linha[c]), 3)) for c in linha.sort_values(ascending=False).index[:5]],
        "top_relativo": [(c, round(float(desvio[c]), 3)) for c in desvio.index[:5]],
    }


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def fig_variacoes_kmeans(df_km: pd.DataFrame):
    sub = df_km[df_km.init == "k-means++"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(sub.k, sub.inercia, "o-")
    ax[0].set(title="Inércia (método do cotovelo)", xlabel="k", ylabel="Inércia")
    ax[1].plot(sub.k, sub.silhueta_euclid, "o-", label="Euclidiana")
    ax[1].plot(sub.k, sub.silhueta_jaccard, "s--", label="Jaccard")
    ax[1].set(title="Coeficiente de silhueta", xlabel="k")
    ax[1].legend()
    ax[2].plot(sub.k, sub.davies_bouldin, "o-", color="C3")
    ax[2].set(title="Índice Davies-Bouldin (menor é melhor)", xlabel="k")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "kmeans_variacoes.png", dpi=200)
    plt.close(fig)


def fig_variacoes_aglo(df_ag: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for (lk, mt), sub in df_ag.groupby(["linkage", "metric"]):
        if mt not in ("euclidean", "jaccard"):
            continue
        estilo = "-" if mt == "euclidean" else "--"
        ax[0].plot(sub.k, sub.silhueta_euclid, estilo, marker="o", ms=3, label=f"{lk}/{mt}")
        ax[1].plot(sub.k, sub.menor_grupo, estilo, marker="o", ms=3, label=f"{lk}/{mt}")
    ax[0].set(title="Silhueta (euclidiana)", xlabel="k")
    ax[1].set(title="Tamanho do menor grupo", xlabel="k")
    ax[1].legend(fontsize=7, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "aglomerativo_variacoes.png", dpi=200)
    plt.close(fig)


def fig_dendrograma(X: np.ndarray, k: int):
    Z = linkage(X, method="ward")
    corte = (Z[-k, 2] + Z[-k + 1, 2]) / 2
    fig, ax = plt.subplots(figsize=(12, 4.5))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=corte, above_threshold_color="grey")
    ax.axhline(corte, ls="--", c="k", lw=0.8)
    ax.set(title=f"Dendrograma (ligação de Ward) - corte em {k} grupos", ylabel="Distância")
    fig.tight_layout()
    fig.savefig(FIG / "dendrograma_ward.png", dpi=200)
    plt.close(fig)


def fig_pca(X, rot_km, rot_ag, nomes, destaque):
    pca = PCA(n_components=2, random_state=SEED)
    P = pca.fit_transform(X)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for a, rot, tit in ((ax[0], rot_km, "K-Means"), (ax[1], rot_ag, "Aglomerativo (Ward)")):
        for g in np.unique(rot):
            m = rot == g
            a.scatter(P[m, 0], P[m, 1], s=22, alpha=0.75, label=f"Grupo {g}")
        for nome in destaque:
            if nome in nomes:
                i = nomes.index(nome)
                a.scatter(P[i, 0], P[i, 1], s=140, facecolors="none", edgecolors="k", lw=1.5)
                a.annotate(nome, P[i], xytext=(6, 6), textcoords="offset points", fontsize=8)
        a.set(title=tit, xlabel=f"CP1 ({pca.explained_variance_ratio_[0]:.1%})",
              ylabel=f"CP2 ({pca.explained_variance_ratio_[1]:.1%})")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "pca_grupos.png", dpi=200)
    plt.close(fig)


def fig_centroides(cent: pd.DataFrame, nome_arq: str, titulo: str):
    fig, ax = plt.subplots(figsize=(13, 0.6 * len(cent) + 2.2))
    im = ax.imshow(cent.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(cent.shape[1]), cent.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cent)), [f"Grupo {g}" for g in cent.index])
    for i in range(cent.shape[0]):
        for j in range(cent.shape[1]):
            ax.text(j, i, f"{cent.values[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=6)
    ax.set_title(titulo)
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    fig.tight_layout()
    fig.savefig(FIG / nome_arq, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Principal
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrantes", nargs="+", default=["Henrique Rodrigues"])
    ap.add_argument("--k-kmeans", type=int, default=K_ESCOLHIDO_KMEANS)
    ap.add_argument("--k-aglo", type=int, default=K_ESCOLHIDO_AGLO)
    ap.add_argument("--linkage", default=LINKAGE_ESCOLHIDO)
    args = ap.parse_args()

    SAIDA.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    df, musicas, jogos = carregar_dados()
    X = df.values.astype(float)
    nomes = list(df.index)
    D_jac = pairwise_distances(X.astype(bool), metric="jaccard")
    media = df.mean()

    # Estatística descritiva
    desc = pd.DataFrame({"proporcao_gosta": media.round(3), "tipo": ["Música" if c in musicas else "Jogo" for c in df.columns]})
    desc.to_csv(SAIDA / "descritiva.csv", encoding="utf-8-sig")

    # 1. Variações
    df_km = variacoes_kmeans(X, D_jac)
    df_ag = variacoes_aglomerativo(X, D_jac)
    df_km.round(4).to_csv(SAIDA / "variacoes_kmeans.csv", index=False, encoding="utf-8-sig")
    df_ag.round(4).to_csv(SAIDA / "variacoes_aglomerativo.csv", index=False, encoding="utf-8-sig")
    fig_variacoes_kmeans(df_km)
    fig_variacoes_aglo(df_ag)

    # 2. Configurações escolhidas
    km, c_km, ag, c_ag = executar_escolhidos(X, args.k_kmeans, args.k_aglo, args.linkage)
    rot_km, c_km = renumerar_por_tamanho(km.labels_, c_km)
    rot_ag, c_ag = renumerar_por_tamanho(ag.labels_, c_ag)

    # 2a. Classe de cada amostra
    rotulos = pd.DataFrame({"KMeans": rot_km, "Aglomerativo": rot_ag}, index=df.index)
    rotulos.to_csv(SAIDA / "grupos_por_amostra.csv", encoding="utf-8-sig")
    # 2b. Centroides
    cent_km = pd.DataFrame(c_km, columns=df.columns).rename_axis("Grupo")
    cent_ag = pd.DataFrame(c_ag, columns=df.columns).rename_axis("Grupo")
    cent_km.round(4).to_csv(SAIDA / "centroides_kmeans.csv", encoding="utf-8-sig")
    cent_ag.round(4).to_csv(SAIDA / "centroides_aglomerativo.csv", encoding="utf-8-sig")

    tab_contingencia = pd.crosstab(rotulos.KMeans, rotulos.Aglomerativo)
    tab_contingencia.to_csv(SAIDA / "contingencia_kmeans_x_aglomerativo.csv", encoding="utf-8-sig")

    fig_dendrograma(X, args.k_aglo)
    fig_pca(X, rot_km, rot_ag, nomes, args.integrantes)
    fig_centroides(cent_km, "centroides_kmeans.png", f"Centroides - K-Means (k = {args.k_kmeans})")
    fig_centroides(cent_ag, "centroides_aglomerativo.png", f"Centroides - Aglomerativo {args.linkage} (k = {args.k_aglo})")

    # 3. Interpretação
    resumo = {"n_amostras": len(df), "n_musicas": len(musicas), "n_jogos": len(jogos)}
    for nome_alg, rot, cent in (("KMeans", rot_km, cent_km), ("Aglomerativo", rot_ag, cent_ag)):
        r = {
            "tamanhos": {int(g): int(n) for g, n in enumerate(np.bincount(rot))},
            "metricas": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                         for k, v in metricas(X, rot, D_jac).items()},
            "extremos_por_grupo": {int(g): extremos(cent.loc[g], media) for g in cent.index},
            "media_musicas_por_grupo": cent[musicas].mean(axis=1).round(3).to_dict(),
            "media_jogos_por_grupo": cent[jogos].mean(axis=1).round(3).to_dict(),
            "integrantes": {n: int(rotulos.loc[n, nome_alg]) for n in args.integrantes if n in rotulos.index},
            "rec_brega": recomendar(cent, media, "Brega", True, jogos),
            "rec_nao_tabuleiro": recomendar(cent, media, "Tabuleiro", False, musicas),
        }
        resumo[nome_alg] = r
    resumo["ari_kmeans_vs_aglomerativo"] = adjusted_rand_score(rot_km, rot_ag)

    # Verificação direta nos dados (sem agrupamento), apenas como referência
    brega = df[df.Brega == 1]
    nao_tab = df[df.Tabuleiro == 0]
    resumo["referencia_direta"] = {
        "n_gosta_brega": len(brega),
        "jogos_mais_comuns_entre_quem_gosta_brega": brega[jogos].mean().sort_values(ascending=False).head(5).round(3).to_dict(),
        "n_nao_gosta_tabuleiro": len(nao_tab),
        "musicas_mais_comuns_entre_quem_nao_gosta_tabuleiro": nao_tab[musicas].mean().sort_values(ascending=False).head(5).round(3).to_dict(),
    }

    with open(SAIDA / "resumo.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2, default=str)

    # Saída no terminal
    pd.set_option("display.width", 200)
    print("=== Variações K-Means (k-means++) ===")
    print(df_km[df_km.init == "k-means++"].round(3).to_string(index=False))
    print("\n=== Variações K-Means (random) ===")
    print(df_km[df_km.init == "random"].round(3).to_string(index=False))
    print("\n=== Variações Aglomerativo ===")
    print(df_ag.round(3).to_string(index=False))
    print("\n=== Contingência K-Means x Aglomerativo ===")
    print(tab_contingencia)
    print("\n=== Resumo ===")
    print(json.dumps(resumo, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
