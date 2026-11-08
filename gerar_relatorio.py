"""
Gera o relatório da atividade no formato de artigo (modelo SBC, o mesmo do
relatório de referência), em PDF (reportlab) e DOCX editável (python-docx), a
partir dos resultados produzidos por clustering.py.

Uso:
  python clustering.py
  python gerar_relatorio.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

BASE = Path(__file__).resolve().parent
RES = BASE / "resultados"
FIG = RES / "figuras"
SAIDA_PDF = BASE / "Relatorio_Aprendizagem_Nao_Supervisionada.pdf"
SAIDA_DOCX = BASE / "Relatorio_Aprendizagem_Nao_Supervisionada.docx"

TITULO = "Agrupamento de Gostos Musicais e de Jogos com K-Means e Agrupamento Hierárquico Aglomerativo"
AUTOR = "Henrique Ribeiro Rodrigues"
AUTOR_DATASET = "Henrique Rodrigues"
AFILIACAO = "Universidade Federal da Fronteira Sul (UFFS) – Chapecó, SC – Brasil"
DISCIPLINA = "Inteligência Artificial (GEX1225) – Prof. Felipe Grando"
EMAIL = "henrique.ribeiro@uffs.edu.br"
ANO = "2026"
FONTE = f"Fonte: elaborada pelo autor ({ANO})."


# ==========================================================================
# Dados e estatísticas usados no texto
# ==========================================================================
def _ler():
    df = pd.read_csv(BASE / "MusicasJogosDataset_20232_20262.csv", index_col=0, encoding="utf-8")
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.str.strip()
    df = df.astype(int)
    rot = pd.read_csv(RES / "grupos_por_amostra.csv", index_col=0, encoding="utf-8-sig")
    ckm = pd.read_csv(RES / "centroides_kmeans.csv", index_col=0, encoding="utf-8-sig")
    cag = pd.read_csv(RES / "centroides_aglomerativo.csv", index_col=0, encoding="utf-8-sig")
    vkm = pd.read_csv(RES / "variacoes_kmeans.csv", encoding="utf-8-sig")
    vag = pd.read_csv(RES / "variacoes_aglomerativo.csv", encoding="utf-8-sig")
    resumo = json.loads((RES / "resumo.json").read_text(encoding="utf-8"))
    return df, rot, ckm, cag, vkm, vag, resumo


DF, ROT, CKM, CAG, VKM, VAG, RESUMO = _ler()
X = DF.values.astype(float)
N = len(DF)
MUSICAS = list(DF.columns[: list(DF.columns).index("FPS")])
JOGOS = list(DF.columns[list(DF.columns).index("FPS"):])
GOSTOS = DF.sum(axis=1)
GOSTOS_KM = GOSTOS.groupby(ROT.KMeans.values).mean()
GOSTOS_AG = GOSTOS.groupby(ROT.Aglomerativo.values).mean()
_pca = PCA(n_components=2, random_state=42).fit(X)
CORR_CP1 = abs(float(np.corrcoef(_pca.transform(X)[:, 0], GOSTOS.values)[0, 1]))
VAR_PCA = _pca.explained_variance_ratio_
CONC = int(pd.crosstab(ROT.KMeans, ROT.Aglomerativo).values.diagonal().sum())
ARI = RESUMO["ari_kmeans_vs_aglomerativo"]
TAM_KM = ROT.KMeans.value_counts().sort_index()
TAM_AG = ROT.Aglomerativo.value_counts().sort_index()
I_HEN = list(DF.index).index(AUTOR_DATASET)
G_HEN_KM = int(ROT.loc[AUTOR_DATASET, "KMeans"])
G_HEN_AG = int(ROT.loc[AUTOR_DATASET, "Aglomerativo"])
SIL_HEN_KM = float(silhouette_samples(X, ROT.KMeans.values)[I_HEN])
SIL_HEN_AG = float(silhouette_samples(X, ROT.Aglomerativo.values)[I_HEN])
N_BREGA = int(DF.Brega.sum())
N_NTAB = int((1 - DF.Tabuleiro).sum())
BREGA_KM = DF.Brega.groupby(ROT.KMeans.values).sum()
BREGA_AG = DF.Brega.groupby(ROT.Aglomerativo.values).sum()
NTAB_KM = (1 - DF.Tabuleiro).groupby(ROT.KMeans.values).sum()
NTAB_AG = (1 - DF.Tabuleiro).groupby(ROT.Aglomerativo.values).sum()


def f2(x, casas=2):
    return f"{x:.{casas}f}".replace(".", ",")


def pct(n, tot=N):
    return f"{100 * n / tot:.1f}".replace(".", ",") + "%"


def v(cent, g, col):
    return f2(cent.loc[g, col])


def linha(df, **filtro):
    m = np.ones(len(df), dtype=bool)
    for c, val in filtro.items():
        m &= df[c].values == val
    return df[m].iloc[0]


# ==========================================================================
# Tabelas: ("tab", título, cabeçalho, linhas, larguras relativas em cm)
# ==========================================================================
def tab_diferencas():
    return ("tab", "Principais diferenças teóricas entre os algoritmos",
            ["Aspecto", "K-Means", "Aglomerativo (ligação de Ward)"],
            [
                ["Estratégia", "Particional e iterativo: refina uma partição inicial.",
                 "Hierárquico ascendente: parte de grupos unitários e funde pares."],
                ["Número de grupos", "Fixado antes da execução (k).",
                 "Definido pelo corte do dendrograma (n_clusters ou distance_threshold)."],
                ["Critério", "Minimiza a inércia global (soma dos quadrados das distâncias ao centroide).",
                 "A cada passo, funde o par que menos aumenta a variância intragrupo (decisão local)."],
                ["Revisão", "Amostras podem trocar de grupo a cada iteração.",
                 "Fusões são definitivas e não são revistas."],
                ["Determinismo", "Depende da inicialização; pode parar em mínimos locais (mitigado por n_init).",
                 "Determinístico para os mesmos dados."],
                ["Centroides", "Gerados pelo algoritmo (cluster_centers_).",
                 "Não gerados; precisam ser calculados à parte."],
                ["Custo", "Aproximadamente linear no número de amostras.",
                 "Pelo menos quadrático (matriz de distâncias)."],
            ],
            [2.6, 6.2, 6.2])


def tab_variacoes_kmeans():
    sub = VKM[VKM.init == "k-means++"]
    linhas = [[str(r.k), f2(r.inercia, 1), f2(r.silhueta_euclid, 3), f2(r.silhueta_jaccard, 3),
               f2(r.davies_bouldin, 3), f2(r.calinski_harabasz, 2), f2(r.ari_estabilidade_n_init1, 3),
               f"{r.menor_grupo}–{r.maior_grupo}"] for r in sub.itertuples()]
    return ("tab", "Variações do K-Means (init = k-means++, n_init = 50)",
            ["k", "Inércia", "Silhueta\n(euclid.)", "Silhueta\n(Jaccard)", "Davies-\nBouldin",
             "Calinski-\nHarabasz", "ARI entre\nsementes", "Grupos\n(mín.–máx.)"],
            linhas, [0.8, 1.7, 1.7, 1.7, 1.7, 1.9, 1.8, 2.3])


def tab_variacoes_aglo(k=3):
    nomes = {"euclidean": "euclidiana", "manhattan": "manhattan", "cosine": "cosseno", "jaccard": "Jaccard"}
    sub = VAG[VAG.k == k]
    linhas = [[r.linkage, nomes[r.metric], f2(r.silhueta_euclid, 3), f2(r.silhueta_jaccard, 3),
               f2(r.davies_bouldin, 3), f2(r.calinski_harabasz, 2), f"{r.menor_grupo}–{r.maior_grupo}"]
              for r in sub.itertuples()]
    return ("tab", f"Variações do agrupamento aglomerativo (ligação × métrica) com k = {k}",
            ["Ligação", "Métrica", "Silhueta\n(euclid.)", "Silhueta\n(Jaccard)", "Davies-\nBouldin",
             "Calinski-\nHarabasz", "Grupos\n(mín.–máx.)"],
            linhas, [1.8, 2.0, 1.9, 1.9, 1.9, 2.0, 2.1])


def tab_configuracoes():
    return ("tab", "Configurações escolhidas",
            ["Algoritmo", "Parâmetros", "Centroides"],
            [
                ["KMeans", "n_clusters = 3; init = k-means++; n_init = 50; random_state = 42; demais no padrão "
                           "(max_iter = 300, tol = 1e-4)", "cluster_centers_"],
                ["AgglomerativeClustering", "n_clusters = 3; linkage = ward; metric = euclidean (exigida pelo Ward); "
                                            "demais no padrão", "NearestCentroid (média de cada grupo)"],
            ],
            [4.4, 6.8, 3.8])


def tab_tamanhos():
    linhas = [[f"Grupo {g}", f"{TAM_KM[g]} ({pct(TAM_KM[g])})", f"{TAM_AG[g]} ({pct(TAM_AG[g])})"] for g in TAM_KM.index]
    linhas.append(["Total", f"{N} (100,0%)", f"{N} (100,0%)"])
    return ("tab", "Quantidade de amostras por grupo", ["Grupo", "K-Means", "Aglomerativo"], linhas, [3.0, 3.5, 3.5])


def tab_contingencia():
    ct = pd.crosstab(ROT.KMeans, ROT.Aglomerativo)
    linhas = [[f"K-Means {g}"] + [str(x) for x in ct.loc[g]] + [str(ct.loc[g].sum())] for g in ct.index]
    linhas.append(["Total"] + [str(x) for x in ct.sum()] + [str(N)])
    return ("tab", "Contingência entre os grupos do K-Means (linhas) e do aglomerativo (colunas)",
            ["", "Aglo. 0", "Aglo. 1", "Aglo. 2", "Total"], linhas, [2.6, 1.8, 1.8, 1.8, 1.8])


def tab_centroides():
    media = DF.mean()
    linhas = [[f"{c} ({'M' if c in MUSICAS else 'J'})"] + [f2(CKM.loc[g, c]) for g in CKM.index]
              + [f2(CAG.loc[g, c]) for g in CAG.index] + [f2(media[c])] for c in DF.columns]
    return ("tab", "Centroides dos grupos (proporção de pessoas do grupo que gostam de cada estilo)",
            ["Estilo", "KM 0", "KM 1", "KM 2", "AG 0", "AG 1", "AG 2", "Média"],
            linhas, [4.0, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4, 1.6])


def tab_recomendacoes():
    return ("tab", "Resumo das recomendações",
            ["Perfil", "Algoritmo", "Grupo usado", "Recomendação"],
            [
                ["Gosta de Brega", "K-Means", f"0 (Brega = {v(CKM, 0, 'Brega')})", f"Survival ({v(CKM, 0, 'Survival')})"],
                ["Gosta de Brega", "Aglomerativo", f"0 (Brega = {v(CAG, 0, 'Brega')})", f"RPG ({v(CAG, 0, 'RPG')})"],
                ["Não gosta de Tabuleiro", "K-Means", f"1 (Tabuleiro = {v(CKM, 1, 'Tabuleiro')})", f"Rock ({v(CKM, 1, 'Rock')})"],
                ["Não gosta de Tabuleiro", "Aglomerativo", f"1 (Tabuleiro = {v(CAG, 1, 'Tabuleiro')})", f"Rock ({v(CAG, 1, 'Rock')})"],
            ],
            [3.8, 2.8, 3.8, 3.4])


# ==========================================================================
# Conteúdo: ("h", título) | ("p", parágrafo) | ("fig", arquivo, título) | ("tab", ...) | ("ref", texto)
# ==========================================================================
ABSTRACT = (
    "This report applies two clustering algorithms from the scikit-learn library, K-Means and agglomerative "
    f"hierarchical clustering, to a dataset describing the preferences of {N} people for 21 music genres and 21 game "
    "genres. Several parameter settings were tested and evaluated with internal metrics. Both algorithms were run "
    f"with three clusters and produced similar partitions ({100 * CONC / N:.1f}% agreement), revealing three profiles: "
    "eclectic heavy gamers; rock, indie and metal listeners with few interests; and sertanejo, reggaeton and funk "
    "listeners who prefer casual games. Based on the centroids, Survival or RPG games are recommended for people "
    "who like Brega music, and Rock music for people who do not like board games."
)
RESUMO_PT = (
    "Este relatório aplica dois algoritmos de agrupamento da biblioteca scikit-learn, o K-Means e o agrupamento "
    f"hierárquico aglomerativo, a um conjunto de dados com os gostos de {N} pessoas por 21 estilos musicais e 21 "
    "estilos de jogos. Diversas configurações de parâmetros foram testadas e avaliadas com métricas internas. Os dois "
    f"algoritmos foram executados com três grupos e produziram partições semelhantes ({pct(CONC)} de concordância), "
    "revelando três perfis: ecléticos que jogam muito; ouvintes de rock, indie e metal com poucos interesses; e "
    "ouvintes de sertanejo, reggaeton e funk que preferem jogos casuais. A partir dos centroides, recomenda-se "
    "Survival ou RPG para quem gosta de Brega e Rock para quem não gosta de jogos de tabuleiro."
)


def conteudo():
    k, a = CKM, CAG
    km2 = linha(VKM, init="k-means++", k=2)
    km3 = linha(VKM, init="k-means++", k=3)
    km4 = linha(VKM, init="k-means++", k=4)
    rd3 = linha(VKM, init="random", k=3)
    w2 = linha(VAG, linkage="ward", k=2)
    w3 = linha(VAG, linkage="ward", k=3)
    w4 = linha(VAG, linkage="ward", k=4)
    s2 = linha(VAG, linkage="single", metric="euclidean", k=2)
    ac2 = linha(VAG, linkage="average", metric="cosine", k=2)
    cj3 = linha(VAG, linkage="complete", metric="jaccard", k=3)
    hen = DF.loc[AUTOR_DATASET]
    ref = RESUMO["referencia_direta"]

    return [
        ("h", "1. Introdução"),
        ("p", "O agrupamento (*clustering*) é uma tarefa de aprendizagem não supervisionada que divide amostras sem "
              "rótulo em grupos, de modo que amostras do mesmo grupo sejam parecidas entre si e diferentes das amostras "
              "dos outros grupos. Neste trabalho, essa técnica é aplicada a um conjunto de dados sobre gostos musicais e "
              "de jogos, disponibilizado no SIGAA, com o objetivo de identificar perfis de pessoas e usá-los para fazer "
              "recomendações."),
        ("p", "Foram utilizados dois algoritmos da biblioteca scikit-learn (Pedregosa *et al.*, 2011): o K-Means, exigido "
              "pela atividade, e o agrupamento hierárquico aglomerativo (AgglomerativeClustering), escolhido por seguir "
              "uma estratégia bem diferente, o que torna a comparação mais informativa. Para cada algoritmo foram testadas "
              "diversas configurações de parâmetros; a configuração escolhida foi executada e o grupo de cada amostra e os "
              "centroides de cada grupo foram armazenados."),
        ("p", "A Seção 2 descreve os dados e as métricas de avaliação; a Seção 3 apresenta os algoritmos, os experimentos "
              "e as configurações escolhidas; a Seção 4 analisa os resultados, identifica o grupo do autor e responde às "
              "perguntas de recomendação; a Seção 5 conclui o trabalho."),

        ("h", "2. Dados e Metodologia"),
        ("h", "2.1 Conjunto de Dados"),
        ("p", f"O arquivo MusicasJogosDataset_20232_20262.csv contém {N} amostras (pessoas) e {len(DF.columns)} atributos "
              f"binários: {len(MUSICAS)} estilos musicais (de Pop a Clássica) e {len(JOGOS)} estilos de jogos (de FPS a "
              "Realidade Virtual). O valor 1 indica que a pessoa gosta do estilo e 0 que não gosta. O nome da pessoa foi "
              "usado apenas como identificador. Não há valores ausentes, e nenhum dado externo foi usado para completar o "
              "conjunto."),
        ("p", "Como todos os atributos estão na mesma escala, não foi aplicada normalização. Nesse caso, o centroide de um "
              "grupo indica a proporção de pessoas do grupo que gostam de cada estilo. Os estilos mais populares são Rock "
              f"({f2(DF.Rock.mean())}) e Pop ({f2(DF.Pop.mean())}); os menos populares são Brega ({f2(DF.Brega.mean())}) e "
              f"K-Pop ({f2(DF['K-Pop'].mean())}). Em média, cada pessoa gosta de {f2(GOSTOS.mean(), 1)} dos "
              f"{len(DF.columns)} estilos."),
        ("h", "2.2 Métricas de Avaliação"),
        ("p", "Como não existem rótulos verdadeiros, as configurações foram comparadas com métricas internas, além da "
              "interpretação dos centroides:"),
        ("p", "**Silhueta:** varia de −1 a 1; valores maiores indicam grupos mais coesos e separados (Rousseeuw, 1987). "
              "Foi calculada com a distância euclidiana e com a distância de Jaccard, que ignora as coausências (dois "
              "\"não gosto\") e é adequada a dados binários."),
        ("p", "**Davies-Bouldin e Calinski-Harabasz:** o primeiro é melhor quanto menor; o segundo, quanto maior."),
        ("p", "**Inércia:** soma dos quadrados das distâncias ao centroide, usada no método do cotovelo do K-Means."),
        ("p", "**Índice de Rand ajustado (ARI):** mede a concordância entre duas partições (1 = idênticas; perto de 0 = "
              "casual). Foi usado para medir a estabilidade do K-Means entre sementes e para comparar os algoritmos."),
        ("p", "**Tamanho do menor e do maior grupo:** usado para detectar partições degeneradas, como um grupo com uma "
              "única pessoa."),

        ("h", "3. Algoritmos e Configurações"),
        ("h", "3.1 Algoritmos Escolhidos"),
        ("p", "O **K-Means** (MacQueen, 1967) divide as amostras em *k* grupos representados por centroides. A cada "
              "iteração, atribui cada amostra ao centroide mais próximo e recalcula os centroides como a média das "
              "amostras do grupo, minimizando a inércia. O resultado depende dos centroides iniciais; por isso, usa-se a "
              "inicialização k-means++ (Arthur; Vassilvitskii, 2007) e várias reinicializações (n_init), mantendo a de "
              "menor inércia."),
        ("p", "O **agrupamento hierárquico aglomerativo** começa com cada amostra em um grupo próprio e, a cada passo, funde "
              "os dois grupos mais próximos, formando uma hierarquia (dendrograma) que é cortada no número de grupos "
              "desejado. A proximidade entre grupos é definida pela ligação: *single* (pontos mais próximos), *complete* "
              "(pontos mais distantes), *average* (média das distâncias) ou *ward* (menor aumento da variância intragrupo; "
              "Ward, 1963). A Tabela 1 resume as principais diferenças teóricas entre os dois algoritmos."),
        tab_diferencas(),
        ("h", "3.2 Variações do K-Means"),
        ("p", "O K-Means foi executado com k de 2 a 10 e com as inicializações k-means++ e random, sempre com n_init = 50 e "
              "random_state = 42. Para medir a sensibilidade à inicialização, cada k também foi executado dez vezes com "
              "n_init = 1 e sementes diferentes, calculando-se o ARI médio entre essas execuções. A Tabela 2 e a Figura 1 "
              "mostram os resultados com k-means++."),
        tab_variacoes_kmeans(),
        ("fig", "kmeans_variacoes.png", "Inércia, silhueta e índice de Davies-Bouldin do K-Means em função de k"),
        ("p", f"A inércia não apresenta um cotovelo nítido, mas a redução diminui a partir de k = 3 (queda de "
              f"{f2(km2.inercia - km3.inercia, 1)} de k = 2 para 3 e de {f2(km3.inercia - km4.inercia, 1)} de 3 para 4). A "
              f"silhueta é máxima em k = 2 ({f2(km2.silhueta_euclid, 3)}), mas k = 3 fica muito próximo "
              f"({f2(km3.silhueta_euclid, 3)}) e tem o melhor Davies-Bouldin entre k = 2 e k = 5 ({f2(km3.davies_bouldin, 3)}). "
              f"A partir de k = 4, a silhueta com Jaccard despenca ({f2(km4.silhueta_jaccard, 3)}), indicando que os grupos "
              "adicionais deixam de ser distintos. A inicialização random gerou valores praticamente iguais, mas foi menos "
              f"estável em k = 3 (ARI de {f2(rd3.ari_estabilidade_n_init1, 3)} contra {f2(km3.ari_estabilidade_n_init1, 3)})."),
        ("h", "3.3 Variações do Agrupamento Aglomerativo"),
        ("p", "O aglomerativo foi testado com k de 2 a 10 e com as ligações ward (euclidiana), complete, average e single, "
              "estas três combinadas com as métricas euclidiana, manhattan, cosseno e Jaccard (esta última por meio de uma "
              "matriz de distâncias pré-calculada). A Tabela 3 mostra os resultados para k = 3 e a Figura 2 compara as "
              "combinações para todos os valores de k."),
        tab_variacoes_aglo(3),
        ("fig", "aglomerativo_variacoes.png", "Silhueta e tamanho do menor grupo do aglomerativo em função de k"),
        ("p", f"As ligações single e average geraram partições degeneradas: com single e k = 2, um grupo tem "
              f"{s2.menor_grupo} pessoa e o outro {s2.maior_grupo} (efeito de encadeamento); com average e cosseno, "
              f"{ac2.menor_grupo} contra {ac2.maior_grupo}. Ainda assim, essas partições obtiveram as maiores silhuetas "
              f"com Jaccard (até {f2(ac2.silhueta_jaccard, 3)}), mostrando que a silhueta isolada pode premiar soluções "
              f"inúteis. A ligação complete com Jaccard também ficou desbalanceada em k = 3 ({cj3.menor_grupo} contra "
              f"{cj3.maior_grupo}). Somente Ward produziu grupos equilibrados ({w3.menor_grupo} a {w3.maior_grupo} pessoas "
              "em k = 3), com os melhores valores de Calinski-Harabasz."),
        ("p", f"Com Ward, k = 2 tem silhueta um pouco maior ({f2(w2.silhueta_euclid, 3)}) que k = 3 "
              f"({f2(w3.silhueta_euclid, 3)}), mas k = 3 tem a maior silhueta com Jaccard entre todos os k "
              f"({f2(w3.silhueta_jaccard, 3)}); em k = 4 ela cai para {f2(w4.silhueta_jaccard, 3)}. O dendrograma "
              "(Figura 3) mostra uma separação clara em três ramos."),
        ("fig", "dendrograma_ward.png", "Dendrograma com ligação de Ward e corte em três grupos"),
        ("h", "3.4 Configurações Escolhidas"),
        ("p", "Escolheu-se **k = 3 para os dois algoritmos**. Esse valor equilibra as métricas internas e gera grupos mais "
              "informativos que k = 2: com dois grupos, a divisão reflete principalmente a quantidade de estilos de que "
              "cada pessoa gosta, enquanto com três surge um perfil de gosto bem definido. Usar o mesmo k também permite "
              "comparar os algoritmos diretamente. A Tabela 4 apresenta as configurações."),
        tab_configuracoes(),
        ("p", f"O valor n_init = 50 foi adotado porque, com uma única inicialização, a concordância entre execuções em k = 3 "
              f"é baixa (ARI de {f2(km3.ari_estabilidade_n_init1, 3)}). Como a classe AgglomerativeClustering não gera "
              "centroides, eles foram calculados com NearestCentroid, que obtém a média das amostras de cada grupo. Os "
              "grupos foram renumerados em ordem decrescente de tamanho, e o grupo de cada amostra e os centroides foram "
              "armazenados nos arquivos grupos_por_amostra.csv, centroides_kmeans.csv e centroides_aglomerativo.csv."),

        ("h", "4. Resultados"),
        ("h", "4.1 Grupos Gerados"),
        ("p", "Os dois algoritmos geraram três grupos. A Tabela 5 mostra quantas amostras caíram em cada um."),
        tab_tamanhos(),
        ("p", f"O grupo 2 tem {TAM_KM[2]} pessoas nos dois algoritmos, e a diferença está na fronteira entre os grupos 0 e 1 "
              f"({TAM_KM[0]}/{TAM_KM[1]} no K-Means e {TAM_AG[0]}/{TAM_AG[1]} no aglomerativo). A Tabela 6 cruza as duas "
              f"partições: {CONC} das {N} pessoas ({pct(CONC)}) ficaram no grupo correspondente nos dois algoritmos, e o "
              f"ARI entre elas é {f2(ARI, 3)}, uma concordância alta, mas não total."),
        tab_contingencia(),
        ("p", "A Figura 4 projeta as amostras nos dois primeiros componentes principais (PCA), apenas para visualização. "
              f"Eles explicam {f2(100 * VAR_PCA[0], 1)}% e {f2(100 * VAR_PCA[1], 1)}% da variância, e os grupos aparecem "
              "como regiões contíguas, sem lacunas entre si."),
        ("fig", "pca_grupos.png", "Projeção das amostras nos dois primeiros componentes principais"),
        ("h", "4.2 Caracterização dos Grupos"),
        ("p", "As Figuras 5 e 6 mostram os centroides (verde = a maioria do grupo gosta; vermelho = a maioria não gosta), "
              "e a Tabela 7 traz os valores completos. As características predominantes foram identificadas pelos valores "
              "extremos de cada centroide, comparados com a média geral."),
        ("fig", "centroides_kmeans.png", "Centroides dos grupos do K-Means"),
        ("fig", "centroides_aglomerativo.png", "Centroides dos grupos do agrupamento aglomerativo"),
        ("p", f"**Grupo 0 – ecléticos que jogam muito** (K-Means: {TAM_KM[0]}; aglomerativo: {TAM_AG[0]}). Gostam de mais "
              f"estilos (média de {f2(GOSTOS_KM[0], 1)} dos 42). Valores muito altos em Rock ({v(k, 0, 'Rock')}), Survival "
              f"({v(k, 0, 'Survival')}), RPG ({v(k, 0, 'RPG')}), Estratégia ({v(k, 0, 'Estratégia')}), Puzzle "
              f"({v(k, 0, 'Puzzle')}) e Tabuleiro ({v(k, 0, 'Tabuleiro')}), e os maiores valores acima da média em Luta, "
              f"Arcade, Plataforma e Rítmico. No aglomerativo, também se destaca em Samba ({v(a, 0, 'Samba')}) e Pagode "
              f"({v(a, 0, 'Pagode')}). Os valores mais baixos são Brega ({v(k, 0, 'Brega')}) e K-Pop ({v(k, 0, 'K-Pop')})."),
        ("p", f"**Grupo 1 – ouvintes de rock, indie e metal com poucos interesses** (K-Means: {TAM_KM[1]}; aglomerativo: "
              f"{TAM_AG[1]}). Gostam de menos estilos (média de {f2(GOSTOS_KM[1], 1)}). Valores altos em Rock "
              f"({v(k, 1, 'Rock')}; {v(a, 1, 'Rock')} no aglomerativo), Pop ({v(k, 1, 'Pop')}) e Estratégia "
              f"({v(k, 1, 'Estratégia')}) e, apesar de gostarem de poucos estilos, ficam acima da média em Metal "
              f"({v(k, 1, 'Metal')}) e Indie ({v(k, 1, 'Indie')}). Valores muito baixos em Brega ({v(k, 1, 'Brega')}), "
              f"Reggaeton ({v(k, 1, 'Reggaeton')}), Realidade Virtual ({v(k, 1, 'Realidade Virtual')}), Sertanejo "
              f"({v(k, 1, 'Sertanejo')}), Samba ({v(k, 1, 'Samba')}) e Luta ({v(k, 1, 'Luta')})."),
        ("p", f"**Grupo 2 – ouvintes de música popular brasileira e latina que preferem jogos casuais** (K-Means: "
              f"{TAM_KM[2]}; aglomerativo: {TAM_AG[2]}). Valores muito altos em Sertanejo ({v(k, 2, 'Sertanejo')}), "
              f"Reggaeton ({v(k, 2, 'Reggaeton')}), Funk ({v(k, 2, 'Funk')}) e Eletrônica ({v(k, 2, 'Eletrônica')}), e "
              f"nos jogos Cartas ({v(k, 2, 'Cartas')}), Tabuleiro ({v(k, 2, 'Tabuleiro')}), Esporte ({v(k, 2, 'Esporte')}) "
              f"e Corrida ({v(k, 2, 'Corrida')}). Valores muito baixos em RPG ({v(k, 2, 'RPG')}), Rítmico "
              f"({v(k, 2, 'Rítmico')}), Hack and slash ({v(k, 2, 'Hack and slash')}), MOBA ({v(k, 2, 'MOBA')}), Indie "
              f"({v(k, 2, 'Indie')}) e Metal ({v(k, 2, 'Metal')}). É o grupo mais distinto e o mais parecido entre os "
              "dois algoritmos."),
        tab_centroides(),
        ("h", "4.3 Diferenças e Observações"),
        ("p", "**Estrutura de grupos fraca:** todas as silhuetas ficaram abaixo de 0,10. Os gostos formam um contínuo, e "
              "não grupos naturalmente separados; os grupos são úteis como perfis, mas pessoas próximas da fronteira "
              "podem mudar de grupo com pequenas alterações."),
        ("p", f"**Quantidade de gostos como eixo principal:** o primeiro componente principal tem correlação de "
              f"{f2(CORR_CP1)} com o número de estilos de que a pessoa gosta. Assim, a separação entre os grupos 0 e 1 se "
              "deve em boa parte a \"gostar de muitas coisas\" contra \"gostar de poucas\"; já o grupo 2 é definido pelo "
              "tipo de gosto."),
        ("p", f"**Diferença entre os algoritmos:** as {N - CONC} discordâncias ocorrem sobretudo entre os grupos 0 e 1. O "
              "K-Means, que reatribui amostras a cada iteração, coloca mais pessoas no grupo 0; o aglomerativo, cujas "
              "fusões não são revistas, desloca algumas para o grupo 1, e seu grupo 0 fica mais marcado por Samba, Pagode, "
              "MPB e MMORPG."),
        ("p", f"**Sensibilidade à inicialização:** com n_init = 1, o K-Means chega a partições bem diferentes conforme a "
              f"semente (ARI médio de {f2(km3.ari_estabilidade_n_init1, 3)} em k = 3). O aglomerativo, por ser "
              "determinístico, não tem esse problema."),
        ("p", "**Métricas equivalentes:** para dados binários, as distâncias euclidiana e manhattan produziram exatamente "
              "as mesmas partições, pois a euclidiana é a raiz quadrada da manhattan e a ordem entre as distâncias é "
              "preservada."),
        ("h", "4.4 Grupo do Integrante"),
        ("p", f"O autor consta no conjunto de dados como \"{AUTOR_DATASET}\" e foi alocado no **grupo {G_HEN_KM} pelo "
              f"K-Means** e no **grupo {G_HEN_AG} pelo aglomerativo**, isto é, no perfil de ecléticos que jogam muito. Ele "
              f"gosta de {int(hen.sum())} dos 42 estilos, perto da média do grupo ({f2(GOSTOS_KM[G_HEN_KM], 1)})."),
        ("p", "As características predominantes do grupo se confirmam no seu perfil: gosta de Rock, Survival, RPG, "
              "Estratégia, Puzzle, Simulação e Tabuleiro, que estão entre os maiores valores do centroide, e de Luta, um "
              "dos atributos mais acima da média. No aglomerativo, o grupo também se destaca em Samba, Pagode, MPB e MMORPG, "
              "de que o autor também gosta. Ele não gosta de Brega, K-Pop e House, os valores mais baixos do centroide. A "
              f"principal divergência é Pop, de que o autor não gosta, apesar de o centroide indicar {v(k, 0, 'Pop')}. A "
              f"silhueta individual é positiva nos dois algoritmos ({f2(SIL_HEN_KM, 3)} e {f2(SIL_HEN_AG, 3)}), o que "
              "indica uma alocação coerente."),
        ("h", "4.5 Recomendação para Quem Gosta de Brega"),
        ("p", f"Brega é o estilo menos popular ({N_BREGA} das {N} pessoas). Para a recomendação, identificou-se o grupo "
              "cujo centroide tem o maior valor de Brega e, nele, o estilo de jogo de maior valor."),
        ("p", f"**K-Means:** o grupo 0 tem o maior valor de Brega ({v(k, 0, 'Brega')}, contra {v(k, 1, 'Brega')} e "
              f"{v(k, 2, 'Brega')}) e reúne {BREGA_KM[0]} das {N_BREGA} pessoas que gostam de Brega. O jogo de maior valor é "
              f"**Survival** ({v(k, 0, 'Survival')}), seguido de RPG, Estratégia e Puzzle ({v(k, 0, 'RPG')})."),
        ("p", f"**Aglomerativo:** também é o grupo 0 (Brega = {v(a, 0, 'Brega')}; {BREGA_AG[0]} das {N_BREGA} pessoas). "
              f"Survival e RPG empatam como maior valor ({v(a, 0, 'RPG')}), mas RPG é o jogo mais acima da média no "
              "grupo; por isso, recomenda-se **RPG**."),
        ("p", f"O grupo 2 também tem valor relativamente alto de Brega ({v(k, 2, 'Brega')}) e prefere Cartas e Tabuleiro. "
              f"Consultando os dados diretamente, sem agrupamento, os jogos mais comuns entre as {ref['n_gosta_brega']} "
              "pessoas que gostam de Brega são Simulação e Tabuleiro "
              f"({100 * ref['jogos_mais_comuns_entre_quem_gosta_brega']['Simulação']:.0f}% cada). A diferença mostra que a "
              "recomendação por grupos reflete o perfil geral do grupo, já que quem gosta de Brega é minoria em todos eles."),
        ("h", "4.6 Recomendação para Quem Não Gosta de Tabuleiro"),
        ("p", "Neste caso, procurou-se o grupo cujo centroide tem o menor valor de Tabuleiro e, nele, o estilo musical de "
              "maior valor."),
        ("p", f"**K-Means:** o grupo 1 tem o menor valor de Tabuleiro ({v(k, 1, 'Tabuleiro')}, contra "
              f"{v(k, 0, 'Tabuleiro')} e {v(k, 2, 'Tabuleiro')}) e reúne {NTAB_KM[1]} das {N_NTAB} pessoas que não gostam "
              f"de Tabuleiro. O estilo de maior valor é **Rock** ({v(k, 1, 'Rock')}); os mais característicos do grupo são "
              f"Metal ({v(k, 1, 'Metal')}) e Indie ({v(k, 1, 'Indie')})."),
        ("p", f"**Aglomerativo:** também é o grupo 1 (Tabuleiro = {v(a, 1, 'Tabuleiro')}; {NTAB_AG[1]} das {N_NTAB} "
              f"pessoas), com Rock ainda mais alto ({v(a, 1, 'Rock')}). Recomenda-se **Rock**, com Metal e Indie como "
              "alternativas."),
        ("p", "Aqui os dois algoritmos concordam, e a recomendação coincide com a consulta direta aos dados: Rock é o estilo "
              "de que mais gostam as pessoas que não gostam de Tabuleiro "
              f"({100 * ref['musicas_mais_comuns_entre_quem_nao_gosta_tabuleiro']['Rock']:.0f}% delas). A Tabela 8 resume "
              "as recomendações."),
        tab_recomendacoes(),

        ("h", "5. Conclusão"),
        ("p", "O K-Means e o agrupamento aglomerativo com ligação de Ward, ambos com três grupos, produziram partições "
              f"muito parecidas ({pct(CONC)} de concordância; ARI de {f2(ARI, 3)}) e perfis com interpretação clara. As "
              "diferenças se concentram na fronteira entre os dois primeiros perfis, o que é coerente com os algoritmos: o "
              "K-Means otimiza uma partição global e revisável, enquanto o aglomerativo constrói os grupos por fusões "
              "locais e definitivas."),
        ("p", "Os experimentos mostraram que a escolha dos parâmetros importa: as ligações single e average geraram "
              "partições degeneradas, e o K-Means com uma só inicialização é instável. Também mostraram que as métricas "
              "internas devem ser analisadas em conjunto com o tamanho e os centroides dos grupos. Por fim, as silhuetas "
              "baixas indicam que os gostos formam um contínuo, e os grupos devem ser vistos como perfis aproximados, "
              "adequados a recomendações gerais como as feitas neste trabalho."),

        ("h", "6. Declaração de Uso de Inteligência Artificial"),
        ("p", "Durante a elaboração deste trabalho, foi utilizado o assistente de inteligência artificial Claude (modelo "
              "Claude Opus 5.5, da Anthropic), por meio da ferramenta Claude Code, para escrever o código em Python "
              "(scikit-learn, pandas e matplotlib) que executa os experimentos, armazena os grupos e centroides e gera as "
              "figuras; para executar o código e organizar os resultados; e para redigir a versão inicial deste texto e "
              "formatá-lo. A escolha das configurações, a interpretação dos grupos e as recomendações foram revisadas pelo "
              "autor, que se responsabiliza pelo conteúdo final. Nenhum dado externo foi usado para completar o conjunto "
              "de dados."),

        ("h", "7. Referências"),
        ("ref", "ARTHUR, D.; VASSILVITSKII, S. k-means++: the advantages of careful seeding. *In*: ANNUAL ACM-SIAM "
                "SYMPOSIUM ON DISCRETE ALGORITHMS, 18., 2007, New Orleans. **Proceedings** [...]. Philadelphia: SIAM, 2007. "
                "p. 1027-1035."),
        ("ref", "MACQUEEN, J. Some methods for classification and analysis of multivariate observations. *In*: BERKELEY "
                "SYMPOSIUM ON MATHEMATICAL STATISTICS AND PROBABILITY, 5., 1967, Berkeley. **Proceedings** [...]. "
                "Berkeley: University of California Press, 1967. v. 1, p. 281-297."),
        ("ref", "PEDREGOSA, F. *et al.* Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, "
                "[*s. l.*], v. 12, p. 2825-2830, 2011."),
        ("ref", "ROUSSEEUW, P. J. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. "
                "**Journal of Computational and Applied Mathematics**, [*s. l.*], v. 20, p. 53-65, 1987."),
        ("ref", "WARD, J. H. Hierarchical grouping to optimize an objective function. **Journal of the American Statistical "
                "Association**, [*s. l.*], v. 58, n. 301, p. 236-244, 1963."),
    ]


# ==========================================================================
# Marcação simples (**negrito**, *itálico*)
# ==========================================================================
def md_xml(txt: str) -> str:
    t = txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"\*(.+?)\*", r"<i>\1</i>", t)
    return t.replace("\n", "<br/>")


def md_runs(txt: str):
    for p in re.split(r"(\*\*.+?\*\*|\*.+?\*)", txt):
        if not p:
            continue
        if p.startswith("**"):
            yield p[2:-2], True, False
        elif p.startswith("*"):
            yield p[1:-1], False, True
        else:
            yield p, False, False


# ==========================================================================
# PDF
# ==========================================================================
def gerar_pdf(blocos):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.fonts import addMapping
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import CondPageBreak, Image, KeepTogether, Paragraph, SimpleDocTemplate, Table, TableStyle

    fd = Path("C:/Windows/Fonts")
    for nome, arq in (("Times", "times.ttf"), ("Times-B", "timesbd.ttf"), ("Times-I", "timesi.ttf"), ("Times-BI", "timesbi.ttf")):
        pdfmetrics.registerFont(TTFont(nome, str(fd / arq)))
    addMapping("Times", 0, 0, "Times")
    addMapping("Times", 1, 0, "Times-B")
    addMapping("Times", 0, 1, "Times-I")
    addMapping("Times", 1, 1, "Times-BI")

    W, H = A4
    ML, MR, MT, MB = 3 * cm, 3 * cm, 3.5 * cm, 2.5 * cm
    LARG = W - ML - MR

    def S(nome, **kw):
        return ParagraphStyle(nome, **{"fontName": "Times", "fontSize": 12, "leading": 14, **kw})

    st = {
        "titulo": S("titulo", fontName="Times-B", fontSize=16, leading=19, alignment=TA_CENTER, spaceBefore=12),
        "autor": S("autor", fontName="Times-B", alignment=TA_CENTER, spaceBefore=12),
        "afil": S("afil", alignment=TA_CENTER, spaceBefore=12),
        "afil2": S("afil2", alignment=TA_CENTER),
        "email": S("email", fontSize=10, leading=12, alignment=TA_CENTER, spaceBefore=6, spaceAfter=6),
        "abs": S("abs", fontName="Times-I", alignment=TA_JUSTIFY, leftIndent=0.8 * cm, rightIndent=0.8 * cm,
                 spaceBefore=6, spaceAfter=6),
        "h": S("h", fontName="Times-B", fontSize=13, leading=15, spaceBefore=12, keepWithNext=1),
        "p": S("p", alignment=TA_JUSTIFY, firstLineIndent=1.27 * cm, spaceBefore=6),
        "cap": S("cap", fontName="Times-B", fontSize=10, leading=12, alignment=TA_CENTER, spaceBefore=10, spaceAfter=4),
        "fonte": S("fonte", fontSize=10, leading=12, alignment=TA_CENTER, spaceBefore=4, spaceAfter=6),
        "tab": S("tab", fontSize=10, leading=11.5, alignment=TA_LEFT),
        "tabc": S("tabc", fontSize=10, leading=11.5, alignment=TA_CENTER),
        "tabh": S("tabh", fontName="Times-B", fontSize=10, leading=11.5, alignment=TA_CENTER),
        "ref": S("ref", alignment=TA_JUSTIFY, spaceBefore=6),
    }

    doc = SimpleDocTemplate(str(SAIDA_PDF), pagesize=A4, leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
                            title=TITULO, author=AUTOR, subject="GEX1225 – Inteligência Artificial")
    story = [
        Paragraph(TITULO, st["titulo"]),
        Paragraph(f"{AUTOR}<super>1</super>", st["autor"]),
        Paragraph(f"<super>1</super>{AFILIACAO}", st["afil"]),
        Paragraph(DISCIPLINA, st["afil2"]),
        Paragraph(EMAIL, st["email"]),
        Paragraph(f"<b>Abstract.</b> {md_xml(ABSTRACT)}", st["abs"]),
        Paragraph(f"<b>Resumo.</b> {md_xml(RESUMO_PT)}", st["abs"]),
    ]
    nfig = ntab = 0
    for b in blocos:
        tipo = b[0]
        if tipo == "h":
            story.append(Paragraph(md_xml(b[1]), st["h"]))
        elif tipo == "p":
            story.append(Paragraph(md_xml(b[1]), st["p"]))
        elif tipo == "ref":
            story.append(Paragraph(md_xml(b[1]), st["ref"]))
        elif tipo == "fig":
            nfig += 1
            img = Image(str(FIG / b[1]))
            esc = LARG / img.imageWidth
            img.drawWidth, img.drawHeight = img.imageWidth * esc, img.imageHeight * esc
            story.append(KeepTogether([Paragraph(f"Figura {nfig} – {md_xml(b[2])}", st["cap"]), img,
                                       Paragraph(FONTE, st["fonte"])]))
        elif tipo == "tab":
            ntab += 1
            titulo, cab, linhas, largs = b[1], b[2], b[3], b[4]
            tot = sum(largs)
            cw = [x / tot * min(LARG, tot * cm) for x in largs]
            dados = [[Paragraph(md_xml(c), st["tabh"]) for c in cab]]
            for ln in linhas:
                dados.append([Paragraph(md_xml(str(c)), st["tab"] if (j == 0 or len(str(c)) > 16) else st["tabc"])
                              for j, c in enumerate(ln)])
            t = Table(dados, colWidths=cw, repeatRows=1, hAlign="CENTER")
            t.setStyle(TableStyle([
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]))
            cap = Paragraph(f"Tabela {ntab} – {md_xml(titulo)}", st["cap"])
            fonte = Paragraph(FONTE, st["fonte"])
            story += [CondPageBreak(10 * cm), cap, t, fonte] if len(linhas) > 20 else [KeepTogether([cap, t, fonte])]
    doc.build(story)
    return nfig, ntab


# ==========================================================================
# DOCX
# ==========================================================================
def gerar_docx(blocos):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor

    d = Document()
    sec = d.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin, sec.right_margin, sec.top_margin, sec.bottom_margin = Cm(3), Cm(3), Cm(3.5), Cm(2.5)

    normal = d.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    rpr = normal.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.append(rf)
    for att in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(att), "Times New Roman")
    pf = normal.paragraph_format
    pf.line_spacing = 1.0
    pf.space_before = pf.space_after = Pt(0)

    def par(alinh=WD_ALIGN_PARAGRAPH.JUSTIFY, antes=6, depois=0):
        p = d.add_paragraph()
        p.alignment = alinh
        p.paragraph_format.space_before = Pt(antes)
        p.paragraph_format.space_after = Pt(depois)
        return p

    def add(p, txt, tam=None, negrito=None, italico=None):
        for t, b, i in md_runs(txt):
            r = p.add_run(t)
            r.bold = True if (b or negrito) else None
            r.italic = True if (i or italico) else None
            if tam:
                r.font.size = Pt(tam)
        return p

    def borda(el, lado, sz):
        e = OxmlElement(f"w:{lado}")
        e.set(qn("w:val"), "single" if sz else "nil")
        if sz:
            e.set(qn("w:sz"), str(sz))
            e.set(qn("w:color"), "000000")
        el.append(e)

    # Cabeçalho do artigo
    add(par(WD_ALIGN_PARAGRAPH.CENTER, 12), TITULO, 16, negrito=True)
    p = par(WD_ALIGN_PARAGRAPH.CENTER, 12)
    p.add_run(AUTOR).bold = True
    r = p.add_run("1")
    r.bold = True
    r.font.superscript = True
    p = par(WD_ALIGN_PARAGRAPH.CENTER, 12)
    p.add_run("1").font.superscript = True
    p.add_run(AFILIACAO)
    par(WD_ALIGN_PARAGRAPH.CENTER, 0).add_run(DISCIPLINA)
    add(par(WD_ALIGN_PARAGRAPH.CENTER, 6, 6), EMAIL, 10)
    for rotulo, txt in (("Abstract.", ABSTRACT), ("Resumo.", RESUMO_PT)):
        p = par(antes=6, depois=6)
        p.paragraph_format.left_indent = p.paragraph_format.right_indent = Cm(0.8)
        r = p.add_run(rotulo + " ")
        r.bold = r.italic = True
        add(p, txt, italico=True)

    nfig = ntab = 0
    for b in blocos:
        tipo = b[0]
        if tipo == "h":
            p = par(WD_ALIGN_PARAGRAPH.LEFT, 12)
            p.paragraph_format.keep_with_next = True
            add(p, b[1], 13, negrito=True)
        elif tipo == "p":
            p = par()
            p.paragraph_format.first_line_indent = Cm(1.27)
            add(p, b[1])
        elif tipo == "ref":
            add(par(), b[1])
        elif tipo == "fig":
            nfig += 1
            p = par(WD_ALIGN_PARAGRAPH.CENTER, 10, 4)
            p.paragraph_format.keep_with_next = True
            add(p, f"Figura {nfig} – {b[2]}", 10, negrito=True)
            p = par(WD_ALIGN_PARAGRAPH.CENTER, 0)
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(FIG / b[1]), width=Cm(15))
            add(par(WD_ALIGN_PARAGRAPH.CENTER, 4, 6), FONTE, 10)
        elif tipo == "tab":
            ntab += 1
            titulo, cab, linhas, largs = b[1], b[2], b[3], b[4]
            p = par(WD_ALIGN_PARAGRAPH.CENTER, 10, 4)
            p.paragraph_format.keep_with_next = True
            add(p, f"Tabela {ntab} – {titulo}", 10, negrito=True)
            t = d.add_table(rows=1 + len(linhas), cols=len(cab))
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            esc = min(15.0, sum(largs)) / sum(largs)
            for i, row in enumerate([cab] + linhas):
                for j, val in enumerate(row):
                    c = t.cell(i, j)
                    c.width = Cm(largs[j] * esc)
                    cp = c.paragraphs[0]
                    cp.paragraph_format.space_before = cp.paragraph_format.space_after = Pt(1)
                    cp.alignment = (WD_ALIGN_PARAGRAPH.CENTER if (i == 0 or (j > 0 and len(str(val)) <= 16))
                                    else WD_ALIGN_PARAGRAPH.LEFT)
                    add(cp, str(val), 10, negrito=(i == 0) or None)
            bd = OxmlElement("w:tblBorders")
            for lado, sz in (("top", 8), ("left", 0), ("bottom", 8), ("right", 0), ("insideH", 0), ("insideV", 0)):
                borda(bd, lado, sz)
            t._tbl.tblPr.append(bd)
            for c in t.rows[0].cells:
                tb = OxmlElement("w:tcBorders")
                borda(tb, "bottom", 8)
                c._tc.get_or_add_tcPr().append(tb)
            t.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
            add(par(WD_ALIGN_PARAGRAPH.CENTER, 4, 6), FONTE, 10)

    d.core_properties.author = AUTOR
    d.core_properties.title = TITULO
    d.save(str(SAIDA_DOCX))


def main():
    blocos = conteudo()
    nfig, ntab = gerar_pdf(blocos)
    gerar_docx(blocos)
    print(f"PDF:  {SAIDA_PDF}  ({nfig} figuras, {ntab} tabelas)")
    print(f"DOCX: {SAIDA_DOCX}")


if __name__ == "__main__":
    main()
