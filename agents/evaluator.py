"""
Module d'évaluation (parallélisé) pour la prédiction de prix.

La classe `Tester` exécute un prédicteur sur un jeu de données (objets `Item`), mesure l'erreur,
colore chaque prédiction (vert/orange/rouge) et trace deux graphiques de synthèse.
La fonction `evaluate(...)` est un raccourci pratique.
"""

import re
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import accumulate
import math
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor

# Codes de couleur ANSI pour colorer le texte affiché
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
# Les clés "red"/"orange"/"green" servent aussi de logique : à NE PAS traduire
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

WORKERS = 5          # Nombre de threads parallèles
DEFAULT_SIZE = 200   # Nombre d'exemples évalués par défaut


class Tester:
    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE, workers=WORKERS):
        self.predictor = predictor          # Fonction de prédiction à évaluer
        self.data = data                    # Jeu de données (objets Item)
        self.title = title or self.make_title(predictor)
        self.size = size
        self.titles = []
        self.guesses = []
        self.truths = []
        self.errors = []
        self.colors = []
        self.workers = workers

    @staticmethod
    def make_title(predictor) -> str:
        # Construit un titre lisible à partir du nom de la fonction
        return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    @staticmethod
    def post_process(value):
        # Extrait un nombre (le prix) de la sortie : retire "$" et virgules, isole le premier nombre
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", value)
            return float(match.group()) if match else 0
        else:
            return value

    def color_for(self, error, truth):
        # Couleur selon la qualité : vert (faible), orange (moyenne), rouge (importante)
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i):
        # Évalue un seul point de données d'indice i
        datapoint = self.data[i]
        value = self.predictor(datapoint)
        guess = self.post_process(value)
        truth = datapoint.price
        error = abs(guess - truth)
        color = self.color_for(error, truth)
        title = datapoint.title if len(datapoint.title) <= 40 else datapoint.title[:40] + "..."
        return title, guess, truth, error, color

    def chart(self, title):
        # Nuage de points "prix prédit vs prix réel"
        df = pd.DataFrame(
            {
                "truth": self.truths,
                "guess": self.guesses,
                "title": self.titles,
                "error": self.errors,
                "color": self.colors,
            }
        )

        # Texte de survol pré-formaté
        df["hover"] = [
            f"{t}\nPrédit=${g:,.2f} Réel=${y:,.2f}"
            for t, g, y in zip(df["title"], df["guess"], df["truth"])
        ]

        max_val = float(max(df["truth"].max(), df["guess"].max()))

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=title,
            labels={"truth": "Prix réel", "guess": "Prix prédit"},
            width=1000,
            height=800,
        )

        # Données de survol par trace (une couleur/catégorie = une trace)
        for tr in fig.data:
            mask = df["color"] == tr.name
            tr.customdata = df.loc[mask, ["hover"]].to_numpy()
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"
            tr.marker.update(size=6)

        # Ligne de référence y = x (prédiction parfaite)
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(width=2, dash="dash", color="deepskyblue"),
                name="y = x",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.update_xaxes(range=[0, max_val])
        fig.update_yaxes(range=[0, max_val])
        fig.update_layout(showlegend=False)
        fig.show()

    def error_trend_chart(self):
        # Évolution de l'erreur moyenne cumulée, avec intervalle de confiance à 95 %
        n = len(self.errors)

        # Moyenne et écart-type glissants (en Python pur)
        running_sums = list(accumulate(self.errors))
        x = list(range(1, n + 1))
        running_means = [s / i for s, i in zip(running_sums, x)]

        running_squares = list(accumulate(e * e for e in self.errors))
        running_stds = [
            math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0
            for i, sq_sum, mean in zip(x, running_squares, running_means)
        ]

        # Intervalle de confiance à 95 % de la moyenne
        ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, running_stds)]
        upper = [m + c for m, c in zip(running_means, ci)]
        lower = [m - c for m, c in zip(running_means, ci)]

        # Graphique
        fig = go.Figure()

        # Bande ombrée représentant l'intervalle de confiance
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="IC 95%",
            )
        )

        # Courbe principale, avec texte de survol montrant l'intervalle de confiance
        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_means,
                mode="lines",
                line=dict(width=3, color="firebrick"),
                name="Erreur moyenne cumulée",
                customdata=list(
                    zip(
                        ci,
                    )
                ),
                hovertemplate=(
                    "n=%{x}<br>"
                    "Erreur moyenne=$%{y:,.2f}<br>"
                    "±IC 95%=$%{customdata[0]:,.2f}<extra></extra>"
                ),
            )
        )

        # Titre avec les statistiques finales
        final_mean = running_means[-1]
        final_ci = ci[-1]
        title = f"{self.title} — Erreur : ${final_mean:,.2f} ± ${final_ci:,.2f}"

        fig.update_layout(
            title=title,
            xaxis_title="Nombre de points de données",
            yaxis_title="Erreur absolue moyenne ($)",
            width=1000,
            height=360,
            template="plotly_white",
            showlegend=False,
        )

        fig.show()

    def report(self):
        # Calcule les métriques globales et affiche les deux graphiques
        average_error = sum(self.errors) / self.size
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100
        title = f"{self.title} — résultats<br><b>Erreur :</b> ${average_error:,.2f} <b>MSE :</b> {mse:,.0f} <b>r² :</b> {r2:.1f}%"
        self.error_trend_chart()
        self.chart(title)

    def run(self):
        # Lance les prédictions en parallèle (pool de threads) puis affiche le rapport
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for title, guess, truth, error, color in tqdm(
                ex.map(self.run_datapoint, range(self.size)), total=self.size
            ):
                self.titles.append(title)
                self.guesses.append(guess)
                self.truths.append(truth)
                self.errors.append(error)
                self.colors.append(color)
                # Affiche l'erreur de chaque prédiction, colorée selon sa qualité
                print(f"{COLOR_MAP[color]}${error:.0f} ", end="")
        self.report()


def evaluate(function, data, size=DEFAULT_SIZE, workers=WORKERS):
    # Raccourci pratique : crée un Tester et lance l'évaluation
    Tester(function, data, size=size, workers=workers).run()
