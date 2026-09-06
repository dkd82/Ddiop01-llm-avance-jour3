import logging
import queue
import threading
import time
import gradio as gr
import pandas as pd
from deal_agent_framework import DealAgentFramework
from log_utils import reformat
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(override=True)


class QueueHandler(logging.Handler):
    # Gestionnaire de logs qui pousse chaque message dans une file (queue),
    # afin de les afficher en direct dans l'interface Gradio.
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def html_for(log_data):
    # Construit un bloc HTML affichant les 18 dernières lignes de log dans une zone défilante
    output = "<br>".join(log_data[-18:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border: 1px solid #ccc; background-color: #222229; padding: 10px;">
    {output}
    </div>
    """


def setup_logging(log_queue):
    # Branche le QueueHandler sur le logger racine
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class App:
    TABLE_HEADERS = ["Offres trouvées jusqu'ici", "Prix", "Estimation", "Remise", "URL"]

    def __init__(self):
        self.agent_framework = None

    def get_agent_framework(self):
        # Crée (paresseusement) le cadre d'agents et le réutilise ensuite
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    def run(self):
        with gr.Blocks(title="Le Juste Prix", fill_width=True) as ui:
            log_data = gr.State([])

            def table_for(opps):
                # Transforme les opportunités en lignes de tableau (description, prix, estimation, remise, URL)
                rows = [
                    [
                        opp.deal.product_description,
                        f"${opp.deal.price:.2f}",
                        f"${opp.estimate:.2f}",
                        f"${opp.discount:.2f}",
                        opp.deal.url,
                    ]
                    for opp in opps
                ]
                return pd.DataFrame(rows, columns=self.TABLE_HEADERS)

            def update_output(log_data, log_queue, result_queue):
                # Générateur qui diffuse en continu les logs et le résultat final vers l'interface
                initial_result = table_for(self.get_agent_framework().memory)
                final_result = None
                while True:
                    try:
                        message = log_queue.get_nowait()
                        log_data.append(reformat(message))
                        yield log_data, html_for(log_data), final_result or initial_result
                    except queue.Empty:
                        try:
                            final_result = result_queue.get_nowait()
                            yield log_data, html_for(log_data), final_result or initial_result
                        except queue.Empty:
                            if final_result is not None:
                                break
                            time.sleep(0.1)

            def get_initial_plot():
                # Graphique affiché pendant le chargement de la base vectorielle
                fig = go.Figure()
                fig.update_layout(
                    title="Chargement de la base vectorielle...",
                    height=400,
                )
                return fig

            def get_plot():
                # Construit la visualisation 3D des produits (projection t-SNE)
                documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=800)
                # Nuage de points 3D
                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=vectors[:, 0],
                            y=vectors[:, 1],
                            z=vectors[:, 2],
                            mode="markers",
                            marker=dict(size=2, color=colors, opacity=0.7),
                        )
                    ]
                )

                fig.update_layout(
                    scene=dict(
                        xaxis_title="x",
                        yaxis_title="y",
                        zaxis_title="z",
                        aspectmode="manual",
                        aspectratio=dict(x=2.2, y=2.2, z=1),  # Allonge l'axe x
                        camera=dict(
                            eye=dict(x=1.6, y=1.6, z=0.8)  # Position de la caméra
                        ),
                    ),
                    height=400,
                    margin=dict(r=5, b=1, l=5, t=2),
                )

                return fig

            def do_run():
                # Lance un cycle du cadre d'agents et renvoie le tableau des opportunités
                new_opportunities = self.get_agent_framework().run()
                table = table_for(new_opportunities)
                return table

            def run_with_logging(initial_log_data):
                # Exécute do_run() dans un thread, tout en diffusant les logs en temps réel
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                setup_logging(log_queue)

                def worker():
                    try:
                        result = do_run()
                    except Exception:
                        logging.exception("Erreur pendant l'exécution du cycle d'agents")
                        result = table_for(self.get_agent_framework().memory)
                    result_queue.put(result)

                thread = threading.Thread(target=worker)
                thread.start()

                for log_data, output, final_result in update_output(
                    initial_log_data, log_queue, result_queue
                ):
                    yield log_data, output, final_result

            def do_select(selected_index: gr.SelectData):
                # Au clic sur une ligne : envoie une alerte (notification) pour l'opportunité sélectionnée
                opportunities = self.get_agent_framework().memory
                row = selected_index.index[0]
                if row >= len(opportunities):
                    return
                opportunity = opportunities[row]
                self.get_agent_framework().planner.messenger.alert(opportunity)

            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:24px"><strong>Le Juste Prix</strong> — cadre d\'agents autonomes qui chasse les bonnes affaires</div>'
                )
            with gr.Row():
                gr.Markdown(
                    '<div style="text-align: center;font-size:14px">Un LLM propriétaire fine-tuné déployé sur Modal et un pipeline RAG avec un modèle de pointe collaborent pour envoyer des notifications push sur les meilleures offres en ligne.</div>'
                )
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    value=table_for(self.get_agent_framework().memory),
                    headers=self.TABLE_HEADERS,
                    wrap=True,
                    column_widths=[6, 1, 1, 1, 3],
                    row_count=10,
                    column_count=5,
                    max_height=400,
                )
            with gr.Row():
                with gr.Column(scale=1):
                    logs = gr.HTML()
                with gr.Column(scale=1):
                    plot = gr.Plot(value=get_plot(), show_label=False)

            # Au chargement de l'interface, on lance un cycle avec affichage des logs
            ui.load(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

            # Minuteur : relance un cycle toutes les 300 secondes (5 minutes)
            timer = gr.Timer(value=300, active=True)
            timer.tick(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe],
            )

            opportunities_dataframe.select(do_select)

        ui.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    App().run()
