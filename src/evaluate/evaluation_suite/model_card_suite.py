from .model_card_data import ModelCardResult
from evaluate import EvaluationSuite
from evaluate.visualization import radar_plot
import base64
import io
import pandas as pd


def fig_to_base64str(fig) -> str:
    """Converts a Matplotlib figure to a base64 string encoding.

    Args:
    fig: A matplotlib Figure.

    Returns:
    A base64 encoding of the figure.
    """
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches='tight', format='png')
    fig_str = base64.b64encode(buf.getbuffer().tobytes()).decode('ascii')
    return "![Hello World](data:image/png;base64,{})".format(fig_str)


def results_to_markdown(results, drop=["data_preprocessor"]):
    """
    Converts results from `EvaluationSuite` to Markdown table

    Args:
        results (`dict`): results from `EvaluationSuite`
    
    Return:
        A `str`. Containing Markdown table
    """
    if len(results) < 1:
        return ''
    columns = [column for column in results[0].keys() if column not in drop]
    df = pd.DataFrame(results, columns=columns)
    return df.to_markdown(index=False)


def results_to_dataclass_helper(results, header=None, graphic=None):
    """ Adds `EvaluationSuite` results to `ModelCardResult` dataclass

    Args:
        results (`dict`): results from `EvaluationSuite`
    
    Return:
        A `ModelCardResult`. Containing evaluation results
    """
    header = f"### {header or 'Evaluation Suite Results'}"
    table = results_to_markdown(results)
    if graphic is not None:
        graphic = fig_to_base64str(graphic)
    return ModelCardResult(header, table, graphic)


class ModelCardSuiteResults(EvaluationSuite):
    """ Class for custom a EvaluationSuite that returns results
    in dataclass format to be added to Model Card with custom
    graphics.
    """
    def __init__(self, name, header=None):
        super().__init__(name)
        self.header = header or f"Evaluation Suite Results: {self.name}"
        self.result_keys = []
        self.summary = ""

    def process_results(self, results):
        """Preprocess results to be used as `plot_results` input
        Args:
            results (`dict`): results from `EvaluationSuite`
    
        Return:
            A `list`. Containing dict of processed results.
        """
        result_dict = {}
        for result in results:
            for key in self.result_keys:
                value = result.get(key)
                if value is not None:
                    name = f"{key} {result.get('task_name', '')}"
                    result_dict[name] = value
        return [result_dict]

    def plot_results(self, results, model_or_pipeline):
        """Plot to be associated with EvaluationSuite and added to
        model card

        Args:
            results (`dict`): processed results from `EvaluationSuite`
            model_or_pipeline ('str'): name of model or pipeline 
    
        Return:
            A `matplotlib.plot`
        """
        results = self.process_results(results)
        graphic = radar_plot(results, [model_or_pipeline])
        return graphic
    
    def run(self, model_or_pipeline, plot=True):
        results = super().run(model_or_pipeline)
        if isinstance(model_or_pipeline, str):
            model_name = model_or_pipeline
        else:
            model_name = model_or_pipeline.__class__.__name__
        graphic = self.plot_results(results, model_name) if plot else None
        mc_results = results_to_dataclass_helper(results, header=self.header, graphic=graphic)
        return results, mc_results