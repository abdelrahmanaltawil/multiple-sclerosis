# env imports
import sys
from pathlib import Path
from dash import html, Dash, dcc
import plotly

# setup work directory
src_path = Path(__file__).parents[2].__str__()
sys.path.append(src_path) if src_path not in sys.path else None

# local imports
from dashboard.layout.options import options_section
from dashboard.layout.components.table import table
from dashboard.layout.components.features_boxplot import features_box


def main_section() -> html.Div:

    dropdown = html.Div(
        children=[
            # table
            table(plotly.data.iris()),
 
            # graphs
            features_box(),

            # options
            options_section()
        ],
        className="main-layout"
    )

    return dropdown
    


if __name__ == "__main__":

    # init dash and load css
    app = Dash(__name__, assets_folder=Path(src_path + '/dashboard/resources/'))

    # create layout
    app.layout = main_section()

    # run dash
    app.run_server(debug=True)

