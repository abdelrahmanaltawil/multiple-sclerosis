# env imports
import sys
from pathlib import Path
from dash import html, Dash, dcc
import plotly.express as px

# setup work directory
src_path = Path(__file__).parents[2].__str__()
sys.path.append(src_path) if src_path not in sys.path else None

# local imports
from dashboard.layout.components.feature_dropdown import features_dropdown
from dashboard.layout.components.feature_scatter import features_scatter

def options_section() -> html.Div:

    dropdown = html.Div(
        children=[
            # options
            # option_sidebar()
            features_dropdown(),

            # graphs
            features_scatter()
        ],
        className="options-layout"
    )

    return dropdown
    


if __name__ == "__main__":

    # init dash and load css
    app = Dash(__name__, assets_folder=Path(src_path + '/dashboard/resources/'))

    # create layout
    app.layout = options_section()

    # run dash
    app.run_server(debug=True)
    