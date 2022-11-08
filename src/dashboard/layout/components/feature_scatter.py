# env imports
import sys
from pathlib import Path
from dash import html, Dash, dcc
import plotly.express as px

# setup work directory
src_path = Path(__file__).parents[3].__str__()
sys.path.append(src_path) if src_path not in sys.path else None

# local imports

def features_scatter() -> html.Div:

    dropdown = html.Div(
        children=[
            html.P(
                children=["Scatter Plot"], 
                className="graph-title",
                ),
            
            dcc.Graph(figure=px.scatter(), className="graph")
        ],
        className="graph-flex"
    )


    return dropdown
    


if __name__ == "__main__":

    # init dash and load css
    app = Dash(__name__, assets_folder=Path(src_path + '/dashboard/resources/'))

    # create layout
    app.layout = features_scatter()

    # run dash
    app.run_server(debug=True)