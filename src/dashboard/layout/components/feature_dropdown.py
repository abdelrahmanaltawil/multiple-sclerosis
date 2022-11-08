# env imports
import sys
from pathlib import Path
from dash import html, Dash, dcc

# setup work directory
src_path = Path(__file__).parents[3].__str__()
sys.path.append(src_path) if src_path not in sys.path else None

# local imports

def features_dropdown() -> html.Div:

    dropdown = html.Div(
        children=[
            html.Label(
                children=["Features"], 
                className="label",
                ),
            
            dcc.Dropdown(
                options=[
                    {"label": "item1", "value": 1}, 
                    {"label": "item2", "value": 2}
                    ],
                className="dropdown"
                )
        ],
        className="label-input-flex"
    )


    return dropdown
    


if __name__ == "__main__":

    # init dash and load css
    app = Dash(__name__, assets_folder=Path(src_path + '/dashboard/resources/'))

    # create layout
    app.layout = features_dropdown()

    # run dash
    app.run_server(debug=True)