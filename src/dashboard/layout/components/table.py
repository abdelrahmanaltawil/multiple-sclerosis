# env imports
import sys
from pathlib import Path
from dash import html, Dash, dash_table
import pandas as pd
import plotly

# setup work directory
src_path = Path(__file__).parents[3].__str__()
sys.path.append(src_path) if src_path not in sys.path else None

# local imports

def table(data: pd.DataFrame) -> html.Div:

    dropdown = html.Div(
        children=[
            html.P("Data Set", className="graph-title"),

            html.Div(
                dash_table.DataTable(
                    data=data.to_dict('records'), 
                    columns=[{"name": column, "id": column} for column in data.columns],
                    style_header={
                        "background-color": "lightblue"
                        },
                    style_cell={
                        "textAlign": "left"
                        }          
                    ),
                className="table"
                ), 
            ],
        className="graph-flex"
        )

    return dropdown
    


if __name__ == "__main__":

    # init dash and load css
    app = Dash(__name__, assets_folder=Path(src_path + '/dashboard/resources/'))

    # create layout
    app.layout = table(data=plotly.data.iris())

    # run dash
    app.run_server(debug=True)