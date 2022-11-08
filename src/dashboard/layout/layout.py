# env imports
import sys
from pathlib import Path
from dash import html, Dash
import plotly

# setup work directory
src_path = Path(__file__).parents[2].__str__()
sys.path.append(src_path) if src_path not in sys.path else None

# local imports
# from dashboard.layout.main import main_section
from dashboard.layout.options import options_section
from dashboard.layout.components.table import table
from dashboard.layout.components.features_boxplot import features_box

def create_layout() -> html.Div:
    '''
    Sets the layout of the dash application. The layout is provided by `css` file
    '''

    layout = html.Div(
        children=[
            html.Header(
                children=[
                    "Data Dashboard"
                ],
                contentEditable="True",
                className="header" 
            ),
            
            html.Main(
                children=[
                    # table
                    table(plotly.data.iris()),
 
                    # graphs
                    features_box(),

                    # options
                    options_section()
                ],
                className="main-layout"
            ), 

            # main_section(),
            
            html.Footer(
                children=[
                    html.P(u"\u00A9"+" Altawil")
                ],
                className="footer"
            )
        ],
        className="parent"
    )


    return layout






if __name__ == "__main__":

    # init dash and load css
    app = Dash(__name__, assets_folder=Path(src_path + '/dashboard/resources/'))

    # create layout
    app.layout = create_layout()

    # run dash
    app.run_server(debug=True)