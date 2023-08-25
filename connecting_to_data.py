from dash import Dash, html, dash_table
import pandas as pd

df=pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder2007.csv')

#init the app
app = Dash(__name__)

#app layout
app.layout = html.Div([
    html.Div(children='My First App with Data'),
    dash_table.DataTable(data=df.to_dict('records'), page_size=10)
])

#Run the app
if __name__ == '__main__':
    app.run(debug=True)





