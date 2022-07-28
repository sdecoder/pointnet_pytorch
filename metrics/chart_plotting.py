import altair as alt
import plotly
from vega_datasets import data
import pandas
import numpy

def _test_altair():
  cars = data.cars()
  base = alt.Chart(cars).mark_point().encode(
    x='Horsepower:Q',
    y='Miles_per_Gallon:Q',
  ).properties(
    width=150,
    height=150
  )

  chart = alt.hconcat(
    base.encode(color=alt.Color('Cylinders:Q', scale=alt.Scale(scheme='reds'))).properties(title='quantitative'),
    base.encode(color=alt.Color('Cylinders:O', scale=alt.Scale(scheme='reds'))).properties(title='ordinal'),
    base.encode(color='Cylinders:N').properties(title='nominal')

  )

  chart.save('chart.png')
  chart.show()
  pass


def _test_plotly():
  import plotly.graph_objects as go
  import numpy as np
  np.random.seed(1)

  N = 100
  x = np.random.rand(N)
  y = np.random.rand(N)
  colors = np.random.rand(N)
  sz = np.random.rand(N) * 30

  fig = go.Figure()
  fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode="markers",
    marker=go.scatter.Marker(
      size=sz,
      color=colors,
      opacity=0.6,
      colorscale="Viridis"
    )
  ))

  _unified_font_family = "Garamond"
  fig.update_layout(
    font_family=_unified_font_family,
    font_color="blue",
    font_size=10,
    title_font_family=_unified_font_family,
    title_font_color="red",
    legend_title_font_color="green"
  )

  fig.show()

  import os
  if not os.path.exists("images"):
    os.mkdir("images")

  print(f'[trace] exec@writing image')
  fig.write_image("images/fig1.png")
  print(f'[trace] writing image	done')
  pass


def _test_plotly_pandas():
  import plotly.express as px
  df = px.data.gapminder().query("country=='Canada'")
  fig = px.line(df, x="year", y="lifeExp", title='Life expectancy in Canada')
  fig.show()
  pass

def _worker_code():

  print(f'[trace] worker code')

  import pandas as pd
  import numpy as np

  '''
  df = px.data.gapminder().query("continent=='Oceania'")
  df.to_csv('test.csv')

  '''
  import plotly.express as px

  df = pd.read_csv('optlevel-runtime.csv', sep=',')
  df_lines = df.shape[0] / 3
  epochs = []
  for i in range(3):
    for j in range(int(df_lines)):
      epochs.append(j)
  df['Epoch'] = epochs
  print(f"[trace] df.size {df_lines}")
  print(df.to_string())
  fig = px.line(df, x="Epoch", y="Runtime", color='OptLevel')

  _unified_font_family = "Product Sans"
  fig.update_layout(
    font_family=_unified_font_family,
    font_color="blue",
    font_size=12,
    title_font_family=_unified_font_family,
    title_font_color="red",
    legend_title_font_color="green",
  )
  fig.update_layout(yaxis_range=[0, 200])

  _target_path = 'o1-runtime.png'
  fig.write_image(_target_path)

  pass


def _main():
  # _test_altair()
  # _test_plotly()
  _worker_code()
  pass


if __name__ == '__main__':
  _main()
  pass
