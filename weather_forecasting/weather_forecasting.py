import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv("DailyDelhiClimateTrain.csv")
#print(data.head())
#print(data.describe())
#print(data.info())

fig = px.line(data, 
              x="date",
              y="meantemp",
              title="Mean temperature in delhi over the years")
fig.show()

fig = px.line(data, 
              x="date",
              y="humidity")
fig.show()

fig = px.line(data, 
              x="date",
              y="wind_speed"
              )
fig.show()


