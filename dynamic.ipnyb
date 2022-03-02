import datetime
import pickle
# import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import dash
from dash import dcc
from dash import html
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

with open('resources_msgs.pkl', 'rb') as f:
    data_resources = pickle.load(f)

with open('events_msgs.pkl', 'rb') as f:
    data_events = pickle.load(f)

with open("messages.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for key, resource in data_resources.items():
        writer.writerows(resource)
    for key, event in data_events.items():
        writer.writerows(event)

def generate_bins(bin_size: float = 1) -> tuple:
    """
    generates bins over the selected time range based on bin_size

    Args:
        bin_size: parameter in hours

    Returns:
        tuple containing bins in list(datetime) for df operations and list(str) for x-axis labels
    """

    if bin_size not in [0.5, 1.0, 1.5, 2, 3, 4, 6, 8, 12]:
        raise

    start_time = datetime(2020, 4, 6, 0, 0, 0)
    end_time = datetime(2020, 4, 10, 12, 0, 0)
    start_label = start_time.strftime("%a %I %p")

    current_time = start_time + timedelta(hours=bin_size)
    slider_label = current_time.strftime("%I %p")

    time_bins = [start_time, current_time]
    slider_bins = [start_label, slider_label]
    axis_bins = [start_time.strftime("%a %d %I %p"), current_time.strftime("%a %d %I %p")]

    while current_time < end_time:
        current_time += timedelta(hours=bin_size)
        if current_time.hour == 0:
            slider_label = current_time.strftime("%a %I %p")
        else:
            slider_label = current_time.strftime("%I %p")

        time_bins.append(current_time)
        slider_bins.append(slider_label)
        axis_bins.append(current_time.strftime("%a %d %I %p"))

    return time_bins, slider_bins, axis_bins


time_bins, slider_bins, label_bins = generate_bins(bin_size=3)
slider_marks = {key: {'label': value} for key, value in enumerate(slider_bins)}


# todo Possible optimization?
def bin_df(row: pd.Series) -> str:
    t_prev = time_bins[0]
    for index in range(1, len(time_bins)):
        if t_prev < row['Timestamp'] <= time_bins[index]:
            return label_bins[index]
        t_prev = time_bins[index]


colnames = ['Timestamp', 'Area', 'User', 'Message', 'Category', 'Type']
df = pd.read_csv("messages.csv", names=colnames, header=None)
df['Timestamp'] = pd.to_datetime(df.Timestamp)
# df['TimestampBins'] = df.apply(bin_df, axis=1) # todo Possible optimization?

categories = df['Category'].unique()

water = df.loc[df['Category'] == 'Water']
energy = (df[df['Category'] == "Energy"])
medical = (df[df['Category'] == "Medical"])
shelter = (df[df['Category'] == "Shelter"])
transportation = (df[df['Category'] == "Transportation"])
food = (df[df['Category'] == "Food"])
earthquake = (df[df['Category'] == "Earthquake"])
grounds = (df[df['Category'] == "Grounds"])
flooding = (df[df['Category'] == "Flooding"])
aftershock = (df[df['Category'] == "Aftershock"])
fire = (df[df['Category'] == "Fire"])

streamgraph = app.layout = html.Div([
    html.H4("Select time range"),
    dcc.RangeSlider(id='time-slider',
                    min=0,
                    max=len(label_bins) - 1,
                    marks=slider_marks,
                    step=1,
                    value=[0, len(label_bins) - 1]
                    ),
    dcc.Graph(
        id='graph'
    ),
])

@app.callback(
    Output("graph", "figure"),
    Input("time-slider", "value"))

def display_area(time):

    y_water_filtered = []
    y_energy_filtered = []
    y_medical_filtered = []
    y_shelter_filtered = []
    y_transportation_filtered = []
    y_food_filtered = []
    y_earthquake_filtered = []
    y_grounds_filtered = []
    y_flooding_filtered = []
    y_aftershock_filtered = []
    y_fire_filtered = []

    x_labels = label_bins[time[0] + 1: time[1] + 1]
    slider_timebins = time_bins[time[0]: time[1] + 1]
    t_prev = slider_timebins[0]
    for index in range(1, len(slider_timebins)):
        filtered_df = water[(t_prev < water['Timestamp']) & (water['Timestamp'] <= time_bins[index])]
        y_water_filtered.append(len(filtered_df))

        filtered_df = energy[(t_prev < energy['Timestamp']) & (energy['Timestamp'] <= time_bins[index])]
        y_energy_filtered.append(len(filtered_df))

        filtered_df = medical[(t_prev < medical['Timestamp']) & (medical['Timestamp'] <= time_bins[index])]
        y_medical_filtered.append(len(filtered_df))

        filtered_df = shelter[(t_prev < shelter['Timestamp']) & (shelter['Timestamp'] <= time_bins[index])]
        y_shelter_filtered.append(len(filtered_df))

        filtered_df = transportation[
            (t_prev < transportation['Timestamp']) & (transportation['Timestamp'] <= time_bins[index])]
        y_transportation_filtered.append(len(filtered_df))

        filtered_df = food[(t_prev < food['Timestamp']) & (food['Timestamp'] <= time_bins[index])]
        y_food_filtered.append(len(filtered_df))

        filtered_df = earthquake[(t_prev < earthquake['Timestamp']) & (earthquake['Timestamp'] <= time_bins[index])]
        y_earthquake_filtered.append(len(filtered_df))

        filtered_df = grounds[(t_prev < grounds['Timestamp']) & (grounds['Timestamp'] <= time_bins[index])]
        y_grounds_filtered.append(len(filtered_df))

        filtered_df = flooding[(t_prev < flooding['Timestamp']) & (flooding['Timestamp'] <= time_bins[index])]
        y_flooding_filtered.append(len(filtered_df))

        filtered_df = aftershock[(t_prev < aftershock['Timestamp']) & (aftershock['Timestamp'] <= time_bins[index])]
        y_aftershock_filtered.append(len(filtered_df))

        filtered_df = fire[(t_prev < fire['Timestamp']) & (fire['Timestamp'] <= time_bins[index])]
        y_fire_filtered.append(len(filtered_df))

        t_prev = time_bins[index]

    plt = go.Figure(
        layout=go.Layout(
            height=300,
            margin=go.layout.Margin(t=30)
        ))
    # plt.update_layout(yaxis_range=[0, 1500])
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_earthquake_filtered,
        name='Earthquake',
        mode='lines',
        line=dict(width=0.5, color='red'),
        stackgroup='one'))
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_water_filtered,
        name='Water',
        mode='lines',
        line=dict(width=0.5, color='orange'),
        stackgroup='one'))
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_energy_filtered,
        name='Energy',
        mode='lines',
        line=dict(width=0.5, color='green'),
        stackgroup='one'))
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_medical_filtered,
        name='Medical',
        mode='lines',
        line=dict(width=0.5, color='blue'),
        stackgroup='one'))
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_shelter_filtered,
        name='Shelter',
        mode='lines',
        line=dict(width=0.5, color='darkred'),
        stackgroup='one'))
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_transportation_filtered,
        name='Transportation',
        mode='lines',
        line=dict(width=0.5, color='darkblue'),
        stackgroup='one'))
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_food_filtered,
        name='Food',
        mode='lines',
        line=dict(width=0.5, color='darkgreen'),
        stackgroup='one'))
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_grounds_filtered,
        name='Grounds',
        mode='lines',
        line=dict(width=0.5, color='brown'),
        stackgroup='one'))
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_flooding_filtered,
        name='Flooding',
        mode='lines',
        line=dict(width=0.5, color='aqua'),
        stackgroup='one'))
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_aftershock_filtered,
        name='Aftershock',
        mode='lines',
        line=dict(width=0.5, color='lightgreen'),
        stackgroup='one'))
    plt.add_trace(go.Scatter(
        x=x_labels, y=y_fire_filtered,
        name='Fire',
        mode='lines',
        line=dict(width=0.5, color='purple'),
        stackgroup='one'))

    return plt


if __name__ == '__main__':
    app.run_server(debug=True)
