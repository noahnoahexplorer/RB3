import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.express as px
from wordcloud import WordCloud
import ast
import seaborn as sns

# Streamlit app layout
st.set_page_config(
    page_title="Game Analysis App - Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title("Game Analysis Dashboard :bar_chart:")


############################################Visualization Functions################################################
# Function to display KPI cards
def display_kpi_card(title, value, delta=None, delta_color="normal", icon=None, color="lightblue"):
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 10px; border-radius: 5px;">
            <div style="font-size: 20px; color: black;">{title}</div>
            <div style="font-size: 28px; font-weight: bold; color: black;">{value}</div>
            {'<div style="font-size: 16px; color: green;">' + icon + '</div>' if icon else ''}
            {'<div style="font-size: 16px; color:' + delta_color + ';">' + delta + '</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True
    )


# Function to generate scorecard for unique usernames
def generate_unique_username_scorecard(df):
    unique_usernames = df['username'].nunique()
    display_kpi_card("Active Players", unique_usernames, icon="👤", color="#6fa8dc")


# Function to generate scorecard for total cost
def generate_total_cost_scorecard(df):
    total_cost = df['total_cost'].sum()
    display_kpi_card("Total Cost", f"${total_cost:,.2f}", icon="💰", color="#ff9999")
    return total_cost


# Function to generate scorecard for total reward
def generate_total_reward_scorecard(df):
    total_reward = df['rewards'].sum()
    display_kpi_card("Total Reward", f"${total_reward:,.2f}", icon="🏆", color="#b6d7a8")
    return total_reward


# Function to generate scorecard for profit margin
def generate_profit_margin_scorecard(total_cost, total_reward):
    if total_cost != 0:
        profit_margin = ((total_reward - total_cost) / total_cost) * 100
        # Set color based on profit margin
        color = "#b6d7a8" if profit_margin >= 0 else "#ff9999"
        display_kpi_card("Profit Margin", f"{profit_margin:.2f}%", icon="📈", color=color)
    else:
        display_kpi_card("Profit Margin", "N/A", icon="📈", color="#c9daf8")


# Function to generate top 10 winners by net gain/loss
def generate_top_winners_bar_chart(df):
    df["username"] = df["username"].astype(str)
    user_summary = df.groupby("username").agg(
        total_rewards=pd.NamedAgg(column="rewards", aggfunc="sum"),
        total_cost=pd.NamedAgg(column="total_cost", aggfunc="sum"),
    )
    user_summary["net_gain_loss"] = user_summary["total_rewards"] - user_summary["total_cost"]
    top_winners = user_summary.sort_values(by="net_gain_loss", ascending=False).head(10).reset_index()
    top_winners["username"] = top_winners["username"].astype(str)
    fig = px.bar(
        top_winners,
        x="net_gain_loss",
        y="username",
        orientation="h",
        title="Top 10 Winners by Net Gain/Loss",
        labels={"net_gain_loss": "Net Gain/Loss", "username": "Username"},
        color="net_gain_loss",
        color_continuous_scale="Viridis",
        category_orders={"username": top_winners["username"].tolist()}
    )
    fig.update_yaxes(type="category")
    st.plotly_chart(fig)


# Function to generate profit margin by provider & product
def generate_profit_margin_bar_chart(df):
    df["Provider & Product"] = df["ref_provider"] + " - " + df["product_name_en"]
    provider_product_summary = (
        df.groupby(["ref_provider", "product_name_en"])
        .agg(
            total_reward=pd.NamedAgg(column="rewards", aggfunc="sum"),
            total_cost=pd.NamedAgg(column="total_cost", aggfunc="sum"),
        )
        .head(15)
        .reset_index()
    )
    provider_product_summary["profit_margin"] = (
            -(provider_product_summary["total_reward"] - provider_product_summary["total_cost"])
            / provider_product_summary["total_cost"]
    )
    provider_product_summary["Provider & Product"] = (
            provider_product_summary["ref_provider"] + " - " + provider_product_summary["product_name_en"]
    )
    provider_product_summary = provider_product_summary.sort_values(by="profit_margin", ascending=True)
    custom_color_scale = [[0, "red"], [0.5, "yellow"], [1, "green"]]
    fig = px.bar(
        provider_product_summary,
        x="profit_margin",
        y="Provider & Product",
        orientation="h",
        title="Profit Margin by Provider & Product",
        labels={"profit_margin": "Profit Margin", "Provider & Product": "Provider & Product"},
        color="profit_margin",
        color_continuous_scale=custom_color_scale,
        category_orders={"Provider & Product": provider_product_summary["Provider & Product"].tolist()}
    )
    fig.update_yaxes(type="category")
    st.plotly_chart(fig)


# Function to extract numbers from dictionary and create exploded dataframe
def extract_numbers_from_dict(df):
    df['number'] = df['number_cost'].apply(lambda x: list(eval(x).keys()))
    df_exploded = df.explode('number').reset_index(drop=True)
    df_exploded['number'] = df_exploded['number'].astype(str)
    return df_exploded


# Function to generate word cloud from unique numbers and their frequencies
def generate_unique_number_wordcloud(df_exploded):
    number_counts = df_exploded['number'].value_counts().to_dict()
    wordcloud = WordCloud(
        width=400,
        height=200,
        background_color='white',
        colormap='viridis',
        max_words=100,
        prefer_horizontal=1.0
    ).generate_from_frequencies(number_counts)
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)


# Function to display heatmaps based on the betting data
def display_heatmaps(df):
    try:
        number_covered = list(range(1, 101))
        heatmap_matrix = pd.DataFrame(0, index=number_covered, columns=["Betting Coverage"])
        for betting_dict in df['number_cost']:
            for number, amount in betting_dict.items():
                heatmap_matrix.loc[int(number)] += amount
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(heatmap_matrix, cmap='YlGnBu', annot=False, cbar=True, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error in number betting heatmap: {e}")


############################################Data Handling & Visualization################################################
if 'raw_df' in st.session_state:
    df = st.session_state['raw_df']

    # Sidebar filters
    st.sidebar.title("Filters")
    provider_filter = st.sidebar.multiselect('Select Provider(s):', df['ref_provider'].unique())
    product_filter = st.sidebar.multiselect('Select Product Name(s):', df['product_name_en'].unique())

    filtered_df = df.copy()
    if provider_filter:
        filtered_df = filtered_df[filtered_df['ref_provider'].isin(provider_filter)]
    if product_filter:
        filtered_df = filtered_df[filtered_df['product_name_en'].isin(product_filter)]
    filtered_df = filtered_df.reset_index(drop=True)
    st.write("Filtered Data:")
    st.write(filtered_df)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_reward = generate_total_reward_scorecard(filtered_df)
    with col2:
        total_cost = generate_total_cost_scorecard(filtered_df)
    with col3:
        generate_unique_username_scorecard(filtered_df)
    with col4:
        generate_profit_margin_scorecard(total_cost, total_reward)

    # Charts for profit margin and top winners
    col5, col6 = st.columns(2)
    with col5:
        generate_profit_margin_bar_chart(filtered_df)
    with col6:
        generate_top_winners_bar_chart(filtered_df)

    # Word cloud and heatmap
    df_exploded = extract_numbers_from_dict(filtered_df)
    col7, col8 = st.columns(2)
    with col7:
        st.subheader("Hot Numbers")
        generate_unique_number_wordcloud(df_exploded)
    with col8:
        st.subheader("Number Betting Heatmap")
        if 'number_cost' in filtered_df.columns:
            filtered_df['number_cost'] = filtered_df['number_cost'].apply(ast.literal_eval)
            display_heatmaps(filtered_df)
        else:
            st.warning("The column 'number_cost' is missing in the uploaded CSV.")

    ############################################Geo Map##############################################################
    if "heatmap_generated" not in st.session_state:
        st.session_state.heatmap_generated = False
    if st.button("Generate Heatmap"):
        st.session_state.heatmap_generated = True
        ip_list = [{"query": ip} for ip in filtered_df['ip'].tolist()]
        response = requests.post("http://ip-api.com/batch", json=ip_list).json()
        filtered_df['Latitude'] = [ip_info['lat'] for ip_info in response]
        filtered_df['Longitude'] = [ip_info['lon'] for ip_info in response]
        filtered_locations = filtered_df.dropna(subset=['Latitude', 'Longitude'])
        st.session_state.heatmap_data = [[row['Latitude'], row['Longitude']] for index, row in
                                         filtered_locations.iterrows()]
        st.session_state.map_state = filtered_locations
    if st.session_state.heatmap_generated and st.session_state.heatmap_data:
        avg_lat = st.session_state.map_state['Latitude'].mean()
        avg_lon = st.session_state.map_state['Longitude'].mean()
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=2)
        HeatMap(st.session_state.heatmap_data).add_to(m)
        st_data = st_folium(m, width=700, height=500)
else:
    st.write("No data loaded. Please load data on the HomePage first.")
