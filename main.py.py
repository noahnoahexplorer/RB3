import streamlit as st


st.set_page_config(
    page_title='Game Analysis App - Home',
    page_icon=":house:",
)

st.title("Main Page")
# st.sidebar.success("Select a page above")

import streamlit as st
import pandas as pd
import ast  # Import ast for literal_eval
from itertools import combinations  # Import combinations from itertools

# Title of the app
st.title("Game Data Analysis Tool :game_die:")

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Sidebar page selection
page = st.sidebar.selectbox("Select Filter Page", ["Individual Filter", "Related Group Filter", "Distant Group Filter"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the uploaded CSV file and ensure 'username' column is read as a string
    df = pd.read_csv(uploaded_file, dtype={'username': str})
    
    # Store the DataFrame in session state
    st.session_state['raw_df'] = df

    # Remove leading zeros in 'username' column
    df['username'] = df['username'].str.lstrip('0')

    # Clean column names by stripping any leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Display the raw data
    st.write(f"Raw Data (Total number of rows: {df.shape[0]}):")
    st.dataframe(df)

    # --- Individual Filter Page ---
    if page == "Individual Filter":
        st.subheader("Individual Filter")

        # Sidebar filters for Individual Filter
        unique_number_count = st.sidebar.slider("Unique Number Count", min_value=0, max_value=100, value=(70, 100))
        min_average_cost = st.sidebar.slider("Minimum Average Cost", min_value=0, max_value=100, value=0)
        user_profit_rate = st.sidebar.slider("User Profit Rate (%)", min_value=0, max_value=100, value=(0, 10))
        min_user_win_lose = st.sidebar.number_input("Minimum User Win/Lose", min_value=-1000, value=0)  # Allow negative values for user_win_lose

        # Apply filter button
        apply_filter = st.sidebar.button("Apply Filter")

        # Check if required columns exist
        if all(col in df.columns for col in ['average_cost', 'unique_number_count', 'user_profit_rate', 'user_win_lose']):
            if apply_filter:
                # Apply the filters based on unique number count, minimum average cost, minimum user win/lose, and profit rate
                filtered_df = df[(df['unique_number_count'] >= unique_number_count[0]) &
                                 (df['unique_number_count'] <= unique_number_count[1]) &
                                 (df['average_cost'] >= min_average_cost) &
                                 (df['user_profit_rate'] >= user_profit_rate[0]) &
                                 (df['user_profit_rate'] <= user_profit_rate[1]) &
                                 (df['user_win_lose'] >= min_user_win_lose)]  # Filtering based on user_win_lose

                # Display filtered dataframe
                st.write(f"Filtered Data (Total number of rows: {filtered_df.shape[0]}):")
                st.dataframe(filtered_df)

                # Button to download the filtered data as CSV
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_individual.csv', mime='text/csv')

        else:
            st.write("The required columns ('average_cost', 'unique_number_count', 'user_profit_rate', 'user_win_lose') are not found in the dataset. Please check your data.")

    # --- Related Group Filter Page ---
    elif page == "Related Group Filter":
        st.subheader("Related Group Filter")

        # Sidebar checkboxes for filtering by related entries
        filter_ip = st.sidebar.checkbox("Filter by IP", value=True)
        filter_registered_ip = st.sidebar.checkbox("Filter by Registered IP", value=True)
        filter_hash_password = st.sidebar.checkbox("Filter by Hash Password", value=True)
        filter_device_id = st.sidebar.checkbox("Filter by Device ID", value=True)
        filter_rng = st.sidebar.checkbox("Filter by RNG", value=True)

        # Apply filter button
        apply_filter = st.sidebar.button("Apply Filter")

        if apply_filter:
            # Only keep the columns that are selected by the user
            selected_columns = []
            if filter_ip:
                selected_columns.append('ip')
            if filter_registered_ip:
                selected_columns.append('registered_ip')
            if filter_hash_password:
                selected_columns.append('hash_password')
            if filter_device_id:
                selected_columns.append('device_id')
            if filter_rng:
                selected_columns.append('rng')

            if selected_columns:
                # Group by the selected columns and filter groups where all selected column values are exactly the same but username is different
                grouped_df = df.groupby(selected_columns).filter(lambda x: len(x['username'].unique()) > 1)

                if not grouped_df.empty:
                    # Display filtered dataframe
                    st.write(f"Filtered Data (Total number of rows: {grouped_df.shape[0]}):")
                    st.dataframe(grouped_df)

                    # Button to download the filtered data as CSV
                    csv = grouped_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_related.csv', mime='text/csv')
                else:
                    st.write("No related records found for the selected criteria.")
            else:
                st.write("Please select at least one column to filter by.")

    # --- Distant Group Filter Page ---
    elif page == "Distant Group Filter":
        st.subheader("Distant Group Filter")

        # Sidebar filters for Distant Group Filter
        avg_cost_threshold = st.sidebar.slider("Minimum Average Cost Threshold", min_value=0, max_value=200, value=50)
        unique_count_threshold = st.sidebar.slider("Minimum Unique Number Count Threshold", min_value=0, max_value=100, value=10)
        group_unique_number_threshold = st.sidebar.slider("Combinations of rows with more than how many unique numbers", min_value=0, max_value=100, value=70)
        group_profit_rate_range = st.sidebar.slider("Group Profit Rate (%)", min_value=0, max_value=100, value=(0, 10))
        group_max_overlapping_number_threshold = st.sidebar.slider("Group Maximum Overlapping Number Count", min_value=0, max_value=100, value=5)
        avg_cost_diff_threshold = st.sidebar.slider("Max Average Cost Difference", min_value=0, max_value=100, value=50)

        # Apply filter button
        apply_filter = st.sidebar.button("Apply Filter")

        # Check if 'number_cost' and 'rewards' columns exist
        if 'number_cost' in df.columns and 'rewards' in df.columns:
            # Convert the 'number_cost' column from string to dictionary
            df['number_cost'] = df['number_cost'].apply(ast.literal_eval)

            if apply_filter:
                # Filter rows based on avg_cost and unique_count
                filtered_df = df[(df['average_cost'] > avg_cost_threshold) & 
                                 (df['unique_number_count'] > unique_count_threshold)]

                # Additional logic for distant group combinations
                filtered_number_dicts = list(filtered_df['number_cost'])
                filtered_indices = list(filtered_df.index)

                valid_combinations = []

                for r in range(2, len(filtered_number_dicts) + 1):
                    for comb_indices in combinations(range(len(filtered_number_dicts)), r):
                        comb_dicts = [filtered_number_dicts[i] for i in comb_indices]
                        unique_numbers = set()
                        overlapping_numbers = set()

                        for i, d in enumerate(comb_dicts):
                            current_numbers = set(d.keys())
                            if i == 0:
                                unique_numbers = current_numbers
                            else:
                                overlapping_numbers.update(unique_numbers.intersection(current_numbers))
                            unique_numbers.update(current_numbers)

                        if len(overlapping_numbers) > group_max_overlapping_number_threshold:
                            continue

                        if len(unique_numbers) > group_unique_number_threshold:
                            total_cost = sum([filtered_df.iloc[i]['total_cost'] for i in comb_indices])
                            total_reward = sum([filtered_df.iloc[i]['rewards'] for i in comb_indices])
                            user_win_lose = total_reward - total_cost
                            profit_rate = (user_win_lose / total_reward * 100) if total_reward > 0 else 0
                            avg_costs = [filtered_df.iloc[i]['average_cost'] for i in comb_indices]
                            avg_cost_diff = max(avg_costs) - min(avg_costs)

                            # Create the list of [ref_provider, username] for each entry in the combination
                            combination_info = [[filtered_df.iloc[i]['ref_provider'], filtered_df.iloc[i]['username']] for i in comb_indices]

                            if group_profit_rate_range[0] < profit_rate < group_profit_rate_range[1] and avg_cost_diff < avg_cost_diff_threshold:
                                valid_combinations.append({
                                    'Combination': combination_info,  # Display [ref_provider, username] instead of index
                                    'Unique Numbers': len(unique_numbers),
                                    'Overlapping Numbers': len(overlapping_numbers),
                                    'Total Cost': total_cost,
                                    'Total Reward': total_reward,
                                    'User Win/Lose': user_win_lose,
                                    'Profit Rate (%)': f"{profit_rate:.2f}%",
                                    'Average Cost Difference': avg_cost_diff
                                })

                if valid_combinations:
                    valid_combinations_df = pd.DataFrame(valid_combinations)
                    st.write(f"Combinations of rows with more than {group_unique_number_threshold} unique numbers, fewer than {group_max_overlapping_number_threshold} overlapping numbers, and profit rate between {group_profit_rate_range[0]}% and {group_profit_rate_range[1]}%:")
                    st.dataframe(valid_combinations_df)

                    # Button to download the filtered data as CSV
                    csv = valid_combinations_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_distant.csv', mime='text/csv')
                else:
                    st.write("No valid combinations found with the current filters.")

else:
    st.write("Please upload a CSV file.")

