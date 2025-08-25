# dataframe_filter_module.py

import streamlit as st
import pandas as pd
import datetime # Required for date filtering

# --- Callback functions ---
def update_pending_multiselect(key_prefix, col_name):
    new_value = st.session_state[f"{key_prefix}_dialog_multiselect_{col_name}"]
    if new_value:
        st.session_state[f'{key_prefix}_pending_filters'][col_name] = new_value
    else:
        if col_name in st.session_state[f'{key_prefix}_pending_filters']:
            del st.session_state[f'{key_prefix}_pending_filters'][col_name]

def update_pending_slider(key_prefix, col_name, min_val, max_val):
    new_value = st.session_state[f"{key_prefix}_dialog_slider_{col_name}"]
    if new_value != (min_val, max_val):
        st.session_state[f'{key_prefix}_pending_filters'][col_name] = new_value
    else:
        if col_name in st.session_state[f'{key_prefix}_pending_filters']:
            del st.session_state[f'{key_prefix}_pending_filters'][col_name]

def update_pending_date_input(key_prefix, col_name, min_date, max_date):
    new_value = st.session_state[f"{key_prefix}_dialog_date_input_{col_name}"]
    if len(new_value) == 2 and new_value != (min_date, max_date):
        st.session_state[f'{key_prefix}_pending_filters'][col_name] = new_value
    else:
        if col_name in st.session_state[f'{key_prefix}_pending_filters']:
            del st.session_state[f'{key_prefix}_pending_filters'][col_name]

def update_text_filter_toggle(key_prefix, col_name):
    toggle_key = f"{col_name}_filter_active"
    is_filter_active = st.session_state[f"{key_prefix}_dialog_toggle_{col_name}"]

    st.session_state[f'{key_prefix}_pending_filters'][toggle_key] = is_filter_active

    if not is_filter_active and col_name in st.session_state[f'{key_prefix}_pending_filters']:
        del st.session_state[f'{key_prefix}_pending_filters'][col_name]

def _update_pending_long_text_mode(key_prefix, col_name):
    mode_key = f"{col_name}_long_text_mode"
    st.session_state[f'{key_prefix}_pending_filters'][mode_key] = st.session_state[f"{key_prefix}_radio_{col_name}"]

def _update_pending_long_text_value(key_prefix, col_name):
    value_key = f"{col_name}_long_text_value"
    new_value = st.session_state[f"{key_prefix}_text_input_{col_name}"]
    if new_value:
        st.session_state[f'{key_prefix}_pending_filters'][value_key] = new_value
    else:
        if value_key in st.session_state[f'{key_prefix}_pending_filters']:
            del st.session_state[f'{key_prefix}_pending_filters'][value_key]


# --- The Dialog Function ---
@st.dialog("⚙️ Filter Options", width="large")
def filter_dialog_content(original_df, key_prefix, long_text_cols):
    text_cols = original_df.select_dtypes(include='object').columns
    number_cols = original_df.select_dtypes(include=['int64', 'float64']).columns
    date_cols = original_df.select_dtypes(include=['datetime64[ns]']).columns

    #st.write("Adjust your filters below:")

    tab1, tab2, tab3 = st.tabs(["🔤 Text Filters", "🔢 Numeric Filters", "📅 Date Filters"])

    with tab1:
        for col in text_cols:
            with st.container(border=True):
                st.markdown(f"**{col}**")
                if col in long_text_cols:
                    long_text_mode_key = f"{col}_long_text_mode"
                    long_text_value_key = f"{col}_long_text_value"

                    current_mode = st.session_state[f'{key_prefix}_pending_filters'].get(long_text_mode_key, "Include")
                    current_text = st.session_state[f'{key_prefix}_pending_filters'].get(long_text_value_key, "")

                    col_radio, col_text_input = st.columns([0.4, 0.6])

                    with col_radio:
                        selected_mode = st.radio(
                            "Condition:",
                            ["Include", "Don't Include"],
                            index=0 if current_mode == "Include" else 1,
                            key=f"{key_prefix}_radio_{col}",
                            horizontal=True,
                            on_change=_update_pending_long_text_mode,
                            args=(key_prefix, col,)
                        )
                        st.session_state[f'{key_prefix}_pending_filters'][long_text_mode_key] = selected_mode

                    with col_text_input:
                        search_text_input = st.text_input(
                            "Search Text:",
                            value=current_text,
                            key=f"{key_prefix}_text_input_{col}",
                            on_change=_update_pending_long_text_value,
                            args=(key_prefix, col,),
                            placeholder="Enter text to search"
                        )
                        if search_text_input:
                            st.session_state[f'{key_prefix}_pending_filters'][long_text_value_key] = search_text_input
                        else:
                            if long_text_value_key in st.session_state[f'{key_prefix}_pending_filters']:
                                del st.session_state[f'{key_prefix}_pending_filters'][long_text_value_key]
                else:
                    toggle_key = f"{col}_filter_active"

                    initial_toggle_value = col in st.session_state[f'{key_prefix}_pending_filters'] and \
                                           bool(st.session_state[f'{key_prefix}_pending_filters'].get(col))

                    col_toggle, col_multiselect = st.columns([0.4, 0.6])

                    with col_toggle:
                        is_filter_active = st.toggle(
                            "Show All / Filter",
                            value=st.session_state[f'{key_prefix}_pending_filters'].get(toggle_key, initial_toggle_value),
                            key=f"{key_prefix}_dialog_toggle_{col}",
                            on_change=update_text_filter_toggle,
                            args=(key_prefix, col,)
                        )

                    with col_multiselect:
                        if is_filter_active:
                            options = original_df[col].unique().tolist()
                            st.multiselect(
                                "Select options:",
                                options,
                                default=st.session_state[f'{key_prefix}_pending_filters'].get(col, []),
                                key=f"{key_prefix}_dialog_multiselect_{col}",
                                on_change=update_pending_multiselect,
                                args=(key_prefix, col,),
                                placeholder="Select values to filter..."
                            )
                        else:
                            if col in st.session_state[f'{key_prefix}_pending_filters']:
                                del st.session_state[f'{key_prefix}_pending_filters'][col]

    with tab2:
        for col in number_cols:
            with st.container(border=True):
                st.markdown(f"**{col}**")
                min_val = original_df[col].min()
                max_val = original_df[col].max()

                if pd.isna(min_val) or pd.isna(max_val):
                    st.warning(f"Column '{col}' contains no valid data to filter.")
                else:
                    min_val = float(min_val)
                    max_val = float(max_val)
                    current_range_val = st.session_state[f'{key_prefix}_pending_filters'].get(col, (min_val, max_val))

                    if min_val == max_val:
                        st.text_input("Value:", value=min_val, disabled=True, key=f"{key_prefix}_text_{col}")
                    else:
                        st.slider(
                            "Select Range:",
                            min_val,
                            max_val,
                            value=(current_range_val[0], current_range_val[1]),
                            key=f"{key_prefix}_dialog_slider_{col}",
                            on_change=update_pending_slider,
                            args=(key_prefix, col, min_val, max_val)
                        )

    with tab3:
        for col in date_cols:
            with st.container(border=True):
                st.markdown(f"**{col}**")
                min_date = original_df[col].min().date()
                max_date = original_df[col].max().date()
                current_date_range_val = st.session_state[f'{key_prefix}_pending_filters'].get(col, (min_date, max_date))

                st.date_input(
                    "Select Date Range:",
                    value=(current_date_range_val[0], current_date_range_val[1]),
                    min_value=min_date,
                    max_value=max_date,
                    key=f"{key_prefix}_dialog_date_input_{col}",
                    on_change=update_pending_date_input,
                    args=(key_prefix, col, min_date, max_date)
                )

    st.markdown("--- ")
    col_dialog_btns_1, col_dialog_btns_2 = st.columns([1,1])

    with col_dialog_btns_1:
        if st.button("Apply Filters", key=f"{key_prefix}_apply_filters_dialog_button"):
            st.session_state[f'{key_prefix}_applied_filters'] = st.session_state[f'{key_prefix}_pending_filters'].copy()
            st.rerun()

    with col_dialog_btns_2:
        if st.button("Cancel", key=f"{key_prefix}_cancel_filters_dialog_button"):
            st.rerun()



# --- Reusable Filter Function ---
def dataframe_filter_ui(df: pd.DataFrame, key_prefix: str = "default_filter", long_text_cols: list = None) -> pd.DataFrame:
    """
    Provides a reusable Streamlit UI for filtering a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to be filtered.
        key_prefix (str): A unique prefix for session state keys and widget keys.
                          Essential for using multiple filter instances on one page.
        long_text_cols (list, optional): A list of column names (strings) that should use
                                        the "Include"/"Don't Include" text search filter
                                        instead of the multiselect. Defaults to None.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if long_text_cols is None:
        long_text_cols = []

    if f'{key_prefix}_applied_filters' not in st.session_state:
        st.session_state[f'{key_prefix}_applied_filters'] = {}
    if f'{key_prefix}_pending_filters' not in st.session_state:
        st.session_state[f'{key_prefix}_pending_filters'] = {}

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Filter Data", icon="🔍", key=f"{key_prefix}_open_filter_button"):
            st.session_state[f'{key_prefix}_pending_filters'] = st.session_state[f'{key_prefix}_applied_filters'].copy()
            filter_dialog_content(df, key_prefix, long_text_cols)

    with col2:
        if st.button("Clear All Filters", icon="🧹", key=f"{key_prefix}_clear_filters_button"):
            st.session_state[f'{key_prefix}_applied_filters'] = {}
            st.session_state[f'{key_prefix}_pending_filters'] = {}
            st.rerun()

    current_filtered_df = df.copy()

    text_cols = df.select_dtypes(include='object').columns
    number_cols = df.select_dtypes(include=['int64', 'float64']).columns
    date_cols = df.select_dtypes(include=['datetime64[ns]']).columns

    applied_filters_for_this_instance = st.session_state[f'{key_prefix}_applied_filters']

    processed_cols = set()

    for col_key, filter_value in applied_filters_for_this_instance.items():
        if col_key.endswith("_long_text_mode"):
            base_col = col_key.replace("_long_text_mode", "")
            if base_col in long_text_cols and base_col not in processed_cols:
                mode = filter_value
                text_to_search = applied_filters_for_this_instance.get(f"{base_col}_long_text_value", "")

                if text_to_search:
                    mask = current_filtered_df[base_col].astype(str).str.contains(text_to_search, case=False, na=False)
                    if mode == "Include":
                        current_filtered_df = current_filtered_df[mask]
                    else:
                        current_filtered_df = current_filtered_df[~mask]
                processed_cols.add(base_col)
            continue

        if col_key.endswith("_long_text_value"):
            continue

        if col_key.endswith("_filter_active"):
            base_col = col_key.replace("_filter_active", "")
            if base_col in text_cols and base_col not in long_text_cols and base_col not in processed_cols:
                if filter_value is False:
                    processed_cols.add(base_col)
            continue

        if col_key in processed_cols:
            continue

        if col_key in text_cols and col_key not in long_text_cols:
            toggle_key = f"{col_key}_filter_active"
            if applied_filters_for_this_instance.get(toggle_key, False) and filter_value:
                current_filtered_df = current_filtered_df[current_filtered_df[col_key].isin(filter_value)]

        elif col_key in number_cols:
            min_val, max_val = filter_value
            original_min = df[col_key].min()
            original_max = df[col_key].max()
            if (min_val > original_min or max_val < original_max):
                current_filtered_df = current_filtered_df[
                    (current_filtered_df[col_key] >= min_val) & (current_filtered_df[col_key] <= max_val)
                ]
        elif col_key in date_cols:
            start_date, end_date = filter_value
            df_column_dates = current_filtered_df[col_key].dt.date
            original_min_date = df[col_key].min().date()
            original_max_date = df[col_key].max().date()
            if (start_date > original_min_date or end_date < original_max_date):
                current_filtered_df = current_filtered_df[
                    (df_column_dates >= start_date) & (df_column_dates <= end_date)
                ]
    return current_filtered_df