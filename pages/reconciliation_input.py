import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from dataframe_filter_module import dataframe_filter_ui

# Removed display_results from here as it's now in reconciliation_results.py

@st.cache_data
def load_data(uploaded_file: 'st.uploaded_file', skiprows: int) -> pd.DataFrame:
    """
    Loads data from an uploaded file (CSV or Excel).

    Args:
        uploaded_file (st.uploaded_file): The file uploaded by the user.
        skiprows (int): The number of rows to skip at the beginning of the file.

    Returns:
        pd.DataFrame: The loaded data as a Pandas DataFrame.
    """
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file, skiprows=skiprows)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                return pd.read_excel(uploaded_file, skiprows=skiprows)
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None

class DataReconciliationApp:

    """
    A Streamlit application for reconciling two datasets.
    """

    def __init__(self):
        """
        Initializes the DataReconciliationApp.
        """
        st.set_page_config(layout="wide")
        self.left_df = None
        self.right_df = None
        self.left_name = "Left Dataset"
        self.right_name = "Right Dataset"

    @staticmethod
    @st.cache_data
    def clean_key(df: pd.DataFrame, columns: list, items_to_ignore: str) -> pd.DataFrame:
        """
        Cleans the key columns of a DataFrame by converting them to lowercase and removing specified items.

        Args:
            df (pd.DataFrame): The DataFrame to clean.
            columns (list): The list of key columns to clean.
            items_to_ignore (str): A comma-separated string of characters or phrases to remove.

        Returns:
            pd.DataFrame: The DataFrame with cleaned key columns.
        """
        df_copy = df.copy()
        # Split the input string by comma and strip whitespace from each item
        items = [item.strip() for item in items_to_ignore.split(',') if item.strip()]
        # Always include comma in the items to ignore
        if ',' not in items:
            items.append(',')
        # Sort items by length in descending order to ensure longer strings are replaced first
        items.sort(key=len, reverse=True)
        #print(items)
        for col in columns:
            df_copy[col] = df_copy[col].astype(str).str.lower()
            # Remove all spaces from the column
            df_copy[col] = df_copy[col].str.replace(' ', '', regex=False)
            for item in items:
                df_copy[col] = df_copy[col].str.replace(item.lower(), '', regex=False)
                #print(df_copy[col])
                #print(f"Removed '{item}' from column '{col}'")
        #st.dataframe(df_copy)
        return df_copy


    def run(self):
        """
        Runs the main Streamlit application.
        """
        st.title("📊 Data Reconciliation App")

        if st.button("Clear Cache and Rerun"):
            st.cache_data.clear()
            st.rerun()
        col1, col2 = st.columns([3,2],border=True)
        with col1:
            # Reconciliation Type Selection
            reconciliation_type = st.radio(
                "Select Reconciliation Type:",
                ("Reconcile by Matching Sum", "Reconcile Line by Line"),
                index=0, # Default to "Reconcile by Matching Sum"
                key="reconciliation_type_radio",horizontal=True,
                help="Choose how you want to reconcile your datasets. 'Reconcile by Matching Sum' aggregates amounts by key columns. 'Reconcile Line by Line' compares individual transaction amounts."
            )
            st.session_state.reconciliation_type = reconciliation_type
        with col2:
            #Roundoff Yes/No
            roundoff_option = st.radio(
                "Match Round-off Value?",
                ("Yes", "No"),
                index=1, # Default to "No"
                key="roundoff_option_radio",horizontal=True,
                help="Choose 'Yes' to Match Value after round-off differences when comparing amounts. Choose 'No' for exact matches only."
            )
            st.session_state.roundoff_option = roundoff_option
        # File uploaders
        col1, col2 = st.columns(2)
        with col1:
            left_file = st.file_uploader("📤 Upload Left Dataset", type=['csv', 'xlsx'], key="fu_left_file", help="Upload the first dataset (CSV or Excel).")
            if left_file:
                self.left_name = os.path.splitext(left_file.name)[0]
            left_skiprows = st.number_input(f"Skip rows at top of {self.left_name}", min_value=0, value=0, key="nu_left_skiprows", help="Number of rows to skip at the top of the left dataset.")
        with col2:
            right_file = st.file_uploader("📥 Upload Right Dataset", type=['csv', 'xlsx'], key="fu_right_file", help="Upload the second dataset (CSV or Excel).")
            if right_file:
                self.right_name = os.path.splitext(right_file.name)[0]
            right_skiprows = st.number_input(f"Skip rows at top of {self.right_name}", min_value=0, value=0, key="nu_right_skiprows", help="Number of rows to skip at the top of the right dataset.")

        if left_file and right_file:
            self.left_df = load_data(left_file, left_skiprows)
            self.right_df = load_data(right_file, right_skiprows)

            if self.left_df is not None and self.right_df is not None:
                st.subheader("📄 Original Dataframes")
                with st.expander("View Original Dataframes"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(self.left_name)
                        st.dataframe(self.left_df)
                    with col2:
                        st.write(self.right_name)
                        st.dataframe(self.right_df)

                with st.form("reconciliation_form"):
                    tab1, tab2, tab3 = st.tabs(["Column Mapping", "Amount Column Selection", "Characters/Phrases to Ignore"])

                    with tab1:
                        st.subheader("🔗 Column Mapping")
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'left_cols' not in st.session_state:
                                st.session_state.left_cols = []
                            st.session_state.left_cols = st.multiselect(f"Select Key Columns from {self.left_name}", self.left_df.columns, key="ms_left_cols", help="Select the columns to use as keys for reconciliation from the left dataset.")
                        with col2:
                            if 'right_cols' not in st.session_state:
                                st.session_state.right_cols = []
                            st.session_state.right_cols = st.multiselect(f"Select Key Columns from {self.right_name}", self.right_df.columns, key="ms_right_cols", help="Select the columns to use as keys for reconciliation from the right dataset.")

                    with tab2:
                        st.subheader("💰 Amount Column Selection")
                        col1, col2 = st.columns(2)
                        with col1:
                            numeric_cols_left = self.left_df.select_dtypes(include=np.number).columns
                            left_amount_col = st.selectbox(f"Select Amount Column from {self.left_name}", numeric_cols_left, key="sb_left_amount_col", help="Select the column containing the amount to be reconciled from the left dataset.")
                        with col2:
                            numeric_cols_right = self.right_df.select_dtypes(include=np.number).columns
                            right_amount_col = st.selectbox(f"Select Amount Column from {self.right_name}", numeric_cols_right, key="sb_right_amount_col", help="Select the column containing the amount to be reconciled from the right dataset.")

                    with tab3:
                        st.subheader("✨ Characters/Phrases to Ignore")
                        items_to_ignore = st.text_input("Enter additional characters or phrases to ignore (comma-separated)", ".,/,*,-,&", key="ti_special_chars", help="Comma (,) is always ignored. Enter additional characters or phrases that should be ignored when comparing key columns. Separate multiple items with commas (e.g., './,*,ABC').")

                    submitted = st.form_submit_button("🚀 Reconcile")

                if submitted:
                    if st.session_state.reconciliation_type == "Reconcile by Matching Sum" and (len(st.session_state.left_cols) != len(st.session_state.right_cols) or len(st.session_state.left_cols) == 0):
                        st.error("For 'Reconcile by Matching Sum', number of key columns must be the same for both datasets and greater than zero.")
                    elif st.session_state.reconciliation_type == "Reconcile Line by Line" and len(st.session_state.left_cols) != len(st.session_state.right_cols):
                        st.error("For 'Reconcile Line by Line' with key columns, number of key columns must be the same for both datasets.")
                    else:
                        st.session_state.left_amount_col = left_amount_col
                        st.session_state.right_amount_col = right_amount_col
                        with st.spinner("Reconciling..."):
                            self.reconcile(st.session_state.left_cols, st.session_state.right_cols, left_amount_col, right_amount_col, items_to_ignore, reconciliation_type)
                            st.toast("Reconciliation complete! 🎉")
                        
                        st.switch_page("pages/reconciliation_results.py") # Navigate to the results page
                        #after this the page is reruned, i want it shall remain as it is without rerun
                       
                            

    def reconcile(self, left_cols: list, right_cols: list, left_amount_col: str, right_amount_col: str, special_chars: str, reconciliation_type: str):
        """
        Performs the reconciliation process.

        Args:
            left_cols (list): Key columns from the left dataset.
            right_cols (list): Key columns from the right dataset.
            left_amount_col (str): Amount column from the left dataset.
            right_amount_col (str): Amount column from the right dataset.
            special_chars (str): Special characters to ignore.
            reconciliation_type (str): Type of reconciliation (e.g., "Reconcile by Matching Sum", "Reconcile Line by Line").
        """
        reconciled_left_df = self.left_df.copy()
        reconciled_right_df = self.right_df.copy()

        reconciled_left_df['reconciliation_status'] = 'unreconciled'
        reconciled_right_df['reconciliation_status'] = 'unreconciled'
        reconciled_left_df['reconciliation_id'] = ''
        reconciled_right_df['reconciliation_id'] = ''

        if reconciliation_type == "Reconcile by Matching Sum":
            left_df_cleaned = self.clean_key(self.left_df, st.session_state.left_cols, special_chars)
            right_df_cleaned = self.clean_key(self.right_df, st.session_state.right_cols, special_chars)

            left_df_cleaned['combined_key'] = left_df_cleaned[st.session_state.left_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            right_df_cleaned['combined_key'] = right_df_cleaned[st.session_state.right_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            #show results for testing
            #st.dataframe(left_df_cleaned)
            #st.dataframe(right_df_cleaned)
            
            left_sum = left_df_cleaned.groupby('combined_key')[left_amount_col].sum().reset_index()
            right_sum = right_df_cleaned.groupby('combined_key')[right_amount_col].sum().reset_index()
            #based on round off match value
            if st.session_state.roundoff_option == "No":
                left_sum = left_sum.rename(columns={left_amount_col: 'amount_left'})
                right_sum = right_sum.rename(columns={right_amount_col: 'amount_right'})
            else:
                left_sum['amount_left'] = left_sum[left_amount_col].apply(lambda x: round(x, 0))
                right_sum['amount_right'] = right_sum[right_amount_col].apply(lambda x: round(x, 0))

            merged_df = pd.merge(left_sum, right_sum, on='combined_key', how='outer')
            merged_df = merged_df.fillna(0)

            merged_df['reconciled'] = np.where(np.isclose(merged_df['amount_left'], merged_df['amount_right']), 'yes', 'no')

            reconciled_keys = merged_df[merged_df['reconciled'] == 'yes']['combined_key']

            for key in reconciled_keys:
                left_indices = left_df_cleaned[left_df_cleaned['combined_key'] == key].index
                right_indices = right_df_cleaned[right_df_cleaned['combined_key'] == key].index

                reconciled_left_df.loc[left_indices, 'reconciliation_status'] = 'reconciled'
                reconciled_right_df.loc[right_indices, 'reconciliation_status'] = 'reconciled'

                left_ids_str = ','.join(map(str, right_indices.tolist()))
                right_ids_str = ','.join(map(str, left_indices.tolist()))

                reconciled_left_df.loc[left_indices, 'reconciliation_id'] = left_ids_str
                reconciled_right_df.loc[right_indices, 'reconciliation_id'] = right_ids_str

        elif reconciliation_type == "Reconcile Line by Line":
            if len(st.session_state.left_cols) > 0 and len(st.session_state.right_cols) > 0:
                # Line by line with key columns
                left_df_cleaned = self.clean_key(self.left_df, st.session_state.left_cols, special_chars)
                right_df_cleaned = self.clean_key(self.right_df, st.session_state.right_cols, special_chars)

                left_df_cleaned['combined_key'] = left_df_cleaned[st.session_state.left_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
                right_df_cleaned['combined_key'] = right_df_cleaned[st.session_state.right_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

                
                # Create a temporary unique identifier for each row in the original dataframes
                self.left_df['_temp_id'] = range(len(self.left_df))
                self.right_df['_temp_id'] = range(len(self.right_df))

                # Merge based on cleaned keys and amount columns
                temp_merged = pd.merge(
                    left_df_cleaned.reset_index().rename(columns={'index':'original_left_index'}),
                    right_df_cleaned.reset_index().rename(columns={'index':'original_right_index'}),
                    left_on=['combined_key', left_amount_col],
                    right_on=['combined_key', right_amount_col],
                    how='inner',
                    suffixes=('_left', '_right')
                )

                # Mark reconciled rows in original dataframes
                for idx_left, idx_right in zip(temp_merged['original_left_index'], temp_merged['original_right_index']):
                    reconciled_left_df.loc[idx_left, 'reconciliation_status'] = 'reconciled'
                    reconciled_right_df.loc[idx_right, 'reconciliation_status'] = 'reconciled'
                    reconciled_left_df.loc[idx_left, 'reconciliation_id'] = str(idx_right)
                    reconciled_right_df.loc[idx_right, 'reconciliation_id'] = str(idx_left)

            else:
                # Line by line without key columns (match only on amount)
                # Sort both dataframes by the amount column
                left_sorted = self.left_df.sort_values(by=left_amount_col).reset_index(drop=False) # Keep original index
                right_sorted = self.right_df.sort_values(by=right_amount_col).reset_index(drop=False) # Keep original index

                # Create temporary reconciliation status columns
                left_sorted['reconciliation_status'] = 'unreconciled'
                right_sorted['reconciliation_status'] = 'unreconciled'
                left_sorted['reconciliation_id'] = ''
                right_sorted['reconciliation_id'] = ''

                # Iterate and compare amounts
                i, j = 0, 0
                while i < len(left_sorted) and j < len(right_sorted):
                    if np.isclose(left_sorted.loc[i, left_amount_col], right_sorted.loc[j, right_amount_col]):
                        left_sorted.loc[i, 'reconciliation_status'] = 'reconciled'
                        right_sorted.loc[j, 'reconciliation_status'] = 'reconciled'
                        left_sorted.loc[i, 'reconciliation_id'] = str(right_sorted.loc[j, 'index']) # Store original index
                        right_sorted.loc[j, 'reconciliation_id'] = str(left_sorted.loc[i, 'index']) # Store original index
                        i += 1
                        j += 1
                    elif left_sorted.loc[i, left_amount_col] < right_sorted.loc[j, right_amount_col]:
                        i += 1
                    else:
                        j += 1
                
                # Update original dataframes based on sorted dataframes
                for original_idx, row in left_sorted.iterrows():
                    reconciled_left_df.loc[row['index'], 'reconciliation_status'] = row['reconciliation_status']
                    reconciled_left_df.loc[row['index'], 'reconciliation_id'] = row['reconciliation_id']
                
                for original_idx, row in right_sorted.iterrows():
                    reconciled_right_df.loc[row['index'], 'reconciliation_status'] = row['reconciliation_status']
                    reconciled_right_df.loc[row['index'], 'reconciliation_id'] = row['reconciliation_id']

        st.session_state.reconciled_left_df = reconciled_left_df
        st.session_state.reconciled_right_df = reconciled_right_df
        st.session_state.left_name = self.left_name
        st.session_state.right_name = self.right_name

app = DataReconciliationApp()
app.run()