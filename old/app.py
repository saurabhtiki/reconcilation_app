
import streamlit as st
import pandas as pd
import numpy as np
import os
from dataframe_filter_module import dataframe_filter_ui
# Function to load data from uploaded file
def load_data(uploaded_file, skiprows):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                return pd.read_csv(uploaded_file, skiprows=skiprows)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                return pd.read_excel(uploaded_file, skiprows=skiprows)
        except Exception as e:
            st.error(f"Error loading file: {e}")
    return None

# Function to clean key columns
    def clean_key(self, df: pd.DataFrame, columns: list, items_to_ignore: str) -> pd.DataFrame:
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
        items = [item.strip() for item in items_to_ignore.split(',')]

        for col in columns:
            df_copy[col] = df_copy[col].astype(str).str.lower()
            for item in items:
                df_copy[col] = df_copy[col].str.replace(item, '', regex=False)
        return df_copy

# Main app
def main():
    st.set_page_config(layout="wide")
    st.title("📊 Data Reconciliation App")

    # File uploaders
    col1, col2 = st.columns(2)
    with col1:
        left_file = st.file_uploader("📤 Upload Left Dataset", type=['csv', 'xlsx'], key="fu_left_file")
        left_name = "Left Dataset"
        if left_file:
            left_name = os.path.splitext(left_file.name)[0]
        left_skiprows = st.number_input(f"Skip rows at top of {left_name}", min_value=0, value=0,key="fu_left_skiprows")
    with col2:
        right_file = st.file_uploader("📥 Upload Right Dataset", type=['csv', 'xlsx'], key="fu_right_file")
        right_name = "Right Dataset"
        if right_file:
            right_name = os.path.splitext(right_file.name)[0]
        right_skiprows = st.number_input(f"Skip rows at top of {right_name}", min_value=0, value=0,key="nu_right_skiprows")
    # If both files are uploaded

    if left_file and right_file:
        left_df = load_data(left_file, left_skiprows)
        right_df = load_data(right_file, right_skiprows)

        if left_df is not None and right_df is not None:
            left_name = os.path.splitext(left_file.name)[0]
            right_name = os.path.splitext(right_file.name)[0]
            
            st.subheader("📄 Original Dataframes")
            #add expander & in that show columns
            with st.expander("View Original Dataframes"):
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(left_name)
                    st.dataframe(left_df)
                with col2:
                    st.write(right_name)
                    st.dataframe(right_df)

            # Column mapping
            st.subheader("🔗 Column Mapping")
            with st.expander("Column Mapping"):
                col1, col2 = st.columns(2)
                with col1:
                    left_cols = st.multiselect(f"Select Key Columns from {left_name}", left_df.columns,key="ms_left_cols")
                with col2:
                    right_cols = st.multiselect(f"Select Key Columns from {right_name}", right_df.columns,key="ms_right_cols")

            # Amount column selection
            st.subheader("💰 Amount Column Selection")
            with st.expander("Select Amount Columns"):
                col1, col2 = st.columns(2)
                with col1:
                    numeric_cols_left = left_df.select_dtypes(include=np.number).columns
                    left_amount_col = st.selectbox(f"Select Amount Column from {left_name}", numeric_cols_left,key="sb_left_amount_col")
                with col2:
                    numeric_cols_right = right_df.select_dtypes(include=np.number).columns
                    right_amount_col = st.selectbox(f"Select Amount Column from {right_name}", numeric_cols_right,key="sb_right_amount_col")

            # Special characters to ignore
            st.subheader("✨ Special Characters to Ignore")
            special_chars = st.text_input("Enter special characters to ignore (e.g., -,/.)", "-,/.",key="ti_special_chars")

            if left_cols and right_cols and left_amount_col and right_amount_col:
                if st.button("🚀 Reconcile", key="bt_reconcile"):
                    if len(left_cols) == len(right_cols) and len(left_cols) > 0:
                        # Clean key columns
                        left_df_cleaned = clean_key(left_df, left_cols, special_chars)
                        right_df_cleaned = clean_key(right_df, right_cols, special_chars)

                        # Create combined key columns
                        left_df_cleaned['combined_key'] = left_df_cleaned[left_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
                        right_df_cleaned['combined_key'] = right_df_cleaned[right_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

                        # Group by key columns and sum amount
                        left_sum = left_df_cleaned.groupby('combined_key')[left_amount_col].sum().reset_index()
                        right_sum = right_df_cleaned.groupby('combined_key')[right_amount_col].sum().reset_index()

                        # Rename amount columns to avoid conflicts
                        left_sum = left_sum.rename(columns={left_amount_col: 'amount_left'})
                        right_sum = right_sum.rename(columns={right_amount_col: 'amount_right'})
                        
                        # Merge aggregated data
                        merged_df = pd.merge(left_sum, right_sum, on='combined_key', how='outer')
                        merged_df = merged_df.fillna(0)

                        # Identify reconciled records
                        merged_df['reconciled'] = np.where(merged_df['amount_left'] == merged_df['amount_right'], 'yes', 'no')

                        # Separate reconciled and unreconciled
                        reconciled_keys = merged_df[merged_df['reconciled'] == 'yes']['combined_key']
                        unreconciled_keys = merged_df[merged_df['reconciled'] == 'no']['combined_key']

                        # Add reconciliation status to original dataframes
                        left_df['reconciled'] = 'no'
                        right_df['reconciled'] = 'no'
                        left_df['opposite_id'] = ''
                        right_df['opposite_id'] = ''

                        # Process reconciled records
                        for key in reconciled_keys:
                            left_indices = left_df_cleaned[left_df_cleaned['combined_key'] == key].index
                            right_indices = right_df_cleaned[right_df_cleaned['combined_key'] == key].index

                            left_df.loc[left_indices, 'reconciled'] = 'yes'
                            right_df.loc[right_indices, 'reconciled'] = 'yes'

                            left_ids = ','.join(map(str, right_df.loc[right_indices].index.tolist()))
                            right_ids = ','.join(map(str, left_df.loc[left_indices].index.tolist()))

                            left_df.loc[left_indices, 'opposite_id'] = left_ids
                            right_df.loc[right_indices, 'opposite_id'] = right_ids
                        
                        # Store dataframes in session state for manual reconciliation
                        st.session_state.left_df = left_df
                        st.session_state.right_df = right_df
                        st.session_state.left_name = left_name
                        st.session_state.right_name = right_name
                        st.session_state.left_amount_col = left_amount_col
                        st.session_state.right_amount_col = right_amount_col

                if 'left_df' in st.session_state:
                    st.subheader("🎉 Reconciliation Results")

                    tab1, tab2, tab3 = st.tabs(["✅ Reconciled Records", "❌ Unreconciled Records", "🛠️ Manual Reconciliation"])

                    with tab1:
                        st.write(f"Reconciled records from {st.session_state.left_name}")
                        st.dataframe(st.session_state.left_df[st.session_state.left_df['reconciled'] != 'no'])
                        st.write(f"Reconciled records from {st.session_state.right_name}")
                        st.dataframe(st.session_state.right_df[st.session_state.right_df['reconciled'] != 'no'])

                    with tab2:
                        st.write(f"Unreconciled records from {st.session_state.left_name}")
                        st.dataframe(st.session_state.left_df[st.session_state.left_df['reconciled'] == 'no'])
                        st.write(f"Unreconciled records from {st.session_state.right_name}")
                        st.dataframe(st.session_state.right_df[st.session_state.right_df['reconciled'] == 'no'])
                    
                    with tab3:
                        st.subheader("Manual Reconciliation")
                        st.write(dataframe_filter_ui(st.session_state.left_df[st.session_state.left_df['reconciled'] == 'no'], "left", colums=st.session_state.ms_left_cols))

if __name__ == '__main__':
    main()
