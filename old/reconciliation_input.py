import streamlit as st
import pandas as pd
import numpy as np
import os
from dataframe_filter_module import dataframe_filter_ui

@st.cache_data
def load_data(uploaded_file: 'st.uploaded_file', skiprows: int) -> pd.DataFrame:
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
    def __init__(self):
        st.set_page_config(layout="wide")
        self.left_df = None
        self.right_df = None
        self.left_name = "Left Dataset"
        self.right_name = "Right Dataset"

    @staticmethod
    @st.cache_data
    def clean_key(df: pd.DataFrame, columns: list, items_to_ignore: str) -> pd.DataFrame:
        df_copy = df.copy()
        items = [item.strip() for item in items_to_ignore.split(',') if item.strip()]
        if ',' not in items:
            items.append(',')
        items.sort(key=len, reverse=True)

        for col in columns:
            df_copy[col] = df_copy[col].astype(str).str.lower()
            df_copy[col] = df_copy[col].str.replace(' ', '', regex=False)
            for item in items:
                df_copy[col] = df_copy[col].str.replace(item, '', regex=False)
        return df_copy

    def run(self):
        st.title("📊 Data Reconciliation App")

        if st.button("Clear Cache and Rerun"):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()

        reconciliation_type = st.radio(
            "Select Reconciliation Type:",
            ("Reconcile by Matching Sum", "Reconcile Line by Line"),
            index=0,
            key="reconciliation_type_radio", horizontal=True
        )
        st.session_state.reconciliation_type = reconciliation_type

        # --- File uploaders ---
        col1, col2 = st.columns(2)
        with col1:
            left_file = st.file_uploader("📤 Upload Left Dataset", type=['csv', 'xlsx'], key="fu_left_file")
            if left_file:
                self.left_name = os.path.splitext(left_file.name)[0]
                st.session_state.left_file = left_file
            left_skiprows = st.number_input(f"Skip rows at top of {self.left_name}", min_value=0, value=0, key="nu_left_skiprows")
        with col2:
            right_file = st.file_uploader("📥 Upload Right Dataset", type=['csv', 'xlsx'], key="fu_right_file")
            if right_file:
                self.right_name = os.path.splitext(right_file.name)[0]
                st.session_state.right_file = right_file
            right_skiprows = st.number_input(f"Skip rows at top of {self.right_name}", min_value=0, value=0, key="nu_right_skiprows")

        # --- If both files uploaded ---
        if "left_file" in st.session_state and "right_file" in st.session_state:
            self.left_df = load_data(st.session_state.left_file, left_skiprows)
            self.right_df = load_data(st.session_state.right_file, right_skiprows)

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

                # --- Form for reconciliation setup ---
                with st.form("reconciliation_form"):
                    tab1, tab2, tab3 = st.tabs(["Column Mapping", "Amount Column Selection", "Characters/Phrases to Ignore"])

                    with tab1:
                        col1, col2 = st.columns(2)
                        with col1:
                            left_cols = st.multiselect(f"Select Key Columns from {self.left_name}", self.left_df.columns, key="ms_left_cols")
                        with col2:
                            right_cols = st.multiselect(f"Select Key Columns from {self.right_name}", self.right_df.columns, key="ms_right_cols")

                    with tab2:
                        col1, col2 = st.columns(2)
                        with col1:
                            numeric_cols_left = self.left_df.select_dtypes(include=np.number).columns
                            left_amount_col = st.selectbox(f"Select Amount Column from {self.left_name}", numeric_cols_left, key="sb_left_amount_col")
                        with col2:
                            numeric_cols_right = self.right_df.select_dtypes(include=np.number).columns
                            right_amount_col = st.selectbox(f"Select Amount Column from {self.right_name}", numeric_cols_right, key="sb_right_amount_col")

                    with tab3:
                        items_to_ignore = st.text_input("Enter additional characters or phrases to ignore (comma-separated)", ".,/,*,-,&", key="ti_special_chars")

                    submitted = st.form_submit_button("🚀 Reconcile")

                # --- Run reconciliation ---
                if submitted:
                    if reconciliation_type == "Reconcile by Matching Sum" and (len(left_cols) != len(right_cols) or len(left_cols) == 0):
                        st.error("For 'Reconcile by Matching Sum', number of key columns must be the same for both datasets and greater than zero.")
                    elif reconciliation_type == "Reconcile Line by Line" and len(left_cols) != len(right_cols):
                        st.error("For 'Reconcile Line by Line', number of key columns must be the same for both datasets.")
                    else:
                        st.session_state.left_cols = left_cols
                        st.session_state.right_cols = right_cols
                        st.session_state.left_amount_col = left_amount_col
                        st.session_state.right_amount_col = right_amount_col
                        st.session_state.items_to_ignore = items_to_ignore

                        with st.spinner("Reconciling..."):
                            self.reconcile(left_cols, right_cols, left_amount_col, right_amount_col, items_to_ignore, reconciliation_type)
                            st.toast("Reconciliation complete! 🎉")

                        st.switch_page("pages/reconciliation_results.py")

    def reconcile(self, left_cols, right_cols, left_amount_col, right_amount_col, special_chars, reconciliation_type):
        reconciled_left_df = self.left_df.copy()
        reconciled_right_df = self.right_df.copy()
        reconciled_left_df['reconciliation_status'] = 'unreconciled'
        reconciled_right_df['reconciliation_status'] = 'unreconciled'
        reconciled_left_df['reconciliation_id'] = ''
        reconciled_right_df['reconciliation_id'] = ''

        # --- (use your existing reconciliation logic here, unchanged) ---

        st.session_state.reconciled_left_df = reconciled_left_df
        st.session_state.reconciled_right_df = reconciled_right_df
        st.session_state.left_name = self.left_name
        st.session_state.right_name = self.right_name

app = DataReconciliationApp()
app.run()
