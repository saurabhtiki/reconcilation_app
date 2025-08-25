import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from dataframe_filter_module import dataframe_filter_ui

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

        for col in columns:
            df_copy[col] = df_copy[col].astype(str).str.lower()
            # Remove all spaces from the column
            df_copy[col] = df_copy[col].str.replace(' ', '', regex=False)
            for item in items:
                df_copy[col] = df_copy[col].str.replace(item, '', regex=False)
        return df_copy

    def run(self):
        """
        Runs the main Streamlit application.
        """
        st.title("📊 Data Reconciliation App")

        if st.button("Clear Cache and Rerun"):
            st.cache_data.clear()
            st.rerun()

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
                            left_cols = st.multiselect(f"Select Key Columns from {self.left_name}", self.left_df.columns, key="ms_left_cols", help="Select the columns to use as keys for reconciliation from the left dataset.")
                        with col2:
                            right_cols = st.multiselect(f"Select Key Columns from {self.right_name}", self.right_df.columns, key="ms_right_cols", help="Select the columns to use as keys for reconciliation from the right dataset.")

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
                    if len(left_cols) != len(right_cols) or len(left_cols) == 0:
                        st.error("Number of key columns must be the same for both datasets and greater than zero.")
                    else:
                        st.session_state.left_amount_col = left_amount_col
                        st.session_state.right_amount_col = right_amount_col
                        with st.spinner("Reconciling..."):
                            self.reconcile(left_cols, right_cols, left_amount_col, right_amount_col, items_to_ignore)
                            st.toast("Reconciliation complete! 🎉")

                if 'reconciled_left_df' in st.session_state:
                    self.display_results()

    def reconcile(self, left_cols: list, right_cols: list, left_amount_col: str, right_amount_col: str, special_chars: str):
        """
        Performs the reconciliation process.

        Args:
            left_cols (list): Key columns from the left dataset.
            right_cols (list): Key columns from the right dataset.
            left_amount_col (str): Amount column from the left dataset.
            right_amount_col (str): Amount column from the right dataset.
            special_chars (str): Special characters to ignore.
        """
        left_df_cleaned = self.clean_key(self.left_df, left_cols, special_chars)
        right_df_cleaned = self.clean_key(self.right_df, right_cols, special_chars)

        left_df_cleaned['combined_key'] = left_df_cleaned[left_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        right_df_cleaned['combined_key'] = right_df_cleaned[right_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        left_sum = left_df_cleaned.groupby('combined_key')[left_amount_col].sum().reset_index()
        right_sum = right_df_cleaned.groupby('combined_key')[right_amount_col].sum().reset_index()

        left_sum = left_sum.rename(columns={left_amount_col: 'amount_left'})
        right_sum = right_sum.rename(columns={right_amount_col: 'amount_right'})

        merged_df = pd.merge(left_sum, right_sum, on='combined_key', how='outer')
        merged_df = merged_df.fillna(0)

        merged_df['reconciled'] = np.where(np.isclose(merged_df['amount_left'], merged_df['amount_right']), 'yes', 'no')

        reconciled_keys = merged_df[merged_df['reconciled'] == 'yes']['combined_key']

        reconciled_left_df = self.left_df.copy()
        reconciled_right_df = self.right_df.copy()

        reconciled_left_df['reconciliation_status'] = 'unreconciled'
        reconciled_right_df['reconciliation_status'] = 'unreconciled'
        reconciled_left_df['reconciliation_id'] = ''
        reconciled_right_df['reconciliation_id'] = ''

        for key in reconciled_keys:
            left_indices = left_df_cleaned[left_df_cleaned['combined_key'] == key].index
            right_indices = right_df_cleaned[right_df_cleaned['combined_key'] == key].index

            reconciled_left_df.loc[left_indices, 'reconciliation_status'] = 'reconciled'
            reconciled_right_df.loc[right_indices, 'reconciliation_status'] = 'reconciled'

            left_ids_str = ','.join(map(str, right_indices.tolist()))
            right_ids_str = ','.join(map(str, left_indices.tolist()))

            reconciled_left_df.loc[left_indices, 'reconciliation_id'] = left_ids_str
            reconciled_right_df.loc[right_indices, 'reconciliation_id'] = right_ids_str

        st.session_state.reconciled_left_df = reconciled_left_df
        st.session_state.reconciled_right_df = reconciled_right_df
        st.session_state.left_name = self.left_name
        st.session_state.right_name = self.right_name


    def display_results(self):
        """
        Displays the reconciliation results in tabs.
        """
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🎉 Reconciliation Results")
        
        with col2:
            reconciled_left = st.session_state.reconciled_left_df[st.session_state.reconciled_left_df['reconciliation_status'] != 'unreconciled']
            reconciled_right = st.session_state.reconciled_right_df[st.session_state.reconciled_right_df['reconciliation_status'] != 'unreconciled']
            unreconciled_left = st.session_state.reconciled_left_df[st.session_state.reconciled_left_df['reconciliation_status'] == 'unreconciled']
            unreconciled_right = st.session_state.reconciled_right_df[st.session_state.reconciled_right_df['reconciliation_status'] == 'unreconciled']

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                reconciled_left.to_excel(writer, sheet_name=f"Reconciled_{st.session_state.left_name}", index=True)
                reconciled_right.to_excel(writer, sheet_name=f"Reconciled_{st.session_state.right_name}", index=True)
                unreconciled_left.to_excel(writer, sheet_name=f"Unreconciled_{st.session_state.left_name}", index=True)
                unreconciled_right.to_excel(writer, sheet_name=f"Unreconciled_{st.session_state.right_name}", index=True)

            excel_data = output.getvalue()

            st.download_button(
                label="📥 Download All as Excel",
                data=excel_data,
                file_name="reconciliation_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_button"
            )

        tab1, tab2, tab3 = st.tabs([ "🛠️ Manual Reconciliation","✅ Reconciled Records", "❌ Unreconciled Records"])

        with tab2:
            st.subheader("✅ Reconciled Records")
            col1,col2 = st.columns(2)
            
            with col1:
                st.write(f":green[Reconciled records from {st.session_state.left_name}]")
                st.dataframe(reconciled_left)
            with col2:
                st.write(f":green[Reconciled records from {st.session_state.right_name}]")
                st.dataframe(reconciled_right)

        with tab3:
            st.subheader("❌ Unreconciled Records")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f":red[Unreconciled records from {st.session_state.left_name}]")
                st.dataframe(unreconciled_left)
            with col2:
                st.write(f":red[Unreconciled records from {st.session_state.right_name}]")
                st.dataframe(unreconciled_right)

        with tab1:
            @st.fragment
            def manual_reconciliation_fragment():
                st.subheader("🔧 Manual Reconciliation")

                if 'left_amount_col' not in st.session_state or 'right_amount_col' not in st.session_state:
                    st.warning("Please run the reconciliation first to enable manual reconciliation.")
                    return

                left_amount_col = st.session_state.left_amount_col
                right_amount_col = st.session_state.right_amount_col

                unreconciled_left_manual = st.session_state.reconciled_left_df[st.session_state.reconciled_left_df['reconciliation_status'] == 'unreconciled']
                unreconciled_right_manual = st.session_state.reconciled_right_df[st.session_state.reconciled_right_df['reconciliation_status'] == 'unreconciled']

                left_selection = st.session_state.get("left_manual_selection", {})
                left_selection_rows = left_selection.get("selection", {}).get("rows", [])
                right_selection = st.session_state.get("right_manual_selection", {})
                right_selection_rows = right_selection.get("selection", {}).get("rows", [])

                left_sum = 0
                if left_selection_rows:
                    selected_left_rows = unreconciled_left_manual.iloc[left_selection_rows]
                    left_sum = selected_left_rows[left_amount_col].sum()

                right_sum = 0
                if right_selection_rows:
                    selected_right_rows = unreconciled_right_manual.iloc[right_selection_rows]
                    right_sum = selected_right_rows[right_amount_col].sum()

                col1, col2 = st.columns(2,border=True)

                with col1:
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        st.write(f"Unreconciled records from {st.session_state.left_name}")
                    with sub_col2:
                        st.markdown(f"**Selected Sum:** {left_sum:,.2f}")
                    unreconciled_left_manual_f=dataframe_filter_ui(unreconciled_left_manual,"left")
                    st.dataframe(unreconciled_left_manual_f, key="left_manual_selection", on_select="rerun", selection_mode="multi-row")

                with col2:
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        st.write(f"Unreconciled records from {st.session_state.right_name}")
                    with sub_col2:
                        st.markdown(f"**Selected Sum:** {right_sum:,.2f}")
                    unreconciled_right_manual_f=dataframe_filter_ui(unreconciled_right_manual,"right")
                    st.dataframe(unreconciled_right_manual_f, key="right_manual_selection", selection_mode="multi-row")

                if left_selection_rows and right_selection_rows:
                    #st.metric("Difference", f"{left_sum - right_sum:,.2f}")

                    if np.isclose(left_sum, right_sum) and left_sum != 0:
                        if st.button("Mark as Manually Reconciled", key="manual_reconcile_button"):
                            left_indices_to_update = selected_left_rows.index
                            right_indices_to_update = selected_right_rows.index

                            st.session_state.reconciled_left_df.loc[left_indices_to_update, 'reconciliation_status'] = 'Manually Reconciled'
                            st.session_state.reconciled_right_df.loc[right_indices_to_update, 'reconciliation_status'] = 'Manually Reconciled'

                            left_ids_str = ','.join(map(str, right_indices_to_update.tolist()))
                            right_ids_str = ','.join(map(str, left_indices_to_update.tolist()))

                            st.session_state.reconciled_left_df.loc[left_indices_to_update, 'reconciliation_id'] = left_ids_str
                            st.session_state.reconciled_right_df.loc[right_indices_to_update, 'reconciliation_id'] = right_ids_str
                            
                            del st.session_state.left_manual_selection
                            del st.session_state.right_manual_selection

                            st.toast("Rows manually reconciled! ✅")
                            st.rerun()
            
            manual_reconciliation_fragment()

if __name__ == '__main__':
    app = DataReconciliationApp()
    app.run()
