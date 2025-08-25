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
            def run(self):
                """
                Runs the main Streamlit application as the first page of a multipage app.
                """
                st.title("📊 Data Reconciliation App")

                if st.button("Clear Cache and Rerun"):
                    st.cache_data.clear()
                    st.rerun()

                # Reconciliation Type Selector
                st.radio(
                    "Select Reconciliation Method 🔄",
                    ["Sum-based", "Line-by-line"],
                    index=0,
                    key="reconciliation_type_radio",
                    help="Sum-based: Match by summing amounts for key columns. Line-by-line: Match amounts directly line by line.",
                    on_change=lambda: setattr(st.session_state, 'reconciliation_type', 
                                            'sum' if st.session_state.reconciliation_type_radio == "Sum-based" else 'line')
                )

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
                                st.dataframe(self.left_df.copy())
                            with col2:
                                st.write(self.right_name)
                                st.dataframe(self.right_df.copy())
                        with st.form("reconciliation_form"):
                            is_sum_based = st.session_state.reconciliation_type == "sum"

                            if is_sum_based:
                                tab1, tab2, tab3 = st.tabs(["Column Mapping", "Amount Column Selection", "Characters/Phrases to Ignore"])
                            else:
                                tab1, tab2, tab3 = st.tabs(["Optional Column Mapping", "Amount Column Selection", "Characters/Phrases to Ignore"])

                            with tab1:
                                st.subheader("🔗 Column Mapping")
                                if is_sum_based:
                                    st.info("Select matching key columns from both datasets for sum-based reconciliation")
                                else:
                                    st.info("Key columns are optional for line-by-line reconciliation. If selected, they'll be used as additional matching criteria.")

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
                            if is_sum_based and (len(left_cols) != len(right_cols) or len(left_cols) == 0):
                                st.error("For sum-based reconciliation, key columns must be selected and must be equal in number.")
                            elif not is_sum_based and ((len(left_cols) > 0 and len(right_cols) == 0) or (len(left_cols) == 0 and len(right_cols) > 0)):
                                st.error("If using optional key columns in line-by-line mode, they must be selected from both datasets.")
                            else:
                                st.session_state.left_amount_col = left_amount_col
                                st.session_state.right_amount_col = right_amount_col
                                with st.spinner("Reconciling..."):
                                    try:
                                        if is_sum_based:
                                            left_df, right_df = self._reconcile_by_sum(
                                                left_cols, right_cols, left_amount_col, right_amount_col, items_to_ignore
                                            )
                                        else:
                                            left_df, right_df = self._reconcile_line_by_line(
                                                left_amount_col, right_amount_col, left_cols, right_cols
                                            )

                                        if left_df is not None and right_df is not None:
                                            st.session_state.reconciled_left_df = left_df
                                            st.session_state.reconciled_right_df = right_df
                                            st.session_state.left_name = self.left_name
                                            st.session_state.right_name = self.right_name
                                            st.toast("Reconciliation complete! 🎉")
                                            st.switch_page("pages/1_Reconciliation_Results.py")
                                        else:
                                            st.error("Reconciliation failed. Please check your data and try again.")
                                    except Exception as e:
                                        st.error(f"An error occurred during reconciliation: {str(e)}")

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
            if len(left_cols) > 0 and len(right_cols) > 0:
                # Line by line with key columns
                left_df_cleaned = self.clean_key(self.left_df, left_cols, special_chars)
                right_df_cleaned = self.clean_key(self.right_df, right_cols, special_chars)

                left_df_cleaned['combined_key'] = left_df_cleaned[left_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
                right_df_cleaned['combined_key'] = right_df_cleaned[right_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

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


    def display_results(self):
        """
        Displays the reconciliation results in tabs.
        """
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🎉 Reconciliation Results")

        reconciled_left = st.session_state.reconciled_left_df[st.session_state.reconciled_left_df['reconciliation_status'] != 'unreconciled']
        reconciled_right = st.session_state.reconciled_right_df[st.session_state.reconciled_right_df['reconciliation_status'] != 'unreconciled']
        unreconciled_left = st.session_state.reconciled_left_df[st.session_state.reconciled_left_df['reconciliation_status'] == 'unreconciled']
        unreconciled_right = st.session_state.reconciled_right_df[st.session_state.reconciled_right_df['reconciliation_status'] == 'unreconciled']

        # Summary calculations
        left_amount_col = st.session_state.left_amount_col
        right_amount_col = st.session_state.right_amount_col

        total_left = st.session_state.reconciled_left_df[left_amount_col].sum()
        total_right = st.session_state.reconciled_right_df[right_amount_col].sum()
        diff = total_left - total_right

        total_only_left = unreconciled_left[left_amount_col].sum()
        total_only_right = unreconciled_right[right_amount_col].sum()
        diff_only = total_only_left - total_only_right

        summary_data = {
            "Description": [
                f"Total as per {st.session_state.left_name}",
                f"Total as per {st.session_state.right_name}",
                f"Difference (Total {st.session_state.left_name} - Total {st.session_state.right_name})",
                f"Total of only in {st.session_state.left_name}",
                f"Total of only in {st.session_state.right_name}",
                f"Difference (Total of only in {st.session_state.left_name} - Total of only in {st.session_state.right_name})",
            ],
            "Amount": [
                total_left,
                total_right,
                diff,
                total_only_left,
                total_only_right,
                diff_only,
            ],
        }
        summary_df = pd.DataFrame(summary_data)

        with col2:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name="Summary Reconciliation", index=False)
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

        tab1, tab2, tab3, tab4 = st.tabs([ "🛠️ Manual Reconciliation", "📊 Summary Reconciliation","✅ Reconciled Records", "❌ Unreconciled Records"])

        with tab2:
            st.subheader("📊 Summary Reconciliation")
            html_table = "<table>"
            html_table += "<thead><tr><th>Description</th><th>Amount</th></tr></thead>"
            html_table += "<tbody>"
            html_table += f"<tr><td>{summary_df['Description'][0]}</td><td>{summary_df['Amount'][0]:,.2f}</td></tr>"
            html_table += f"<tr><td>{summary_df['Description'][1]}</td><td>{summary_df['Amount'][1]:,.2f}</td></tr>"
            html_table += f"<tr><td><b>{summary_df['Description'][2]}</b></td><td><b>{summary_df['Amount'][2]:,.2f}</b></td></tr>"
            html_table += f"<tr><td>{summary_df['Description'][3]}</td><td>{summary_df['Amount'][3]:,.2f}</td></tr>"
            html_table += f"<tr><td>{summary_df['Description'][4]}</td><td>{summary_df['Amount'][4]:,.2f}</td></tr>"
            html_table += f"<tr><td><b>{summary_df['Description'][5]}</b></td><td><b>{summary_df['Amount'][5]:,.2f}</b></td></tr>"
            html_table += "</tbody></table>"
            st.markdown(html_table, unsafe_allow_html=True)

        with tab3:
            st.subheader("✅ Reconciled Records")
            col1,col2 = st.columns(2)
            
            with col1:
                st.write(f":green[Reconciled records from {st.session_state.left_name}]")
                st.dataframe(reconciled_left)
            with col2:
                st.write(f":green[Reconciled records from {st.session_state.right_name}]")
                st.dataframe(reconciled_right)

        with tab4:
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

                col1, col2 = st.columns(2, border=True)

                with col1:
                    unreconciled_left_manual_f = dataframe_filter_ui(unreconciled_left_manual, "left")
                with col2:
                    unreconciled_right_manual_f = dataframe_filter_ui(unreconciled_right_manual, "right")

                left_selection = st.session_state.get("left_manual_selection", {})
                left_selection_rows = left_selection.get("selection", {}).get("rows", [])
                right_selection = st.session_state.get("right_manual_selection", {})
                right_selection_rows = right_selection.get("selection", {}).get("rows", [])

                left_sum = 0
                selected_left_rows = pd.DataFrame()
                if left_selection_rows:
                    try:
                        selected_left_rows = unreconciled_left_manual_f.iloc[left_selection_rows]
                        left_sum = selected_left_rows[left_amount_col].sum()
                    except IndexError:
                        left_selection_rows = []

                right_sum = 0
                selected_right_rows = pd.DataFrame()
                if right_selection_rows:
                    try:
                        selected_right_rows = unreconciled_right_manual_f.iloc[right_selection_rows]
                        right_sum = selected_right_rows[right_amount_col].sum()
                    except IndexError:
                        right_selection_rows = []

                with col1:
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        st.write(f"Unreconciled records from {st.session_state.left_name}")
                    with sub_col2:
                        st.markdown(f"**Selected Sum:** {left_sum:,.2f}")
                    st.dataframe(unreconciled_left_manual_f, key="left_manual_selection", on_select="rerun", selection_mode="multi-row")

                with col2:
                    sub_col1, sub_col2 = st.columns(2)
                    with sub_col1:
                        st.write(f"Unreconciled records from {st.session_state.right_name}")
                    with sub_col2:
                        st.markdown(f"**Selected Sum:** {right_sum:,.2f}")
                    st.dataframe(unreconciled_right_manual_f, key="right_manual_selection", on_select="rerun", selection_mode="multi-row")

                if left_selection_rows and right_selection_rows:
                    if np.isclose(left_sum, right_sum) and left_sum != 0:
                        if st.button("Mark as Manually Reconciled", key="manual_reconcile_button", icon="✅", use_container_width=True):
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