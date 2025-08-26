import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from dataframe_filter_module import dataframe_filter_ui

# All functions are now standalone, not part of a class.

@st.fragment
def display_results():
    """
    Displays the reconciliation results in tabs.
    """
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("🎉 Reconciliation Results")
    #arrange columns to show Key columns amount columns & then other columns

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
            #list of Key columns to left &then adding list amount columns
            key_cols_left = st.session_state.left_cols + [st.session_state.left_amount_col]# Assuming both sides have the same key columns
            key_cols_right = st.session_state.right_cols+ [st.session_state.right_amount_col]
               
            # Get the remaining columns
            remaining_cols_left = [col for col in unreconciled_left_manual.columns if col not in key_cols_left]
            remaining_cols_right = [col for col in unreconciled_right_manual.columns if col not in key_cols_right]

            # Combine for the desired order
            column_order_left = key_cols_left + remaining_cols_left 
            column_order_right = key_cols_right + remaining_cols_right   

            col1, col2 = st.columns(2, border=True)

            with col1:
                unreconciled_left_manual_f = dataframe_filter_ui(unreconciled_left_manual, "left")
            with col2:
                unreconciled_right_manual_f = dataframe_filter_ui(unreconciled_right_manual, "right")

            left_selection = st.session_state.get("left_manual_selection", {})
            left_selection_rows = left_selection.get("selection", {}).get("rows", [])
            right_selection = st.session_state.get("right_manual_selection", {})
            right_selection_rows = right_selection.get("selection", {}).get("rows", [])
            if 'left_sum' not in st.session_state:
                st.session_state.left_sum = 0
            #left_sum = 0
            selected_left_rows = pd.DataFrame()
            if left_selection_rows:
                try:
                    selected_left_rows = unreconciled_left_manual_f.iloc[left_selection_rows]
                    if st.session_state.roundoff_option == "Yes":
                        st.session_state.left_sum = selected_left_rows[left_amount_col].apply(lambda x: round(x, 0))
                    else:
                         st.session_state.left_sum = selected_left_rows[left_amount_col].sum()
                except IndexError:
                    left_selection_rows = []
            if 'right_sum' not in st.session_state:
                st.session_state.right_sum = 0
            #right_sum = 0
            selected_right_rows = pd.DataFrame()
            if right_selection_rows:
                try:
                    selected_right_rows = unreconciled_right_manual_f.iloc[right_selection_rows]
                    if st.session_state.roundoff_option == "Yes":
                        st.session_state.right_sum = selected_right_rows[right_amount_col].apply(lambda x: round(x, 0))
                    else:
                        st.session_state.right_sum = selected_right_rows[right_amount_col].sum()
                except IndexError:
                    right_selection_rows = []

            with col1:
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    st.write(f":red[Unreconciled records from {st.session_state.left_name}]")
                with sub_col2:
                    selected_sumL = float(st.session_state.left_sum.sum()) if hasattr(st.session_state.left_sum, 'sum') else float(st.session_state.left_sum)
                    st.markdown(f"**Selected Sum:** {selected_sumL:,.2f}")
                    #st.markdown(f"**Selected Sum:** {st.session_state.left_sum}")
                st.dataframe(unreconciled_left_manual_f, key="left_manual_selection", on_select="rerun", selection_mode="multi-row",column_order=column_order_left)

            with col2:
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    st.write(f":red[Unreconciled records from {st.session_state.right_name}]")
                with sub_col2:
                    selected_sumR = float(st.session_state.right_sum.sum()) if hasattr(st.session_state.right_sum, 'sum') else float(st.session_state.right_sum)
                    st.markdown(f"**Selected Sum:** {selected_sumR:,.2f}")
                st.dataframe(unreconciled_right_manual_f, key="right_manual_selection", on_select="rerun", selection_mode="multi-row",column_order=column_order_right)

            if left_selection_rows and right_selection_rows:
                #if difference is -1 to 1 the allow reconciliation
                if abs(selected_sumL - selected_sumR) <= 1 and abs(float(st.session_state.left_sum.sum()) - float(st.session_state.right_sum.sum())) <= 1 and float(st.session_state.left_sum.sum()) != 0 and float(st.session_state.right_sum.sum()) != 0:
                #if np.isclose(st.session_state.left_sum, st.session_state.right_sum) and st.session_state.left_sum != 0:
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
            df_copy[col] = df_copy[col].str.replace(item.lower(), '', regex=False)
    return df_copy

def reconcile(left_cols: list, right_cols: list, left_amount_col: str, right_amount_col: str, special_chars: str, reconciliation_type: str):
    reconciled_left_df = st.session_state.left_df.copy()
    reconciled_right_df = st.session_state.right_df.copy()

    reconciled_left_df['reconciliation_status'] = 'unreconciled'
    reconciled_right_df['reconciliation_status'] = 'unreconciled'
    reconciled_left_df['reconciliation_id'] = ''
    reconciled_right_df['reconciliation_id'] = ''

    if reconciliation_type == "Reconcile by Matching Sum":
        left_df_cleaned = clean_key(st.session_state.left_df, left_cols, special_chars)
        right_df_cleaned = clean_key(st.session_state.right_df, right_cols, special_chars)
        left_df_cleaned['combined_key'] = left_df_cleaned[left_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        right_df_cleaned['combined_key'] = right_df_cleaned[right_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        left_sum = left_df_cleaned.groupby('combined_key')[left_amount_col].sum().reset_index()
        right_sum = right_df_cleaned.groupby('combined_key')[right_amount_col].sum().reset_index()
        if st.session_state.roundoff_option == "No":
            left_sum = left_sum.rename(columns={left_amount_col: 'amount_left'})
            right_sum = right_sum.rename(columns={right_amount_col: 'amount_right'})
        else:
            left_sum['amount_left'] = left_sum[left_amount_col].apply(lambda x: round(x, 0))
            right_sum['amount_right'] = right_sum[right_amount_col].apply(lambda x: round(x, 0))
        merged_df = pd.merge(left_sum, right_sum, on='combined_key', how='outer').fillna(0)
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
        if len(left_cols) > 0 and len(right_cols) > 0:
            left_df_cleaned = clean_key(st.session_state.left_df, left_cols, special_chars)
            right_df_cleaned = clean_key(st.session_state.right_df, right_cols, special_chars)
            left_df_cleaned['combined_key'] = left_df_cleaned[left_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            right_df_cleaned['combined_key'] = right_df_cleaned[right_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            temp_merged = pd.merge(
                left_df_cleaned.reset_index().rename(columns={'index':'original_left_index'}),
                right_df_cleaned.reset_index().rename(columns={'index':'original_right_index'}),
                left_on=['combined_key', left_amount_col],
                right_on=['combined_key', right_amount_col],
                how='inner',
                suffixes=('_left', '_right')
            )
            temp_merged.drop_duplicates(subset=['original_left_index'], keep='first', inplace=True)
            temp_merged.drop_duplicates(subset=['original_right_index'], keep='first', inplace=True)
            for idx_left, idx_right in zip(temp_merged['original_left_index'], temp_merged['original_right_index']):
                reconciled_left_df.loc[idx_left, 'reconciliation_status'] = 'reconciled'
                reconciled_right_df.loc[idx_right, 'reconciliation_status'] = 'reconciled'
                reconciled_left_df.loc[idx_left, 'reconciliation_id'] = str(idx_right)
                reconciled_right_df.loc[idx_right, 'reconciliation_id'] = str(idx_left)
        else:
            left_sorted = st.session_state.left_df.sort_values(by=left_amount_col).reset_index(drop=False)
            right_sorted = st.session_state.right_df.sort_values(by=right_amount_col).reset_index(drop=False)
            left_sorted['reconciliation_status'] = 'unreconciled'
            right_sorted['reconciliation_status'] = 'unreconciled'
            left_sorted['reconciliation_id'] = ''
            right_sorted['reconciliation_id'] = ''
            i, j = 0, 0
            while i < len(left_sorted) and j < len(right_sorted):
                if np.isclose(left_sorted.loc[i, left_amount_col], right_sorted.loc[j, right_amount_col]):
                    left_sorted.loc[i, 'reconciliation_status'] = 'reconciled'
                    right_sorted.loc[j, 'reconciliation_status'] = 'reconciled'
                    left_sorted.loc[i, 'reconciliation_id'] = str(right_sorted.loc[j, 'index'])
                    right_sorted.loc[j, 'reconciliation_id'] = str(left_sorted.loc[i, 'index'])
                    i += 1
                    j += 1
                elif left_sorted.loc[i, left_amount_col] < right_sorted.loc[j, right_amount_col]:
                    i += 1
                else:
                    j += 1
            for original_idx, row in left_sorted.iterrows():
                reconciled_left_df.loc[row['index'], 'reconciliation_status'] = row['reconciliation_status']
                reconciled_left_df.loc[row['index'], 'reconciliation_id'] = row['reconciliation_id']
            for original_idx, row in right_sorted.iterrows():
                reconciled_right_df.loc[row['index'], 'reconciliation_status'] = row['reconciliation_status']
                reconciled_right_df.loc[row['index'], 'reconciliation_id'] = row['reconciliation_id']

    st.session_state.reconciled_left_df = reconciled_left_df
    st.session_state.reconciled_right_df = reconciled_right_df
    st.session_state.left_name = st.session_state.left_name # This was missing
    st.session_state.right_name = st.session_state.right_name # This was missing

def show_setup_ui():
    """Shows the UI for uploading files and configuring reconciliation."""
    with st.expander("ℹ️ Instructions", expanded=False):
        st.markdown("""
        1. Upload two datasets (CSV or Excel).
        2. Select key columns and amount columns.
        3. Choose reconciliation type and options.
        4. Click 'Reconcile' to see results in the 'Results' tab.
        """)
    with st.expander("📂 Upload Data & ⚙️ Set Parameters", expanded=True):
        if st.button("Clear Cache and Rerun"):
            st.cache_data.clear()
            st.rerun()
        
        col1, col2 = st.columns([3,2],border=True)
        with col1:
            reconciliation_type = st.radio("Select Reconciliation Type:", ("Reconcile by Matching Sum", "Reconcile Line by Line"), index=0, key="reconciliation_type_radio", horizontal=True, help="...")
            st.session_state.reconciliation_type = reconciliation_type
        with col2:
            roundoff_option = st.radio("Match Round-off Value?", ("Yes", "No"), index=1, key="roundoff_option_radio", horizontal=True, help="...")
            st.session_state.roundoff_option = roundoff_option

        col1, col2 = st.columns(2)
        with col1:
            left_file = st.file_uploader("📤 Upload Left Dataset", type=['csv', 'xlsx'], key="fu_left_file")
            if left_file:
                st.session_state.left_name = os.path.splitext(left_file.name)[0]
            left_skiprows = st.number_input(f"Skip rows at top of {st.session_state.get('left_name', 'Left Dataset')}", min_value=0, value=0, key="nu_left_skiprows")
        with col2:
            right_file = st.file_uploader("📥 Upload Right Dataset", type=['csv', 'xlsx'], key="fu_right_file")
            if right_file:
                st.session_state.right_name = os.path.splitext(right_file.name)[0]
            right_skiprows = st.number_input(f"Skip rows at top of {st.session_state.get('right_name', 'Right Dataset')}", min_value=0, value=0, key="nu_right_skiprows")

    if left_file and right_file:
        st.session_state.left_df = load_data(left_file, left_skiprows)
        st.session_state.right_df = load_data(right_file, right_skiprows)

        if st.session_state.left_df is not None and st.session_state.right_df is not None:
            st.subheader("📄 Original Dataframes")
            with st.expander("View Original Dataframes"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(st.session_state.left_name)
                    st.dataframe(st.session_state.left_df)
                with col2:
                    st.write(st.session_state.right_name)
                    st.dataframe(st.session_state.right_df)

            with st.form("reconciliation_form"):
                tab1, tab2, tab3 = st.tabs(["Column Mapping", "Amount Column Selection", "Characters/Phrases to Ignore"])
                with tab1:
                    st.subheader("🔗 Column Mapping")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.session_state.left_cols = st.multiselect(f"Select Key Columns from {st.session_state.left_name}", st.session_state.left_df.columns, key="ms_left_cols")
                    with col2:
                        st.session_state.right_cols = st.multiselect(f"Select Key Columns from {st.session_state.right_name}", st.session_state.right_df.columns, key="ms_right_cols")
                with tab2:
                    st.subheader("💰 Amount Column Selection")
                    col1, col2 = st.columns(2)
                    with col1:
                        left_amount_col = st.selectbox(f"Select Amount Column from {st.session_state.left_name}", st.session_state.left_df.select_dtypes(include=np.number).columns, key="sb_left_amount_col")
                    with col2:
                        right_amount_col = st.selectbox(f"Select Amount Column from {st.session_state.right_name}", st.session_state.right_df.select_dtypes(include=np.number).columns, key="sb_right_amount_col")
                with tab3:
                    st.subheader("✨ Characters/Phrases to Ignore")
                    items_to_ignore = st.text_input("Enter additional characters or phrases to ignore (comma-separated)", ".,/,*,-,&", key="ti_special_chars")
                submitted = st.form_submit_button("🚀 Reconcile")

            if submitted:
                if st.session_state.reconciliation_type == "Reconcile by Matching Sum" and (len(st.session_state.left_cols) != len(st.session_state.right_cols) or len(st.session_state.left_cols) == 0):
                    st.error("Error...")
                elif st.session_state.reconciliation_type == "Reconcile Line by Line" and len(st.session_state.left_cols) != len(st.session_state.right_cols):
                    st.error("Error...")
                else:
                    st.session_state.left_amount_col = left_amount_col
                    st.session_state.right_amount_col = right_amount_col
                    with st.spinner("Reconciling..."):
                        reconcile(st.session_state.left_cols, st.session_state.right_cols, left_amount_col, right_amount_col, items_to_ignore, st.session_state.reconciliation_type)
                        st.toast("Reconciliation complete! Click the 'Results' tab to view.", icon="🎉")

def login_screen():
    st.title("📊 :blue[Data Reconciliation App]")
    with st.container(border=True):
        #st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png", width=200)    
        st.subheader("Please log in.")
        st.button("Log in with Google", on_click=st.login)

def run():
    st.title("📊 :blue[Data Reconciliation App]")
    
    setup_tab, results_tab = st.tabs([":orange[**Reconciliation Setup**]", ":orange[**Results**]"])
    with setup_tab:
        show_setup_ui()
    with results_tab:
        if "reconciled_left_df" in st.session_state:
            display_results()
        else:
            st.info("Please complete the reconciliation setup first. The results will appear here.")

# --- Main script execution ---
st.set_page_config(layout="wide")
if 'left_df' not in st.session_state: st.session_state.left_df = None
if 'right_df' not in st.session_state: st.session_state.right_df = None
if 'left_name' not in st.session_state: st.session_state.left_name = "Left Dataset"
if 'right_name' not in st.session_state: st.session_state.right_name = "Right Dataset"


if not st.user.is_logged_in:
        login_screen()
else:
    if st.user.email in st.secrets["whitelist"]:
        run()
        with st.sidebar:
            if st.button("Log out"):
                st.logout()
            st.subheader(f":green[Welcome, {st.user.name}!]")
            st.image(st.user.picture, width=100)
            
    else:
        st.error("You are not authorized to access this app.")
