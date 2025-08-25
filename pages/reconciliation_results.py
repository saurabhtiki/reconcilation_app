import streamlit as st
import pandas as pd
import numpy as np
import io
from dataframe_filter_module import dataframe_filter_ui

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

# Call the display_results function when this page is accessed
if __name__ == "__main__":
    if 'reconciled_left_df' in st.session_state:
        display_results()
    else:
        st.warning("No reconciliation results found. Please go back to the input page and run a reconciliation.")