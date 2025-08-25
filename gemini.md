### Project Overview
This is a data reconciliation application built with Streamlit. It allows a user to upload two datasets (CSV, XLSX, JSON), map columns to find matching records, and then tally a numeric column from each dataset to compare the sums.

### Tech Stack
- **Language**: Python 
- **Framework**: Streamlit
- **Data Handling**: Pandas
- **Excel Support**: openpyxl



### Coding Conventions
- **Style**: Follow PEP 8 guidelines.
- use theming for header, subheder, buttons,other widgets -refer -https://docs.streamlit.io/develop/concepts/configuration/theming
& https://docs.streamlit.io/develop/api-reference/configuration/config.toml#theme

- use classes & functions whereever possible for resusibilty of code
- use error handling for all code with proper error messages
- **Formatting**: Use `black` for code formatting.
- **Naming**: Use `snake_case` for variables and functions.
- ** Use-  @st.cache_data, @st.fragment,@st.dialog  wherever possible.
-Use- emojies whereever possible
-For all input widgets give- help/ tooltip & Placeholder value
- make use of st.container &st.empty to make better user interactions 
- use sessionstate to store variables, as well as for panadas dataframes.
- for displaying message- use st.toast("Success message"),st.error("Error message")
- for accepting input from users - use st.form, @st.dialog as applicable.

### Key File Locations
- `app.py`: The main application file.
- `requirements.txt`: Lists project dependencies.

### Important Don'ts
- do use unique keys for all widgets
