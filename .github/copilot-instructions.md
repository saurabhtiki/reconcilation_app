# Data Reconciliation App - AI Agent Instructions

This is a Streamlit-based data reconciliation application designed to compare and reconcile two datasets. Here's what you need to know to work effectively with this codebase:

## Project Architecture

### Core Components
- `app_reco.py`: Main Streamlit application entry point
  - Handles file uploads (CSV/Excel)
  - Manages data cleaning and reconciliation logic
  - Uses a class-based structure with `DataReconciliationApp` as the main component
- `dataframe_filter_module.py`: Reusable filtering module
  - Provides UI components for DataFrame filtering
  - Uses session state for managing filter states
  - Supports multiple filter types (multiselect, slider, date, text)

### Key Patterns
1. Data Loading:
   ```python
   @st.cache_data  # Performance optimization for data loading
   def load_data(uploaded_file, skiprows)
   ```

2. Key Column Cleaning:
   ```python
   @staticmethod
   @st.cache_data
   def clean_key(df, columns, items_to_ignore)
   ```

## Development Workflow

### Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run app_reco.py
   ```

### Important Conventions
- Use `@st.cache_data` for expensive operations
- Follow snake_case for function and variable names
- Use type hints for function parameters
- Document functions with docstrings

## Integration Points
- Streamlit for UI components
- Pandas for data manipulation
- File system integration for CSV/Excel uploads

## State Management
- Uses Streamlit's session state for managing UI state
- Filter states are managed in `dataframe_filter_module.py`
- Key state variables use consistent prefixes for organization

## Best Practices
- Always clean input data using the `clean_key` method before reconciliation
- Use the provided caching decorators for performance
- Handle file upload errors gracefully
- Follow existing error handling patterns:
  ```python
  try:
      # data loading/processing
  except Exception as e:
      st.error(f"Error loading file: {e}")
  ```
