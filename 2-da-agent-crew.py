
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from scipy.stats import skew, kurtosis
import os
from datetime import datetime
import warnings
from io import StringIO # Import StringIO explicitly
from dotenv import load_dotenv
# For CrewAI and LLM integration
from crewai import Agent, Task, Crew, Process,LLM
from crewai_tools import FileReadTool

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0,
)

# Suppress harmless warnings from libraries like seaborn
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Crew Leader / User Interface Agent (The "Project Manager") ---
class CrewLeader:
    def __init__(self):
        self.file_path = None
        self.problem_type = None
        self.label_column = None
        self.df = None
        self.output_dir = "data_analysis_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.report_filepath = os.path.join(self.output_dir, "Comprehensive_Analysis_Report.md")
        self.cleaned_data_filepath = None
        self.log_file = os.path.join(self.output_dir, "analysis_log.txt")
        self._initialize_log()

    def _initialize_log(self):
        with open(self.log_file, 'w') as f:
            f.write(f"Data Analysis Log - Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 50 + "\n\n")

    def _log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        print(message) # Also print to console for immediate feedback

    def get_user_inputs(self):
        self._log("Welcome to the Data Analysis Crew!")
        while True:
            self.file_path = input("Enter the path to your CSV or Excel file: ")
            if not os.path.exists(self.file_path):
                self._log("Error: File not found. Please try again.")
                continue
            try:
                if self.file_path.lower().endswith('.csv'):
                    self.df = pd.read_csv(self.file_path)
                elif self.file_path.lower().endswith(('.xls', '.xlsx')):
                    self.df = pd.read_excel(self.file_path)
                else:
                    self._log("Error: Unsupported file format. Please provide a .csv or .xlsx file.")
                    continue
                self._log(f"File '{os.path.basename(self.file_path)}' loaded successfully.")
                self._log(f"Initial DataFrame shape: {self.df.shape}")
                break
            except Exception as e:
                self._log(f"Error loading file: {e}. Please check file integrity and permissions.")

        while True:
            problem_choice = input("Is this a (C)lassification or (R)egression problem? (C/R): ").upper()
            if problem_choice == 'C':
                self.problem_type = 'classification'
                break
            elif problem_choice == 'R':
                self.problem_type = 'regression'
                break
            else:
                self._log("Invalid choice. Please enter 'C' or 'R'.")

        available_columns = self.df.columns.tolist()
        self._log(f"Available columns: {', '.join(available_columns)}")
        while True:
            self.label_column = input("Enter the name of the label (target) column: ")
            if self.label_column in available_columns:
                break
            else:
                self._log(f"Error: '{self.label_column}' not found in the dataset. Please enter a valid column name.")
        self._log(f"Analysis setup complete. Problem type: {self.problem_type}, Target column: {self.label_column}")

    def orchestrate_analysis(self):
        self._log("\n--- Initiating Data Profiler & Cleaner Agent ---")
        profiler_cleaner = DataProfilerCleaner(self.df.copy(), self.output_dir)
        initial_clean_df, preliminary_report_sections, transformation_instructions = profiler_cleaner.run_profiling_and_cleaning()
        self._log("Data Profiler & Cleaner Agent finished.")
        self._log(f"Preliminary cleaning resulted in a DataFrame of shape: {initial_clean_df.shape}")

        self._log("\n--- Initiating EDA & Reporting Agent ---")
        eda_reporter = EDAReportingAgent(initial_clean_df.copy(), self.label_column, self.problem_type, self.output_dir)
        detailed_report_sections, eda_transform_suggestions = eda_reporter.run_eda_and_reporting()
        self._log("EDA & Reporting Agent finished. Generating comprehensive report.")

        # Combine reports
        full_report_content = ["# Comprehensive Data Analysis Report\n"]
        full_report_content.extend(preliminary_report_sections)
        full_report_content.extend(detailed_report_sections)

        # Append final transformation suggestions for the user to review before execution
        full_report_content.append("\n## Transformation Recommendations (for User Review before Execution)\n")
        full_report_content.append("Based on the analysis, here are the suggested transformations:\n")
        if not eda_transform_suggestions:
            full_report_content.append("- No specific transformations recommended at this stage beyond initial cleaning.\n")
        else:
            for key, value in eda_transform_suggestions.items():
                full_report_content.append(f"- **{key.replace('_', ' ').title()}:** {value}\n")

        with open(self.report_filepath, 'w') as f:
            f.write("\n".join(full_report_content))
        self._log(f"Comprehensive analysis report saved to: {self.report_filepath}")


        # --- LLM Agentic Analysis ---
        self._log("\n--- Initiating LLM Agentic Analysis ---")
        llm_analysis_agent = LLMAgenticAnalysis(
            self.report_filepath,
            self.output_dir,
            self.df.shape, # Pass original shape as context for LLM
            self.label_column,
            self.problem_type
        )
        llm_response_to_user = llm_analysis_agent.run_agentic_analysis()

        self._log("\n--- LLM's Summary and Proposed Actions ---")
        print(llm_response_to_user) # This will be the output from the ActionSuggester agent

        # Now, take user input based on the LLM's prompt
        user_choice = input("\nWhat would you like to do next based on the analysis? Please type your command (e.g., 'apply transformations', 'skip transformations'): ").lower()

        if "apply transformations" in user_choice or "execute transformations" in user_choice:
            self._log("Executing recommended transformations...")
            executor = DataTransformationExecutor(initial_clean_df.copy(), self.output_dir, self.label_column, self.problem_type)
            final_clean_df, execution_summary = executor.run_transformations(eda_transform_suggestions)
            self.cleaned_data_filepath = os.path.join(
                self.output_dir,
                f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            final_clean_df.to_csv(self.cleaned_data_filepath, index=False)
            self._log(f"Final cleaned data saved to: {self.cleaned_data_filepath}")
            self._log("\n--- Execution Summary ---")
            self._log(execution_summary)
        elif "skip transformations" in user_choice or "no transformations" in user_choice:
            self._log("Skipping transformation execution as per user request. No new cleaned CSV will be generated based on recommendations.")
        else:
            self._log(f"Unrecognized command: '{user_choice}'. No further automated action taken.")

        self._log("\nData Analysis Crew: Mission Complete!")


# --- Crew Member 1: Data Profiler & Cleaner Agent (The "Sanitizer") ---
class DataProfilerCleaner:
    def __init__(self, df, output_dir):
        self.df = df
        self.output_dir = output_dir
        self.report_sections = []
        self.transformation_instructions = {} # Not used by this class, but passed through for later consideration

    def _log_report(self, title, content):
        self.report_sections.append(f"\n## {title}\n")
        self.report_sections.append(content)

    def run_profiling_and_cleaning(self):
        self.report_sections.append("# Preliminary Data Cleaning & Profiling Report\n")

        # Fix for the AttributeError: Use a StringIO object to capture the output of df.info()
        buffer = StringIO()
        self.df.info(buf=buffer)
        info_string = buffer.getvalue()
        self._log_report("Initial Data Overview", info_string)

        self._log_report("Descriptive Statistics (Numerical)", self.df.describe().to_markdown())
        self._log_report("Descriptive Statistics (Categorical)", self.df.describe(include='object').to_markdown())


        # 1. Handle Duplicates
        num_duplicates = self.df.duplicated().sum()
        if num_duplicates > 0:
            self.df.drop_duplicates(inplace=True)
            self._log_report("Duplicate Rows Removed", f"Identified and removed {num_duplicates} duplicate rows.")
        else:
            self._log_report("Duplicate Rows", "No duplicate rows found.")
        self._log_report("DataFrame Shape After Duplicate Removal", f"Current shape: {self.df.shape}")

        # 2. Handle Missing Values
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        missing_info_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage (%)': missing_percentage
        })
        missing_info_df = missing_info_df[missing_info_df['Missing Count'] > 0].sort_values(by='Missing Percentage (%)', ascending=False)

        if not missing_info_df.empty:
            self._log_report("Missing Values Analysis", missing_info_df.to_markdown())
            self.report_sections.append("\n**Recommendations for Missing Values:**\n")
            for col in missing_info_df.index:
                if self.df[col].dtype in ['int64', 'float64']:
                    self.report_sections.append(f"- Column `{col}` ({missing_info_df.loc[col, 'Missing Percentage (%)']:.2f}% missing): Consider imputing with mean/median or advanced methods if appropriate, or dropping if too many missing.")
                else: # Categorical/Object
                    self.report_sections.append(f"- Column `{col}` ({missing_info_df.loc[col, 'Missing Percentage (%)']:.2f}% missing): Consider imputing with mode or 'Unknown' category, or dropping if too many missing.")
                # Store preliminary instruction for later user decision in EDA phase
                self.transformation_instructions[f"Impute_{col}"] = "Decision pending based on EDA."
        else:
            self._log_report("Missing Values Analysis", "No missing values found in the dataset.")

        # 3. Data Type Correction (Basic)
        self.report_sections.append("\n## Data Type Review\n")
        for col in self.df.columns:
            original_dtype = self.df[col].dtype
            if pd.api.types.is_string_dtype(original_dtype):
                try:
                    # Attempt to convert to numeric if all values are numeric
                    temp_col = pd.to_numeric(self.df[col], errors='coerce')
                    if not temp_col.isnull().any(): # Check if conversion introduced NaNs
                        self.df[col] = temp_col # Only assign if conversion was successful for all values
                        self.report_sections.append(f"- Column `{col}`: Converted from object/string to numeric (float).")
                        self.transformation_instructions[f"Convert_to_numeric_{col}"] = "Applied."
                    else:
                        self.report_sections.append(f"- Column `{col}`: Identified as string, contains non-numeric values. Remains as object/string.")
                except Exception:
                    self.report_sections.append(f"- Column `{col}`: Identified as string, cannot be converted to numeric. Remains as object/string.")
            elif pd.api.types.is_datetime64_any_dtype(original_dtype):
                self.report_sections.append(f"- Column `{col}`: Identified as datetime.")
            else:
                self.report_sections.append(f"- Column `{col}`: Identified as {original_dtype}.")
        self.report_sections.append("\nDataFrame dtypes after basic type correction:\n")
        self.report_sections.append(self.df.dtypes.to_markdown())

        # 4. Outlier Identification (for Numerical - Not removal)
        self.report_sections.append("\n## Initial Outlier Scan (Numerical Features)\n")
        numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        outlier_summary = []
        for col in numerical_cols:
            if self.df[col].count() > 0: # Ensure column is not entirely NaN after some cleaning
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                if not outliers.empty:
                    outlier_summary.append(f"- Column `{col}`: {len(outliers)} potential outliers detected (using IQR method). Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
                    self.transformation_instructions[f"Outlier_handling_{col}"] = "Decision pending based on EDA."
                else:
                    outlier_summary.append(f"- Column `{col}`: No significant outliers detected (using IQR method).")
            else:
                outlier_summary.append(f"- Column `{col}`: No non-missing numerical data to analyze for outliers.")
        self.report_sections.append("\n".join(outlier_summary))

        # 5. Structural Errors (Categorical - Not correction)
        self.report_sections.append("\n## Structural Error Scan (Categorical Features)\n")
        categorical_cols = self.df.select_dtypes(include='object').columns.tolist()
        structural_summary = []
        for col in categorical_cols:
            unique_values = self.df[col].unique()
            # Filter out NaN for unique value count if present
            unique_values_filtered = [uv for uv in unique_values if pd.notna(uv)]

            if len(unique_values_filtered) > 20: # High cardinality
                structural_summary.append(f"- Column `{col}`: High cardinality detected ({len(unique_values_filtered)} unique values). May require special encoding.")
                self.transformation_instructions[f"High_cardinality_encoding_{col}"] = "Decision pending based on EDA."
            elif len(unique_values_filtered) > 0:
                # Show top 5 unique values to give an idea of potential inconsistencies
                top_values = ', '.join(map(str, unique_values_filtered[:5]))
                structural_summary.append(f"- Column `{col}`: {len(unique_values_filtered)} unique values. Check for inconsistencies like varying cases or typos (e.g., '{top_values}', ...).")
                self.transformation_instructions[f"Categorical_normalization_{col}"] = "Decision pending based on EDA."
            else:
                structural_summary.append(f"- Column `{col}`: No non-missing categorical values or empty.")
        self.report_sections.append("\n".join(structural_summary))

        return self.df, self.report_sections, self.transformation_instructions


# --- Crew Member 2: EDA & Reporting Agent (The "Insight Generator") ---
class EDAReportingAgent:
    def __init__(self, df, label_column, problem_type, output_dir):
        self.df = df
        self.label_column = label_column
        self.problem_type = problem_type
        self.output_dir = output_dir
        self.report_sections = []
        self.transformation_suggestions = {} # For final executor agent

    def _log_report(self, title, content):
        self.report_sections.append(f"\n## {title}\n")
        self.report_sections.append(content)

    def _save_plot(self, fig, filename):
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, bbox_inches='tight')
        plt.close(fig)
        return os.path.basename(filepath) # Return just the filename for report

    def run_eda_and_reporting(self):
        self.report_sections.append("\n# Comprehensive Exploratory Data Analysis (EDA) Report\n")

        # Check if label_column exists after initial cleaning/potential drops
        if self.label_column not in self.df.columns:
            self._log_report("Error: Target column Missing", f"The target column '{self.label_column}' was not found in the DataFrame after preliminary cleaning. Cannot proceed with EDA.")
            return self.report_sections, self.transformation_suggestions

        # Separate features and target
        features_df = self.df.drop(columns=[self.label_column])
        target_series = self.df[self.label_column]

        numerical_features = features_df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = features_df.select_dtypes(include=['object', 'category']).columns.tolist()

        # --- Univariate Analysis ---
        self._log_report("Univariate Analysis", "Exploring individual feature distributions.")

        # Target Variable Analysis
        self.report_sections.append("\n### Target Variable Analysis\n")
        if self.problem_type == 'classification':
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x=target_series, ax=ax, palette='viridis')
            ax.set_title(f'Distribution of Target Variable: {self.label_column}')
            ax.set_xlabel(self.label_column)
            ax.set_ylabel('Count')
            plot_name = self._save_plot(fig, f'{self.label_column}_target_distribution.png')
            self.report_sections.append(f"![{self.label_column} Distribution]({plot_name})\n")
            target_counts = target_series.value_counts()
            self.report_sections.append(target_counts.to_markdown(numalign="left", stralign="left"))
            imbalance_ratio = target_counts.min() / target_counts.max() if target_counts.max() > 0 else 0
            if imbalance_ratio < 0.2: # Arbitrary threshold for high imbalance
                self.report_sections.append(f"\n**Warning:** Significant class imbalance detected (Min/Max Ratio: {imbalance_ratio:.2f}). Consider oversampling, undersampling, or cost-sensitive learning during modeling.")
                self.transformation_suggestions['Class_Imbalance_Handling'] = 'Consider oversampling (e.g., SMOTE) or undersampling techniques.'
        else: # Regression
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.histplot(target_series, kde=True, ax=axes[0])
            axes[0].set_title(f'Histogram of Target Variable: {self.label_column}')
            sns.boxplot(y=target_series, ax=axes[1])
            axes[1].set_title(f'Box Plot of Target Variable: {self.label_column}')
            plot_name = self._save_plot(fig, f'{self.label_column}_target_distribution.png')
            self.report_sections.append(f"![{self.label_column} Distribution]({plot_name})\n")
            self.report_sections.append(f"Mean: {target_series.mean():.2f}, Median: {target_series.median():.2f}, Std Dev: {target_series.std():.2f}\n")
            # Ensure no NaNs when calculating skew/kurtosis
            if target_series.dropna().empty:
                 self.report_sections.append("Skewness: N/A, Kurtosis: N/A (Target series is empty after dropping NaNs)\n")
            else:
                self.report_sections.append(f"Skewness: {skew(target_series.dropna()):.2f}, Kurtosis: {kurtosis(target_series.dropna()):.2f}\n")
                if abs(skew(target_series.dropna())) > 0.5: # Suggest transformation if skewed
                    self.report_sections.append(f"**Recommendation:** Target variable `{self.label_column}` is skewed. Consider log or power transformation if residuals are not normally distributed after initial modeling.")
                    self.transformation_suggestions['Target_Variable_Transformation'] = 'Consider log or power transformation for skewness.'


        self.report_sections.append("\n### Numerical Features Analysis\n")
        for col in numerical_features:
            self.report_sections.append(f"\n#### Feature: `{col}`\n")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            sns.histplot(self.df[col], kde=True, ax=axes[0])
            axes[0].set_title(f'Histogram of {col}')
            sns.boxplot(y=self.df[col], ax=axes[1])
            axes[1].set_title(f'Box Plot of {col}')
            plot_name = self._save_plot(fig, f'{col}_distribution.png')
            self.report_sections.append(f"![{col} Distribution]({plot_name})\n")
            desc_stats = self.df[col].describe().to_markdown()
            self.report_sections.append(desc_stats)
            if self.df[col].dropna().empty:
                self.report_sections.append("Skewness: N/A, Kurtosis: N/A (Feature series is empty after dropping NaNs)\n")
            else:
                self.report_sections.append(f"\nSkewness: {skew(self.df[col].dropna()):.2f}, Kurtosis: {kurtosis(self.df[col].dropna()):.2f}\n")
                if abs(skew(self.df[col].dropna())) > 1.0: # High skewness
                     self.transformation_suggestions[f'Skewness_Transformation_{col}'] = 'Consider log/power transformation for this feature.'


        self.report_sections.append("\n### Categorical Features Analysis\n")
        for col in categorical_features:
            self.report_sections.append(f"\n#### Feature: `{col}`\n")
            fig, ax = plt.subplots(figsize=(10, 6))
            # Ensure plotting non-NaN values for countplot
            sns.countplot(y=self.df[col].dropna(), ax=ax, order=self.df[col].value_counts().index, palette='viridis')
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel('Count')
            ax.set_ylabel(col)
            plot_name = self._save_plot(fig, f'{col}_distribution.png')
            self.report_sections.append(f"![{col} Distribution]({plot_name})\n")
            self.report_sections.append(self.df[col].value_counts(dropna=False).to_markdown(numalign="left", stralign="left")) # include NaN counts here
            if len(self.df[col].unique()) > 10: # High cardinality
                self.report_sections.append(f"\n**Recommendation:** High cardinality for `{col}`. Consider one-hot encoding if less than 20 unique values, or target encoding/feature hashing if more.")
                self.transformation_suggestions[f'Categorical_Encoding_{col}'] = 'Needs encoding (One-Hot or Target/Binary).'
            else:
                 self.transformation_suggestions[f'Categorical_Encoding_{col}'] = 'One-Hot or Label encoding.'

        # --- Bivariate Analysis ---
        self._log_report("Bivariate Analysis", "Exploring relationships between features and with the target.")

        # Numerical Features vs. Target
        self.report_sections.append("\n### Numerical Features vs. Target Variable\n")
        for col in numerical_features:
            fig, ax = plt.subplots(figsize=(8, 6))
            if self.problem_type == 'classification':
                sns.boxplot(x=self.label_column, y=col, data=self.df, ax=ax, palette='coolwarm')
            else: # Regression
                sns.scatterplot(x=self.df[col], y=self.df[self.label_column], ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel(self.label_column)
            ax.set_title(f'{col} vs. {self.label_column}')
            plot_name = self._save_plot(fig, f'{col}_vs_{self.label_column}.png')
            self.report_sections.append(f"![{col} vs. {self.label_column}]({plot_name})\n")

        # Categorical Features vs. Target
        self.report_sections.append("\n### Categorical Features vs. Target Variable\n")
        for col in categorical_features:
            fig, ax = plt.subplots(figsize=(10, 6))
            if self.problem_type == 'classification':
                # Ensure data for plotting is not empty
                if not self.df[[col, self.label_column]].dropna().empty:
                    df_temp = self.df.groupby([col, self.label_column]).size().unstack(fill_value=0)
                    df_temp.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
                    ax.set_ylabel('Count')
                else:
                    ax.text(0.5, 0.5, 'No data for this plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            else: # Regression - use violin plot for distribution or bar for mean
                if not self.df[[col, self.label_column]].dropna().empty:
                    sns.violinplot(x=col, y=self.label_column, data=self.df, ax=ax, palette='pastel')
                    ax.set_ylabel(self.label_column)
                else:
                    ax.text(0.5, 0.5, 'No data for this plot', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

            ax.set_title(f'{col} vs. {self.label_column}')
            ax.set_xlabel(col)
            ax.tick_params(axis='x', rotation=45)
            plot_name = self._save_plot(fig, f'{col}_vs_target.png')
            self.report_sections.append(f"![{col} vs. {self.label_column}]({plot_name})\n")

        # --- Correlation Matrix (for Numerical Features) ---
        if numerical_features:
            self._log_report("Correlation Matrix (Numerical Features)", "")
            # Ensure all columns exist and are numeric before correlation
            cols_for_corr = [c for c in numerical_features if c in self.df.columns]
            if self.label_column in self.df.columns and pd.api.types.is_numeric_dtype(self.df[self.label_column]):
                cols_for_corr.append(self.label_column)

            if cols_for_corr:
                corr_matrix = self.df[cols_for_corr].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title('Correlation Matrix of Numerical Features')
                plot_name = self._save_plot(fig, 'correlation_matrix.png')
                self.report_sections.append(f"![Correlation Matrix]({plot_name})\n")
                # Suggest scaling if not already suggested and problem type is regression
                if self.problem_type == 'regression' and not 'Feature_Scaling' in self.transformation_suggestions:
                     self.transformation_suggestions['Feature_Scaling'] = 'Consider Standard Scaling for numerical features, especially if coefficients/distances are important.'
            else:
                self.report_sections.append("No numerical features available for correlation matrix after processing.\n")

        return self.report_sections, self.transformation_suggestions


# --- Crew Member 3: Data Transformation Executor Agent (The "Implementer") ---
class DataTransformationExecutor:
    def __init__(self, df, output_dir, label_column, problem_type):
        self.df = df
        self.output_dir = output_dir
        self.label_column = label_column
        self.problem_type = problem_type
        self.execution_summary = []

    def _log_execution(self, message):
        self.execution_summary.append(message)
        print(message)

    def run_transformations(self, instructions):
        self._log_execution("Starting data transformation execution based on recommendations.")
        original_shape = self.df.shape
        self._log_execution(f"Original DataFrame shape: {original_shape}")

        # Store the target column temporarily and remove it from features if it's there
        temp_target = None
        if self.label_column in self.df.columns:
            temp_target = self.df[self.label_column].copy()
            self.df = self.df.drop(columns=[self.label_column])
        else:
            self._log_execution(f"Warning: Target column '{self.label_column}' not found in DataFrame for transformation. Skipping target-specific transformations.")


        # Re-identify numerical and categorical columns *after* any potential drops/conversions from previous steps
        numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()

        # 1. Imputation
        for key, value in instructions.items():
            if key.startswith("Impute_"):
                col = key.replace("Impute_", "")
                if col in self.df.columns: # Check if column still exists
                    if self.df[col].isnull().sum() > 0:
                        imputer_method = input(f"How to impute missing values in `{col}`? (mean/median/mode/drop_column/drop_row): ").lower()
                        if imputer_method == 'mean' and col in numerical_cols:
                            self.df[col].fillna(self.df[col].mean(), inplace=True)
                            self._log_execution(f"Imputed `{col}` with mean.")
                        elif imputer_method == 'median' and col in numerical_cols:
                            self.df[col].fillna(self.df[col].median(), inplace=True)
                            self._log_execution(f"Imputed `{col}` with median.")
                        elif imputer_method == 'mode':
                            self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                            self._log_execution(f"Imputed `{col}` with mode.")
                        elif imputer_method == 'drop_column':
                            self.df.drop(columns=[col], inplace=True)
                            self._log_execution(f"Dropped column `{col}` due to missing values.")
                            # Update numerical/categorical lists
                            if col in numerical_cols: numerical_cols.remove(col)
                            if col in categorical_cols: categorical_cols.remove(col)
                        elif imputer_method == 'drop_row':
                            initial_rows = len(self.df)
                            self.df.dropna(subset=[col], inplace=True)
                            self._log_execution(f"Dropped rows with missing values in `{col}`. {initial_rows - len(self.df)} rows removed.")
                        else:
                            self._log_execution(f"No valid imputation method chosen for `{col}`. Skipping imputation.")
                    else:
                        self._log_execution(f"No missing values in `{col}` to impute.")
                else:
                    self._log_execution(f"Column `{col}` not found for imputation (might have been dropped earlier).")

        # 2. Outlier Handling / Skewness Transformation
        # Re-identify numerical_cols in case columns were dropped during imputation
        numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        for key, value in instructions.items():
            if key.startswith("Outlier_handling_") or key.startswith("Skewness_Transformation_"):
                col = key.replace("Outlier_handling_", "").replace("Skewness_Transformation_", "")
                if col in numerical_cols: # Check if column still exists and is numerical
                    action = input(f"For `{col}` (skewness/outliers), apply 'log_transform', 'capping_iqr', 'remove_outliers', or 'skip'? ").lower()
                    if action == 'log_transform':
                        # Ensure column is not empty after dropping NaNs and handles 0/negative
                        if (self.df[col].dropna() <= 0).any():
                            self._log_execution(f"Warning: `{col}` contains non-positive values. Log transform cannot be applied directly. Skipping.")
                        else:
                            self.df[col] = np.log1p(self.df[col]) # log1p for values that might be 0
                            self._log_execution(f"Applied log1p transformation to `{col}`.")
                    elif action == 'capping_iqr':
                        if self.df[col].count() > 0: # Ensure data exists for quantile calculation
                            Q1 = self.df[col].quantile(0.25)
                            Q3 = self.df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
                            self._log_execution(f"Applied IQR capping to `{col}`.")
                        else:
                             self._log_execution(f"Skipping capping for `{col}`: No non-missing data.")
                    elif action == 'remove_outliers':
                        if self.df[col].count() > 0: # Ensure data exists for quantile calculation
                            initial_rows = len(self.df)
                            Q1 = self.df[col].quantile(0.25)
                            Q3 = self.df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                            self._log_execution(f"Removed outliers from `{col}`. {initial_rows - len(self.df)} rows removed.")
                        else:
                             self._log_execution(f"Skipping outlier removal for `{col}`: No non-missing data.")
                    else:
                        self._log_execution(f"Skipping outlier/skewness handling for `{col}`.")
                else:
                    self._log_execution(f"Column `{col}` not found or not numerical for outlier/skewness handling.")

        # Re-align index after potential row removals from outlier handling
        self.df.reset_index(drop=True, inplace=True)
        if temp_target is not None:
            temp_target = temp_target.reindex(self.df.index) # Align target to new index

        # 3. Categorical Encoding
        # Re-identify categorical_cols in case columns were dropped or converted
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols: # Iterate through current categorical columns
            if f'Categorical_Encoding_{col}' in instructions: # Check if EDA suggested encoding for this col
                encoding_method = input(f"How to encode categorical feature `{col}`? (one_hot/label/skip): ").lower()
                if encoding_method == 'one_hot':
                    self.df = pd.get_dummies(self.df, columns=[col], prefix=col)
                    self._log_execution(f"Applied One-Hot Encoding to `{col}`.")
                    # No need to remove from categorical_cols list here, as it's processed.
                elif encoding_method == 'label': # Removed cardinality check here, user decision
                    le = LabelEncoder()
                    # Fit transform only on non-NaN values, then re-insert
                    non_null_values = self.df[col].dropna()
                    if not non_null_values.empty:
                        self.df.loc[non_null_values.index, col] = le.fit_transform(non_null_values.astype(str))
                        self._log_execution(f"Applied Label Encoding to `{col}`.")
                    else:
                        self._log_execution(f"Skipping Label Encoding for `{col}`: No non-missing values.")
                else:
                    self._log_execution(f"Skipping encoding for `{col}`.")
            elif col in self.df.columns: # If not specifically in instructions but still categorical
                self._log_execution(f"Column `{col}` is still categorical and no specific encoding instruction was given or chosen. It remains as is.")


        # 4. Feature Scaling
        numerical_cols_for_scaling = self.df.select_dtypes(include=np.number).columns.tolist() # Get numerical cols again
        if 'Feature_Scaling' in instructions and numerical_cols_for_scaling:
            scaling_method = input("Apply feature scaling to numerical features? (standard/minmax/skip): ").lower()
            if scaling_method == 'standard':
                scaler = StandardScaler()
                # Only scale columns that have non-NaN values
                cols_to_scale = [c for c in numerical_cols_for_scaling if not self.df[c].isnull().all()]
                if cols_to_scale:
                    self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
                    self._log_execution("Applied Standard Scaling to selected numerical features.")
                else:
                    self._log_execution("No numerical columns with non-missing values to scale.")
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
                cols_to_scale = [c for c in numerical_cols_for_scaling if not self.df[c].isnull().all()]
                if cols_to_scale:
                    self.df[cols_to_scale] = scaler.fit_transform(self.df[cols_to_scale])
                    self._log_execution("Applied Min-Max Scaling to selected numerical features.")
                else:
                    self._log_execution("No numerical columns with non-missing values to scale.")
            else:
                self._log_execution("Skipping feature scaling.")
        elif 'Feature_Scaling' in instructions and not numerical_cols_for_scaling:
             self._log_execution("No numerical features found for scaling.")


        # 5. Target Variable Transformation (Regression only)
        if self.problem_type == 'regression' and 'Target_Variable_Transformation' in instructions and temp_target is not None:
            transform_target = input(f"Apply log transformation to target `{self.label_column}`? (yes/no): ").lower()
            if transform_target == 'yes':
                if (temp_target.dropna() <= 0).any():
                    self._log_execution(f"Warning: Target column `{self.label_column}` contains non-positive values. Log transform cannot be applied directly. Skipping.")
                else:
                    temp_target = np.log1p(temp_target)
                    self._log_execution(f"Applied log1p transformation to target `{self.label_column}`.")
            else:
                self._log_execution(f"Skipping transformation for target `{self.label_column}`.")
        elif self.problem_type == 'regression' and temp_target is None:
            self._log_execution(f"Skipping target variable transformation as `{self.label_column}` was not found.")


        # 6. Class Imbalance Handling (Classification only) - placeholder for now
        if self.problem_type == 'classification' and 'Class_Imbalance_Handling' in instructions:
            self._log_execution("Class imbalance detected. Manual handling (e.g., SMOTE) is often performed outside this core pipeline after final feature selection, or during model training.")


        # Re-add the label column if it was successfully extracted
        if temp_target is not None:
            self.df[self.label_column] = temp_target.reset_index(drop=True)
        else:
            self._log_execution(f"Could not re-attach target column '{self.label_column}' as it was not found initially.")


        self._log_execution(f"Final DataFrame shape after transformations: {self.df.shape}")
        self._log_execution(f"Number of rows removed during transformations: {original_shape[0] - self.df.shape[0]}")

        return self.df, "\n".join(self.execution_summary)


# --- New Class for LLM-powered Agents ---
class LLMAgenticAnalysis:
    def __init__(self, report_filepath, output_dir, original_df_shape, label_column, problem_type):
        self.report_filepath = report_filepath
        self.output_dir = output_dir
        self.original_df_shape = original_df_shape
        self.label_column = label_column
        self.problem_type = problem_type

        # IMPORTANT: Ensure 'llm' is correctly configured globally or passed here.
        self.llm = llm # Using the globally defined LLM instance

        # Tools for agents
        self.file_reader = FileReadTool(file_path=self.report_filepath)

    def run_agentic_analysis(self):
        # Define Agents
        report_analyst = Agent(
            role='Data Analysis Report Analyst',
            goal='Read and thoroughly understand the comprehensive data analysis report, extracting key findings, potential issues, and initial recommendations.',
            backstory=(
                "You are an expert data scientist with a keen eye for detail. "
                "You excel at synthesizing complex data analysis reports into concise, actionable insights. "
                "Your goal is to prepare a clear summary for a non-technical stakeholder."
            ),
            verbose=True,
            allow_delegation=False,
            tools=[self.file_reader],
            llm=self.llm
        )

        action_suggester = Agent(
            role='User Interaction and Action Suggester',
            goal='Based on the analysis report summary, interact with the user to propose next steps and gather their preferences for further data transformation or exploration.',
            backstory=(
                "You are a friendly and intuitive AI assistant specializing in guiding users through data analysis workflows. "
                "You can translate technical findings into plain language and offer clear choices for action. "
                "Your aim is to empower the user to make informed decisions about their data."
            ),
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # Define Tasks
        analyze_report_task = Task(
            description=f"""
            Read and analyze the comprehensive data analysis report located at '{self.report_filepath}'.
            Extract the following:
            - Key findings from initial data overview (shape, data types, unique values).
            - Summary of missing values and recommended handling.
            - Summary of duplicate rows.
            - Insights from numerical and categorical feature distributions (skewness, outliers, cardinality).
            - Observations from bivariate analysis and correlations with the target.
            - Any explicit transformation recommendations made by the previous agents in the report (e.g., for imputation, encoding, scaling, class imbalance).

            Format the output as a clear, concise summary suitable for presenting to the user.
            """,
            expected_output="A markdown summary of the data analysis report, including key findings, issues, and initial recommendations.",
            agent=report_analyst
        )

        propose_actions_task = Task(
            description=f"""
            Based on the 'Comprehensive Data Analysis Report' summary provided by the Data Analysis Report Analyst,
            and considering the original DataFrame had {self.original_df_shape[0]} rows and {self.original_df_shape[1]} columns,
            the problem type is '{self.problem_type}' and the label column is '{self.label_column}'.

            1. Present the key findings and recommendations from the report summary to the user in a friendly, easy-to-understand manner.
            2. Propose concrete next actions the user can take, specifically focusing on applying the recommended data transformations.
            3. Frame your suggestion as a clear question to the user. Example: "Based on this analysis, would you like to proceed with applying the recommended transformations to your data (e.g., handling missing values, encoding categories, scaling features)? Please type 'apply transformations' or 'skip transformations'."
            4. Do not offer other choices like exploring specific features or generating more visualizations in this single prompt, as the next step is specifically about executing the transformations.
            """,
            expected_output="A conversational prompt for the user, summarizing findings and asking for their preferred next steps regarding transformations.",
            agent=action_suggester,
            context=[analyze_report_task] # Pass the output of the first task as context
        )

        # Create Crew
        crew = Crew(
            agents=[report_analyst, action_suggester],
            tasks=[analyze_report_task, propose_actions_task],
            verbose=True, # Set to 1 for simpler output, 2 for more detailed agent thought process
            process=Process.sequential # Agents execute tasks sequentially
        )

        self._log("Starting LLM-powered agentic analysis...")
        result = crew.kickoff()
        self._log("LLM-powered agentic analysis finished.")
        return result

    def _log(self, message):
        # Re-use CrewLeader's logging for consistency
        with open(os.path.join(self.output_dir, "analysis_log.txt"), 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        print(message)


# --- Main Execution ---
if __name__ == "__main__":
    crew_leader = CrewLeader()
    crew_leader.get_user_inputs()
    crew_leader.orchestrate_analysis()
    print(f"\nAll operations complete. Check the '{crew_leader.output_dir}' directory for reports and plots.")
    if crew_leader.cleaned_data_filepath:
        print(f"Cleaned data saved to: {crew_leader.cleaned_data_filepath}")
