# Insurance Risk Analytics & Predictive Modeling
## 10 Academy KAIM Week 3 Project

**Objective**: Analyze historical car insurance data (Feb 2014 - Aug 2015) to optimize marketing strategies and identify low-risk segments for AlphaCare Insurance Solutions (ACIS).

### Task 1: EDA and Git Setup
- **Git**: Repository setup with `task-1` branch and CI/CD via GitHub Actions.
- **EDA**: Conducted univariate and bivariate analyses to uncover insights.
  - **Univariate Findings**:
    - `TotalPremium` and `TotalClaims`: Right-skewed distributions with outliers indicating high-value policies/claims.
    - `Province`: Higher policy counts in urban areas (e.g., Gauteng).
  - **Bivariate/Geographic Findings**:
    - Weak correlation between `TotalPremium` and `TotalClaims`.
    - Higher `LossRatio` in urban provinces.
    - Premiums vary by `CoverType` and `Province`.
  - Visualizations: See `visualizations/` folder (e.g., `histogram_TotalPremium.png`, `loss_ratio_province.png`).

### Setup Instructions
1. Clone repository: `git clone https://github.com/Naod-Mergiya/insurance-risk-analytics.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run EDA: `python run_eda.py`

### Key Visualizations
- [Loss Ratio by Province](visualizations/loss_ratio_province.png)
- [Premium vs Claims by PostalCode](visualizations/scatter_premium_claims_postalcode.png)
- [Premium by CoverType and Province](visualizations/premium_province_covertype.png)

## Task 2: Data Versioning and Exploratory Data Analysis (EDA)

### Objective
Implement data versioning using DVC and perform initial EDA on the `MachineLearningRating_v3.txt` dataset.

### Steps Completed
- **Set up DVC**: Initialized DVC and configured a local remote storage at `E:\data_storage`.
- **Tracked Data**: Added `MachineLearningRating_v3.txt` to DVC tracking.
- **Data Loading**: Successfully loaded the dataset in `EDA.ipynb` using `dvc.api.open` and the `load_data` function from `src/data_loader.py`.
- **EDA Integration**: Updated `run_eda.py` to use DVC for data loading and prepared for univariate and bivariate analysis.
- **Version Control**: Committed changes to Git and pushed to the `task-2` branch.
- **DVC Push**: Pushed data to the local DVC remote.

### Files Updated
- `EDA.ipynb`: Added data loading and initial EDA setup.
- `run_eda.py`: Integrated DVC and data loading logic.
- `src/data_loader.py`: Defined the `load_data` function.
- `.dvcignore`, `.dvc/config`: Configured DVC settings.
- `.gitignore`: Updated to ignore data files and caches.

### Next Steps
- Complete univariate and bivariate analysis in `run_eda.py`.
- Generate visualizations and document findings.
- Prepare for Task 3 (e.g., predictive modeling).

### Notes
- Dataset contains 1,000,098 rows with columns (32, 37) having mixed types (resolved with `low_memory=False`).
- Current working directory and Python path adjusted using `sys.path.append`.