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