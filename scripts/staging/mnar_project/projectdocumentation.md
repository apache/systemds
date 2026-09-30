# Project Documentation
# Missing Value Imputation for MNAR

## 0. How to Run the Experiments

The full experiment pipeline can be executed from the `mnar_project` directory via:

```bash
python3 src/run_all_experiments.py --input-path data/processed/diabetes_sample.csv
```

For the AQ dataset, use:

```bash
python3 src/run_all_experiments.py --input-path data/processed/AQ_clean.csv
```

The script generates hybrid missingness with `10%` and `5%`, runs the baselines, the final Python prototype, and the DML/SystemDS prototype, and writes the resulting comparison tables to `results/`. It requires Python 3, the Python packages used in the project, Java 17, and the included `systemds` directory.

## 1. Project Goal

The aim of the project is to experimentally develop an imputation method for MNAR (Missing Not At Random) data. Entries classified as MNAR are missing due to a systematic relationship with the value itself or with other associated characteristics. This makes imputation particularly challenging, as standard methods - such as mean imputation, kNN, or standard MICE - rely solely on observed values; if these values ​​are already biased, the methods learn from an incomplete and biased distribution. This is precisely where the project's approach comes in.

## 2. The initial Prototype

The first prototype operated under highly controlled conditions in a special case based on a 10,000-row subset of the CDC Diabetes Health Indicators dataset:

- One missing column: `BMI`
- Missing value rate: `10%`
- Missingness Mechanism:
  - a random component
  - an MNAR-like component that tends to remove high BMI values

This setup was not yet particularly realistic, but it was very well suited to highlighting the underlying problem:

- High BMI values are systematically missing more often
- Standard methods typically impute these high values too low

### 2.1. Initial Prototype Results

The following methods were tested on this setup:

**Baselines**

- Mean
- Median
- Mode
- kNN
- Standard MICE

**Rebalanced Variants**

- First rebalanced MICE variants

**Important finding:**

- For random/MCAR-like cases, standard MICE and kNN performed best
- For MNAR cases, standard methods exhibited strong negative bias
- The rebalanced variants significantly reduced this bias

Example from the old BMI-only setup:

- `standard MICE`, bias on `random`: `-1.0737`
- `standard MICE`, bias on `mnar`: `-9.5859`
- `rebalancing_mice_2`, bias on `random`: `+1.6511`
- `rebalancing_mice_2`, bias on `mnar`: `-5.9800`

This illustrates the fundamental trade-off:

- Standard methods work well for simpler missing data situations
- The rebalanced approach, on the other hand, is more robust against MNAR bias

## 3. Improvement of the Missingness Generator

Following the initial prototype, the missingness mechanism was significantly improved.

### 3.1 Motivation

What was good about the first BMI-only version was that we had mechanisms that operated strictly according to MCAR-like and MNAR-like classifications. This allowed us to trace the analyses based on the exact type of missing data. This was a very good **proof of concept**, but too specific. 

Therefore, a new **hybrid probabilistic missingness generator** was developed.

### 3.2 Basic Concept of the New Generator

For multiple target columns, missingness is no longer generated strictly by rules, but probabilistically.

A score is calculated for each target column \(X_j\):

```text
score_j =
    alpha_j
  + beta_mnar * self_signal_j
  + beta_mar  * observed_signal_j
  + noise
```

The score is then converted into a probability using a sigmoid function:

```text
p_j = sigmoid(score_j)
```

and finally, missing values are generated according to:

```text
M_j ~ Bernoulli(p_j)
```

### 3.3 Interpretation of the Components

- **MNAR component**
  depends on the value of the target column itself
- **MAR component**
  depends on other observed variables
- **Noise component**
  makes the mechanism less rigid and somewhat more realistic

**Important to note:**

This generator no longer clearly separates MCAR, MAR, and MNAR on a row-by-row basis.
Instead, it generates ahybrid missingness probability. This is more realistic, but less analytically rigorous.

## 4. Hybrid Diabetes Dataset

### 4.1 Structure

For the diabetes dataset, a new multi-column case was generated with missing values in:

- `BMI`
- `Income`
- `MentHlth`

The target rate was again approximately `10%` per column.

The generated files are:

- `mnar_project/data/processed/diabetes_mnar.csv`
- `mnar_project/data/processed/mask_diabetes_mnar.csv`

The mask file also directly the `ground_truth` for the removed values.

### 4.2 Validation of the New Testbed

The following were examined:

1. **Distribution comparison**
   between Original, Observed, and Removed
2. **Missingness rate by target value ranges**
3. **Baseline behavior**

The dataset proved to be a good primary testbed because:

- the observed distribution was visibly skewed
- the removed values were systematically located in problematic ranges
- missingness was clearly correlated with the target value

Examples:

- `BMI`: observed mean `27.66`, removed mean `35.57`
- `Income`: observed mean `6.32`, removed mean `3.87`
- `MentHlth`: observed mean `1.87`, removed mean `15.11`

This achieved exactly what is relevant for MNAR imputation:

- the observed values are not representative
- standard methods must learn from a biased sample

## 5. Final Python Prototype

The two main final Python files are:

- `mnar_project/src/rebalanced_mice_complete.py`

And a more general numerical variant:

- `mnar_project/src/rebalanced_mice_general.py`

### 5.1 Basic Logic of the Prototype

The prototype is a multi-column rebalanced MICE.

It consists of four core ideas:

1. **Mean initialization**
   of all missing columns
2. **Validation-based configuration selection**
3. **MICE loop**
   across all incomplete target columns
4. **Rebalancing**
   in the internal training step

### 5.2 Algorithm Flow

#### Step 1: Initialization

For each target column with missing values, the missing values are first filled with the mean of the observed values.

#### Step 2: Configuration Selection via Validation

For each target column, a rebalancing configuration is selected before the actual MICE loop.

To do this:

- observed values are artificially split into train and validation sets
- values are hidden in the validation set
- different rebalancing configurations are tested
- the configuration with the best score is selected

The score used was:

```text
validation_score = MAE + bias_weight * abs(bias)
```

#### Step 3: MICE Loop

An iterative loop then runs through all missing target columns:

- Select the current target column
- Use other columns as features
- Train the model on observed rows
- Predict missing values
- Next target column

### 5.3 Rebalancing Components

#### i) Target weights

The function `scew_row_weights(...)` identifies target ranges that are:

- rare
- tail-heavy
- or skewed

These ranges are weighted more heavily so that underrepresented values are not lost.

#### ii) Similarity weights

In addition, a missingness model is trained:

```text
other features -> probability that the current target is missing
```

The observed rows that are more similar to missing rows are given higher weights.

#### iii) Combination

The final weights are calculated as follows:

```text
final_weights = target_weights * similarity_weights
```

Followed by:

- Clipping
- Stabilization / Normalization

#### iv) Optional Oversampling

Variants using:

- simple oversampling
- SMOTE-style oversampling

were also tested.

However, validation tuning showed that **the best variant on the main diabetes dataset was simple weighting without oversampling**

### 5.4 Internal ML Step

The internal regression step in the Python prototype uses:

- `StandardScaler`
- `Ridge(alpha = 1.0)`
- `sample_weight = final_weights`

This is important because it is precisely this internal ML step that determines the actual imputation.


## 6. Results of the Final Python Prototype

### 6.1 Diabetes Hybrid Dataset

Baseline results on `diabetes_mnar.csv`:

- Standard MICE:
  - `BMI`: `mae 8.7227`, `bias -6.4750`
  - `Income`: `mae 1.9870`, `bias 1.3295`
  - `MentHlth`: `mae 13.2194`, `bias -11.4328`

Final rebalanced prototype:

- `BMI`: `mae 7.9885`, `bias -4.7630`
- `Income`: `mae 1.7343`, `bias 0.4585`
- `MentHlth`: `mae 11.8358`, `bias -7.0048`

Interpretation:

- The final prototype outperforms the standard MICE model on all three target columns
- The improvement is particularly evident in terms of bias

### 6.2 More General Numeric Version

A more general variant, `rebalanced_mice_general.py`, was created that:

- can load any CSV files
- automatically uses numeric columns
- automatically detects missing columns

This version was initially tested on the diabetes dataset and produced results nearly identical to those of the specialized `complete` version.

## 7. Second Dataset: AQ

### 7.1 Motivation

To verify whether the approach works beyond the diabetes dataset, a second numerical dataset was used:

- `AQ_clean.csv`

An analogous hybrid missingness generator was built for AQ:

- `mnar_project/src/gen_hybrid_miss_AQ.py`

Missing values were generated for:

- `CO(GT)`
- `NOx(GT)`
- `NO2(GT)`

### 7.2 Results for AQ

The general Python version `rebalanced_mice_general.py` yielded:

- `CO(GT)`: `MAE 0.4731`, `bias -0.2392`
- `NOx(GT)`: `MAE 107.7312`, `bias -68.4840`
- `NO2(GT)`: `MAE 23.9194`, `bias -4.0347`

For comparison, standard MICE:

- `CO(GT)`: `MAE 0.4704`, `bias -0.2953`
- `NOx(GT)`: `MAE 141.0032`, `bias -117.3130`
- `NO2(GT)`: `MAE 26.0761`, `bias -13.4388`

Here, too, the rebalanced mice performed moderately to significantly better.

## 8. Conversion to DML / SystemDS

### 8.1 Objective

The final imputation prototype should then be converted to DML/SystemDS.

It was important to:

- not port all of the preliminary work
- but rather the actual final imputation core

Python therefore remains responsible for:

- Missingness generation
- Experiments
- Evaluation

DML/SystemDS contains:

- the final imputation prototype

### 8.2 DML Files

Relevant files:

- `mnar_project/dml/rebalanced_mice_complete.dml`
- `mnar_project/dml/run_rebalanced_mice_complete.sh`

In addition, a small preparation script was created for DML:

- `mnar_project/src/prepare_systemds_input.py`

### 8.3 Practical Issue with CSV Import

SystemDS does not automatically interpret empty CSV fields as missing values. For this reason, special input files are generated for the DML run, in which missing values are explicitly stored as `NaN`:

- `diabetes_mnar_systemds.csv`
- `AQ_mnar_systemds.csv`

For the AQ dataset, the following additional considerations applied:

- `Date` and `Time` are not numeric
- Therefore, the SystemDS file contains only the numeric columns

## Results in DML

### 9.1 Results on Diabetes

Current best DML results on the diabetes dataset:

- `BMI`: `mae 8.7895`, `bias -4.8833`
- `Income`: `mae 2.2058`, `bias 0.8958`
- `MentHlth`: `mae 13.2205`, `bias -8.3338`

For comparison, Python:

- `BMI`: `mae 7.9885`, `bias -4.7630`
- `Income`: `mae 1.7343`, `bias 0.4585`
- `MentHlth`: `MAE 11.8358`, `bias -7.0048`

- Standard MICE:
  - `BMI`: `MAE 8.7227`, `bias -6.4750`
  - `Income`: `MAE 1.9870`, `bias 1.3295`
  - `MentHlth`: `MAE 13.2194`, `bias -11.4328`

### 9.2 Results on AQ

Current best DML results on AQ:

- `CO(GT)`: `MAE 0.6704`, `bias -0.3924`
- `NOx(GT)`: `MAE 111.5395`, `bias -74.9147`
- `NO2(GT)`: `MAE 33.4387`, `bias -16.1206`

For comparison, Python:

- `CO(GT)`: `MAE 0.4731`, `bias -0.2392`
- `NOx(GT)`: `MAE 107.7312`, `bias -68.4840`
- `NO₂(GT)`: `MAE 23.9194`, `bias -4.0347`

For comparison, Standard MICE:

- `CO(GT)`: `MAE 0.4704`, `bias -0.2953`
- `NOx(GT)`: `MAE 141.0032`, `bias -117.3130`
- `NO₂(GT)`: `MAE 26.0761`, `bias -13.4388`

### 9.4 Interpretation

The DML version is:

- functionally successfully ported
- but not numerically identical to the Python version

The findings are therfore:

- the algorithm can be implemented in DML
- the results are qualitatively similar
- but not exactly the same


## 10. Debugging the DML Deviations

Targeted debugging steps were carried out to understand why DML deviates from Python.

### 10.1 Similarity Weights

Similarity weights were compared in isolation.

Results for `BMI` on diabetes:

- Correlation of `propensity` values: approximately `0.91`
- Correlation of `similarity_weights`: approximately `0.98`

This means:

- DML and Python already weight the observed rows very similarly here
- the similarity step is not the main reason for the deviation

### 10.2 Internal regression step

Next, the internal weighted regression step was compared in isolation.

Result:

- Correlation of Python vs. DML predictions: only about `0.83`
- Average difference: about `+2.49`

This shows:

The biggest difference between Python and DML arises in the internal regression step.

The following was also attempted:

- Using an explicit Ridge solution in DML instead of `lm(...)`

This did not improve the similarity to sklearn-Ridge.

### 10.3 Preliminary Conclusion

Thus, the most plausible conclusion is:

- The main difference stems from the numerical ML behavior of SystemDS compared to sklearn
- Not primarily from the missingness generator
- Not primarily from the similarity weights


## 11. Limitations

### 11.1 General Algorithmic Limitations

- The approach is currently focused on **numeric data**
- Categorical/string columns are not yet supported
- The hybrid missingness generator no longer clearly distinguishes between MAR, MNAR, and MCAR on a per-row basis
- The approach is more of a **general numeric prototype** than a fully universal framework## # 10.3 Preliminary Conclusion

Thus, the most plausible conclusion is:

- The main difference stems from the numerical ML behavior of SystemDS compared to sklearn
- Not primarily from the missingness generator
- Not primarily from the similarity weights


## 11. Limitations

### 11.1 General Algorithmic Limitations

- The approach is currently focused on numeric data
- Categorical/string columns are not yet supported
- The hybrid missingness generator no longer clearly distinguishes between MAR, MNAR, and MCAR on a row-by-row basis
- The approach is more of a general numerical prototyp than a fully universal framework
