# Fuzzy Logic for Student Academic Success Prediction

An exploratory machine learning project investigating fuzzy rule-based classification and genetic algorithm-based rule selection for predicting student academic outcomes.

**Project status:** Archived — Work in Progress  
**Implementation:** Python, Jupyter Notebook  
**Task:** Multiclass classification  
**Dataset:** Higher Education Predictors of Student Retention

## Overview

This project explores the use of fuzzy logic to represent relationships between student characteristics and academic outcomes. The objective was to develop an interpretable, rule-based classification pipeline and investigate whether a genetic algorithm (GA) could select a smaller subset of fuzzy rules.

The project was developed as an optional academic project. A substantial part of the experimental pipeline was implemented, including exploratory data analysis, feature selection, fuzzy membership functions, fuzzy rule extraction, genetic algorithm-based rule selection, and fuzzy inference.

However, integrating these components into a consistent and reliable end-to-end classification system proved challenging. The project was therefore discontinued before achieving satisfactory predictive performance.

The repository preserves the implementation and its experimental results for reference. It should be considered an exploratory prototype rather than a completed or validated prediction system.

## Objectives

The main objectives were to:

- Explore student academic outcome prediction as a multiclass classification problem.
- Represent selected numerical features using fuzzy linguistic terms.
- Construct interpretable classification rules from training data.
- Investigate genetic algorithm-based selection of fuzzy rules.
- Implement a fuzzy inference mechanism for generating predictions.
- Evaluate the resulting classifier and identify implementation limitations.

## Dataset

The project uses the **Higher Education Predictors of Student Retention** dataset.

- Dataset source: [Kaggle — Higher Education Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
- Number of instances: 4,424
- Number of columns in the included CSV: 35
- Input features: 34
- Target variable: `Target`
- Target classes: `Dropout`, `Enrolled`, and `Graduate`

The dataset contains demographic, academic, application-related, and socioeconomic information about students.

### Target distribution

| Academic outcome | Instances | Approximate proportion |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |
| **Total** | **4,424** | **100%** |

The target distribution is imbalanced, which makes class-wise evaluation particularly important.

The dataset is provided in the `Dataset/` directory. Please consult the original dataset page for source information and applicable dataset terms.

## Methodology and Implementation

The notebook contains the following experimental stages.

### 1. Data exploration and preprocessing

The initial analysis examines the dataset structure, data types, missing values, feature distributions, and target class distribution.

The available notebook outputs indicate that no missing values were detected in the dataset used during the recorded experiment.

### 2. Training and test split

The data is divided into training and test sets using an 80:20 split, with `random_state=42` and stratification by the target variable.

The recorded split contains:

- Training set: 3,539 instances
- Test set: 885 instances

### 3. Feature selection

Filter-based feature selection is investigated using:

- Correlation analysis to identify highly correlated numerical features.
- Mutual information to examine the relationship between input features and the target.

The notebook uses a correlation threshold of 0.95 and a mutual information threshold of 0.01.

### 4. Fuzzy membership functions

Triangular membership functions are implemented to represent selected numerical features through linguistic categories, such as `Low`, `Medium`, and `High`.

Categorical features are handled through their observed discrete values.

The purpose of this representation is to support rule-based reasoning rather than relying exclusively on a conventional black-box classifier.

### 5. Fuzzy rule extraction

The notebook implements a version of the Wang–Mendel fuzzy rule extraction method.

Training instances are converted into candidate rules consisting of feature-based antecedents and an academic outcome as the consequent. Rule weights are also calculated from membership degrees.

The recorded notebook output reports **229 extracted rules**.

### 6. Genetic algorithm-based rule selection

The project uses the DEAP evolutionary computation library to investigate rule subset selection.

The implementation represents candidate rule subsets as binary individuals and defines genetic operators for selection, crossover, and mutation. The fitness function combines predictive accuracy with a penalty related to the number of selected rules.

The notebook records a selected rule set containing **56 rules**.

This is an experimental rule-selection implementation; the recorded results do not establish that the selected subset is optimal or that the resulting classifier generalizes reliably.

### 7. Fuzzy inference and classification

A fuzzy inference procedure is implemented to apply the selected rules to test instances. Rule firing strengths are calculated from the antecedent conditions and associated weights, and the resulting class strengths are used to generate predictions.

The notebook also includes debugging output intended to investigate cases in which no rule fires.

This stage exposed an important limitation in the implementation: the recorded final predictions collapse to a single class.

## Experimental Results

The following results are taken from the saved notebook outputs. They describe the recorded experiment and are not independently reproduced or validated in this repository documentation.

### Recorded test-set evaluation

| Metric | Recorded result |
|---|---:|
| Test-set size | 885 |
| Accuracy | 49.94% |
| Predicted classes | Graduate only |
| Dropout recall | 0.00 |
| Enrolled recall | 0.00 |
| Graduate recall | 1.00 |
| Macro-averaged F1-score | Approximately 0.22 |

The confusion matrix, with rows representing true classes and columns representing predicted classes in the order `Dropout`, `Enrolled`, and `Graduate`, is:

| Actual \ Predicted | Dropout | Enrolled | Graduate |
|---|---:|---:|---:|
| Dropout | 0 | 0 | 284 |
| Enrolled | 0 | 0 | 159 |
| Graduate | 0 | 0 | 442 |

### Interpretation

Although the recorded accuracy is approximately 49.94%, the model predicts `Graduate` for every test instance.

Consequently, the accuracy is approximately equal to the proportion of `Graduate` instances in the test set. The classifier fails to identify either of the other two classes.

This result indicates that the implemented pipeline did not achieve satisfactory multiclass classification performance. It must not be interpreted as evidence of a reliable student outcome prediction system.

## Repository Structure

```text
Fuzzy_Logic_For_Student_Academi_Succes_Prediction/
├── Dataset/
│   └── dataset.csv
├── Documentation/
│   └── Documentation.docx
├── Fuzzy_Logic_For_Studen_Academi_Succes_Prediction.ipynb
├── selected_fuzzy_rules.pkl
├── LICENSE
└── README.md
```

### Main files

- `Fuzzy_Logic_For_Studen_Academi_Succes_Prediction.ipynb`: Main experimental notebook, including preprocessing, feature selection, fuzzy rule extraction, genetic algorithm-based rule selection, and inference.
- `Dataset/dataset.csv`: Dataset included with the project.
- `selected_fuzzy_rules.pkl`: Serialized rule-related artifacts from an experimental run. Compatibility with a fresh execution of the notebook has not been verified.
- `Documentation/Documentation.docx`: Documentation directory placeholder included in the original repository.

## Running the Notebook

The notebook was developed in a Google Colab-style environment and includes KaggleHub-based dataset retrieval.

### Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Install the dependencies listed in the notebook.
3. Configure access to the dataset if required.
4. Review the data-loading cell before executing the notebook.
5. Run the notebook cells in order, keeping in mind that some stages may be computationally expensive.

### Option 2: Local environment

A Python environment with Jupyter Notebook or JupyterLab can be used to inspect the implementation.

The main dependencies include:

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- DEAP
- KaggleHub

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn deap kagglehub jupyter
```

**Important:** The notebook contains environment-specific dataset paths. A local execution may require adjusting the data-loading configuration to use `Dataset/dataset.csv`. End-to-end execution in a fresh environment has not been verified, and the saved pickle file should not be assumed to be compatible with a newly executed notebook.

## Limitations and Future Work

The project remains incomplete. The main limitations identified during development are:

1. **Single-class predictions:** The recorded inference stage predicts only `Graduate`, resulting in zero recall for `Dropout` and `Enrolled`.
2. **Rule coverage:** Some test instances do not activate any of the selected rules, causing the inference procedure to use a default class.
3. **Pipeline integration:** Feature selection, membership-function definitions, rule extraction, rule selection, and inference require consistent feature representations and compatible parameters.
4. **Computational cost:** Rule-based inference and repeated rule evaluation during genetic algorithm optimization can be expensive because of nested operations over samples, rules, and antecedents.
5. **Reproducibility:** The recorded outputs and serialized artifacts have not been verified through a clean, end-to-end rerun.
6. **Evaluation scope:** The recorded experiment is insufficient to establish reliable generalization or superiority over baseline classifiers.

Potential future work includes improving rule coverage, profiling computational bottlenecks, validating the inference mechanism, and comparing the method against simple baseline classifiers using class-wise precision, recall, F1-score, and confusion matrices.

These improvements are possible future directions, not completed features of the current repository.

## What This Project Demonstrates

Despite its incomplete status, the project documents practical experimentation with:

- Fuzzy logic and triangular membership functions.
- Interpretable, rule-based classification.
- Wang–Mendel-style fuzzy rule extraction.
- Genetic algorithms for rule subset selection.
- Feature selection and exploratory data analysis.
- Debugging and evaluation of a multiclass machine learning pipeline.

The repository also illustrates the difference between implementing individual machine learning components and integrating them into a reliable end-to-end system.

## Intended Use

This repository is intended for educational and experimental reference. It is not a validated academic risk assessment tool and should not be used to make decisions about individual students.

## License

The project is distributed under the MIT License, as specified in the included `LICENSE` file.

Please consult the original dataset source for the dataset's own terms of use.

## Project Status

This repository preserves an unfinished academic experiment in its existing state. The current implementation is not claimed to be production-ready, fully reproducible, or predictively reliable.

The recorded results and limitations are documented to make the scope of the work transparent to anyone exploring the code.

## Authors / Project Owners

- **Seyed Hossein Hosseini DolatAbadi**
- **Mohammad Amin Nasiri**
