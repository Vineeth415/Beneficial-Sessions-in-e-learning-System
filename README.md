# E-Learning Session Analysis System

A comprehensive machine learning system for identifying beneficial sessions in e-learning platforms. This project analyzes user session data to determine which learning sessions are most effective and provides insights for improving educational outcomes.

## 🚀 Features

- **Data Processing**: Robust data loading, validation, and preprocessing
- **Machine Learning**: Multiple ML models (Random Forest, Gradient Boosting, Logistic Regression)
- **Data Cleaning**: Automatic duplicate removal, outlier detection, and missing value handling
- **Visualization**: Comprehensive plotting and dashboard creation
- **Testing**: Unit tests and validation frameworks
- **Security**: Input validation and secure file operations

## 📁 Project Structure

```
├── src/
│   ├── data_processing/
│   │   └── data_loader.py          # Data loading and preprocessing
│   ├── models/
│   │   └── ml_model.py             # Machine learning models
│   └── utils/
│       ├── visualization.py        # Plotting and visualization
│       ├── data_generator.py       # Synthetic data generation
│       └── data_cleaner.py         # Data cleaning utilities
├── tests/
│   └── test_data_loader.py         # Unit tests
├── data/                           # Data storage
├── models/                         # Saved models
├── outputs/                        # Generated reports and plots
├── main.py                         # Main application
├── requirements.txt                # Dependencies
└── BUG_FIXES_AND_IMPROVEMENTS.md   # Bug fixes documentation
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Identifying-Beneficial-Sessions-in-an-e-learning-System-Using-Machine-Learning
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Quick Start
Run the complete analysis pipeline:
```bash
python main.py
```

This will:
- Generate synthetic e-learning session data
- Clean and preprocess the data
- Train a machine learning model
- Create visualizations and reports
- Save the trained model

### Individual Components

**Data Loading**:
```python
from src.data_processing.data_loader import SessionDataLoader

loader = SessionDataLoader("data/sessions.csv")
data = loader.load_data()
processed_data = loader.preprocess_data()
```

**Model Training**:
```python
from src.models.ml_model import SessionClassifier

classifier = SessionClassifier(model_type='random_forest')
results = classifier.train(features, labels)
```

**Data Cleaning**:
```python
from src.utils.data_cleaner import SessionDataCleaner

cleaner = SessionDataCleaner()
cleaned_data, report = cleaner.clean_session_data(data)
```

## 📊 Outputs

The system generates:
- **Visualizations**: Session distributions, correlation matrices, feature importance
- **Model Performance**: Accuracy, ROC AUC, confusion matrices
- **Analysis Report**: Comprehensive findings and recommendations
- **Trained Model**: Saved model for future predictions

## 🧪 Testing

Run unit tests:
```bash
python -m pytest tests/
```

## 🐛 Bug Fixes

This project includes comprehensive bug fixes and improvements:
- **DateTime conversion errors** - Fixed timestamp handling
- **Logic errors** - Corrected test data processing
- **Resource leaks** - Added proper file cleanup
- **Security vulnerabilities** - Implemented input validation
- **Performance issues** - Added memory-efficient processing

See `BUG_FIXES_AND_IMPROVEMENTS.md` for detailed documentation.

## 📈 Results

The system successfully:
- Processes 2,000+ session records
- Achieves high model accuracy
- Identifies beneficial learning patterns
- Provides actionable insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Dependencies

- **pandas** >= 1.5.0 - Data manipulation
- **numpy** >= 1.21.0 - Numerical computing
- **scikit-learn** >= 1.1.0 - Machine learning
- **matplotlib** >= 3.5.0 - Visualization
- **seaborn** >= 0.11.0 - Statistical visualization
- **joblib** >= 1.1.0 - Model persistence

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team.
