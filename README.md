# Machine Learning for Music Classification

---

## Overview
A machine learning pipeline designed to classify classical music compositions by six composers using MIDI files.  
The project leverages feature extraction, preprocessing, and classification models like **SVM**, **Logistic Regression**,  
and **Gradient Boosting**, aiming for accuracy above 90%.

---

## Features
- **Feature Extraction**: Utilizes `music21` to extract musical features (e.g., notes, chords, and N-grams).  
- **Data Preprocessing**: Includes normalization, outlier removal, and handling of missing values.  
- **Machine Learning Models**: Implements SVM, Logistic Regression, and Gradient Boosting classifiers.  
- **Evaluation Metrics**: Accuracy & confusion matrix
- **Research Paper**: Results are organized and presented in a structured research paper format. (Not shown along this Github repo)

---
## Workflow
1. **MIDI File Preprocessing**  
   - Filter and converted MIDI files for readability for ML processing.  
   - Extract musical components (e.g., notes, chords) using `music21`.  

2. **Feature Engineering**  
   - Create N-grams to capture musical patterns.   

3. **Model Training and Testing**  
   - Train models using machine learning algorithms.  
   - Perform hyperparameter tuning for optimization.  

4. **Evaluation**  
   - Assess model performance using metrics like confusion matrices and classification reports. (Thus, it shows how our models are doing) 

5. **Results Documentation**  
   - Findings are presented in a structured research paper format, including Abstract, Introduction,  
     Dataset, ML Models, Feature Extraction, Results, and References.  

---
## Dataset
- **Source**: Classical music MIDI files provided via our professor. (6 chosen .MIDI files; provided link below)
- **Classes**: Specified Six composers 
- **Preprocessing**: MIDI files cleaned and processed using the `music21` library.

Source: https://drive.google.com/drive/folders/1l_EGBktIGjNO3djMq-HYJ11nEgeNvTa_ 

---
## Models
1. **Support Vector Machine (SVM)**  

2. **Logistic Regression**    

3. **Gradient Boosting**  

---
## Requirements
- **Python** 
- **Libraries**:  
  - Core: `numpy`, `pandas`, `scikit-learn`  
  - Music Analysis: `music21`  
  - Visualization: `matplotlib`, `seaborn`
    
---
## Usage
**1. Preprocess MIDI Files:**
preprocessed_data = preprocess_midi(midi_data)

**2. Feature Extraction:**
Use music21 to extract relevant features from MIDI data.

**3. Train Machine Learning Models:**

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

from sklearn.svm import SVC
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

**4. Evaluate the Model:**

from sklearn.metrics import classification_report
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

---

## Results
**Accuracy: Achieved >90% across all models.**

**Confusion Matrix: Demonstrates clear separation between composer classes.**

**Insights: Feature engineering, especially N-grams, significantly improved classification accuracy.**

---

## Acknowledgments

**Libraries Used:**

music21: For feature extraction from MIDI files.

scikit-learn: For machine learning algorithms and evaluation.

matplotlib & seaborn: For visualizing results and metrics.

---

Thank You!
