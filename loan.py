import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
#reading csv
data = pd.read_csv('loan_approval_dataset.csv')
data['total_assets_value'] = (data[' residential_assets_value'] + data[' commercial_assets_value'] +
                            data[' luxury_assets_value'] + data[' bank_asset_value'])
#feature engineering
le = LabelEncoder()

data[' education'] = le.fit_transform(data[' education'])
dict(zip(le.classes_, le.transform(le.classes_)))
data[' self_employed'] = le.fit_transform(data[' self_employed'])
dict(zip(le.classes_, le.transform(le.classes_)))

#scaling
scaler = StandardScaler()
data[[ ' income_annum', ' loan_amount']] = scaler.fit_transform(data[[ ' income_annum', ' loan_amount']])
data['debt_to_income_ratio'] = data[' loan_amount'] / data[' income_annum']

data['loan_status'] = le.fit_transform(data[' loan_status'])
dict(zip(le.classes_, le.transform(le.classes_)))

#drop columns
columns_to_drop = ['loan_id',' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value', ' loan_status','debt_to_income_ratio']
data = data.drop(columns=columns_to_drop)

#trAININGGG
X = data.drop('loan_status', axis=1)
y = data['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

#metrics
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
#visualise tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=le.classes_, filled=True, fontsize=10)
plt.title('Decision Tree Visualization')
plt.show()

#param tuning
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f'Best parameters: {grid_search.best_params_}')
best_clf = grid_search.best_estimator_
y_pred_best = best_clf.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)


#tkinter
root = tk.Tk()
root.title("Loan Approval System")

no_of_dependents = tk.IntVar()
education = tk.IntVar()
self_employed = tk.IntVar()
income_annum = tk.IntVar()
loan_amount = tk.IntVar()
loan_term = tk.IntVar()
cibil_score = tk.DoubleVar()
total_assets_value = tk.IntVar()

def predict_loan_approval():
    # Collect inputs into a DataFrame
    new_application = pd.DataFrame({
        ' no_of_dependents': [no_of_dependents.get()],
        ' education': [education.get()],
        ' self_employed': [self_employed.get()],
        ' income_annum': [income_annum.get()],
        ' loan_amount': [loan_amount.get()],
        ' loan_term': [loan_term.get()],
        ' cibil_score': [cibil_score.get()],
        'total_assets_value': [total_assets_value.get()],
    })
    prediction = clf.predict(new_application)
    loan_status = le.inverse_transform(prediction)
    print(f'The predicted loan status is: {loan_status[0]}')
    # Display the result
    if loan_status== " Approved":
        messagebox.showinfo("Loan Status", "The loan is Approved!")
    else:
        messagebox.showwarning("Loan Status", "The loan is Not Approved.")
    
fields = [
    ("Number of Dependents", no_of_dependents),
    ("Education Level (1 for Graduate, 0 for Not Graduate)", education),
    ("Self Employed (1 for Yes, 0 for No)", self_employed),
    ("Annual Income", income_annum),
    ("Loan Amount", loan_amount),
    ("Loan Term (years)", loan_term),
    ("CIBIL Score", cibil_score),
    ("Total Assets Value", total_assets_value)
]

for i, (label_text, var) in enumerate(fields):
    label = tk.Label(root, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root, textvariable=var)
    entry.grid(row=i, column=1, padx=10, pady=5)

# Add a button to submit the form
submit_button = tk.Button(root, text="Check Loan Status", command=predict_loan_approval)
submit_button.grid(row=len(fields), column=0, columnspan=2, pady=20)

# Start the Tkinter event loop
root.mainloop()