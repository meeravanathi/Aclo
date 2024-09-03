# ACLO - THE LOAN APPROVER

## Project Overview
This project involves building a Decision Tree model to classify loan applications based on various applicant data. The goal is to predict whether a loan application should be approved or not.

## Data Description
The dataset used for this project contains the following key features:
- **ApplicantIncome**: Income of the applicant.
- **CoapplicantIncome**: Income of the coapplicant (if any).
- **LoanAmount**: Loan amount requested by the applicant.
- **Loan_Amount_Term**: Term of the loan in months.
- **Credit_History**: Credit history of the applicant.
- **Property_Area**: Urban, Semi-Urban, or Rural area of the applicant.
- **Education**: Education level of the applicant (Graduate/Not Graduate).
- **Self_Employed**: Whether the applicant is self-employed or not.
- **Loan_Status**: Target variable indicating loan approval status (Y/N).

## Modeling Process
1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and splitting the dataset into training and testing sets.
2. **Model Selection**: Implementing a Decision Tree classifier to build the model.
3. **Model Evaluation**: Evaluating the model using accuracy.
   
## GUI Implementation
The project includes a Tkinter-based graphical user interface (GUI) to:
- Display the input fields for user data.
- Allow users to enter applicant details.
- Display the model's prediction of whether a loan should be approved or not.

### Key Features of the GUI:
- **Input Fields**: Text boxes and dropdowns for all relevant features (e.g., income, loan amount, credit history).
- **Submit Button**: Triggers the prediction based on user inputs.
- **Result Display**: Shows the loan approval result on the GUI window.
## Requirements
- Python 3.x
- Pandas
- NumPy
- Tkinter (built-in with Python)
- Scikit-learn
- Matplotlib

## DECSION TREE
![image](https://github.com/user-attachments/assets/d02fbcca-158b-4db7-a8e2-c96ff062418f)

## THE GUI
![image](https://github.com/user-attachments/assets/8e8d032a-9b22-41f0-a5a9-5063e9c7de44)

## LOAN ACCEPTED
![image](https://github.com/user-attachments/assets/b8adc435-fdcd-4b46-a49e-dc0e6a528cbc)

## LOAN DENIED
![Screenshot 2024-09-03 162526](https://github.com/user-attachments/assets/6b1071a1-cd1e-48ca-8ea1-166ace35f1f1)

## Installation
To get started with this project, clone the repository and install the required dependencies:
