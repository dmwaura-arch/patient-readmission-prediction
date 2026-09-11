#data generation
import numpy as np
import pandas as pd
#Reproducibility(We generate using Random Processes and 42 is the mostr common seed and it basically makes the randomness reproducible)
np.random.seed(42)
#Number of Synthetic Patients(now we used 5000 because we want python to generate 5000 patients records and N is what we shall continuously use instead of repeating the 5000)
N = 5000
#Generating unique patient ids as identifiers
patient_ids = [f"P{i:05d}" for i in range(1, N + 1)]
# Generate patient ages..(I am choosing adult age because I am at liberty since its synthetic...)
age = np.random.randint(18, 91, size=N)
# Generate patient sex(so we are using strings here because it is raw dataset but we shall convert it to categorical during feature engineering for the model to be able to learn it)
sex = np.random.choice(["Female", "Male"], size=N)
# Generate length of hospital stay..(use + 1 so that the minimum stay is one day and N roughly centred at 5 days)
length_of_stay = np.random.poisson(lam=5, size=N) + 1
# Generate number of previous hospital admissions(use lam=1.5 as avg no of previous admissions)
previous_admissions = np.random.poisson(lam=1.5, size=N)
# Generate number of previous emergency department visits(using lam=1.2 to be less than the previous admisi=sions)
emergency_visits = np.random.poisson(lam=1.2, size=N)
# Generate number of chronic conditions(the average patient has approximately 2 chronic conditions)
chronic_conditions = np.random.poisson(lam=2, size=N)
# Generate number of medications(lam=6 to show a person could be using an avg of 6 drugs )
number_of_medications = np.random.poisson(lam=6, size=N)
# Generate number of abnormal laboratory indicators(lam=1.5 gives us a synthetic population where the average is around 1.5 abnormalities.)
abnormal_lab_count = np.random.poisson(lam=1.5, size=N)
# Generate previous readmission history(random.bionomia to perform binary trials for N,1 to meanone trial and 0.25 to give a 25% probability of previous readmission)
previous_readmission = np.random.binomial(1, 0.25, size=N)
# Generate post-discharge follow-up status(0.70 to show an assumption that 70% finished follow -up )
follow_up_completed = np.random.binomial(1, 0.70, size=N)
# Generate insurance type(55%, 35%, 10% of each respectively...well there could be assocaition though not to establish causation)
insurance_type = np.random.choice(
    ["Public", "Private", "Uninsured"],
    size=N,
    p=[0.55, 0.35, 0.10]
)
# Generate discharge disposition(as the perecntages suggest...)
discharge_disposition = np.random.choice(
    ["Home", "Home_Health", "Skilled_Nursing", "Other"],
    size=N,
    p=[0.60, 0.20, 0.15, 0.05]
)
print("Number of patients:", N)
print("First 5 patient IDs:", patient_ids[:5])
print("First 5 ages:", age[:5])
print("First 5 sexes:", sex[:5])
print("First 5 lengths of stay:", length_of_stay[:5])
# Create the patient dataset
data = pd.DataFrame({
    "Patient_ID": patient_ids,
    "Age": age,
    "Sex": sex,
    "Length_of_Stay": length_of_stay,
    "Previous_Admissions": previous_admissions,
    "Emergency_Visits": emergency_visits,
    "Chronic_Conditions": chronic_conditions,
    "Number_of_Medications": number_of_medications,
    "Abnormal_Lab_Count": abnormal_lab_count,
    "Previous_Readmission": previous_readmission,
    "Follow_Up_Completed": follow_up_completed,
    "Insurance_Type": insurance_type,
    "Discharge_Disposition": discharge_disposition
})
# Inspect the first five patients
print(data.head())
# Check the shape of the dataset
print("Dataset shape:", data.shape)
# Calculate an underlying readmission risk score
risk_score = (
    -3.2
    + 0.25 * previous_admissions
    + 0.20 * emergency_visits
    + 0.30 * chronic_conditions
    + 0.08 * length_of_stay
    + 0.70 * previous_readmission
    - 0.50 * follow_up_completed
)
# Convert risk score into a probability
readmission_probability = 1 / (1 + np.exp(-risk_score))
# Generate the 30-day readmission outcome
readmitted_30_days = np.random.binomial(
    1,
    readmission_probability,
    size=N
)
# Add target variable to the dataset
data["Readmitted_30_Days"] = readmitted_30_days
print(data["Readmitted_30_Days"].value_counts())
print(data["Readmitted_30_Days"].value_counts(normalize=True))
#at this point our output gave a 70%-30% but because we had the intention of having a 80%-20%, we can modify the baseline risk/intercept:)
#We have to make readmission less likely overall
#The coefficients in our synthetic data-generation process control the distribution of the target.
#We can experiment with the baseline and inspect the resulting proportion.
#Lets go back to tune the readmission risk score..(change -2.5 to -3.2)because the intercept controls the overall baseline risk.
# Save the raw synthetic dataset..we are using index=false because pandas usually add its own row index when saving)
data.to_csv("data/raw/patient_readmission_raw.csv", index=False)

print("Raw dataset saved successfully.")