from flask import Flask, render_template, request
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import os

app = Flask(__name__)

model = BayesianNetwork([
    ('Age', 'Breast Cancer'),
    ('Family History', 'Breast Cancer'),
    ('Breast Cancer', 'Mammogram Result'),
    ('Breast Cancer', 'Lump')
])

cpd_age = TabularCPD(variable='Age', variable_card=2, values=[[0.7], [0.3]])  # P(Age > 50) = 0.3
cpd_family_history = TabularCPD(variable='Family History', variable_card=2, values=[[0.85], [0.15]])  # P(Family History) = 0.15

cpd_breast_cancer = TabularCPD(variable='Breast Cancer', variable_card=2,
                               values=[[0.98, 0.95, 0.88, 0.75],  # P(No Cancer | Age, Family History)
                                       [0.02, 0.05, 0.12, 0.25]],  # P(Cancer | Age, Family History)
                               evidence=['Age', 'Family History'], evidence_card=[2, 2])

cpd_mammogram = TabularCPD(variable='Mammogram Result', variable_card=2,
                           values=[[0.87, 0.1],  # P(Mammogram Positive | Cancer)
                                   [0.13, 0.9]],  # P(Mammogram Negative | No Cancer)
                           evidence=['Breast Cancer'], evidence_card=[2])

cpd_lump = TabularCPD(variable='Lump', variable_card=2,
                      values=[[0.8, 0.1],  # P(Lump detected | Cancer)
                              [0.2, 0.9]],  # P(No Lump detected | No Cancer)
                      evidence=['Breast Cancer'], evidence_card=[2])

# Add CPDs to the model
model.add_cpds(cpd_age, cpd_family_history, cpd_breast_cancer, cpd_mammogram, cpd_lump)
model.check_model()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ""  # Change None to an empty string
    if request.method == 'POST':
        try:
            age = int(request.form.get('age'))
            family_history = int(request.form.get('family_history'))
            mammogram_result = int(request.form.get('mammogram_result'))
            lump = int(request.form.get('lump'))

            # Validate inputs
            if age not in [0, 1] or family_history not in [0, 1] or mammogram_result not in [0, 1] or lump not in [0, 1]:
                raise ValueError("Invalid input value.")

            # Inference
            infer = VariableElimination(model)
            result = infer.query(variables=['Breast Cancer'], evidence={
                'Age': age,
                'Family History': family_history,
                'Mammogram Result': mammogram_result,
                'Lump': lump
            })
            prediction = result.values[1]
            if prediction < 0.2:
                print("No signs of cancer detected.")
            elif prediction < 0.5:
                print("Very low chance of cancer. It's advisable to practice healthy habits.")
            elif prediction < 0.8:
                print("Moderate risk of cancer. Consulting a doctor for further assessment is recommended.")
            else:
                print("Cancer is diagnosed. Please seek immediate medical treatment.")


        except Exception as e:
            prediction = f"Error in input values: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0',debug=True)
