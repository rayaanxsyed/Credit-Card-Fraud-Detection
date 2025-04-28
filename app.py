from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib  


app = Flask(__name__)


model = joblib.load('credit_card_model.pkl')  

# Load dataset
full_data = pd.read_csv('creditcard.csv')

columns_to_use = ['V' + str(i) for i in range(1, 29)] + ['Amount']


normal_data = full_data[full_data['Class'] == 0]
fraud_data = full_data[full_data['Class'] == 1]


sampled_normal = normal_data.sample(6, random_state=42)
sampled_fraud = fraud_data.sample(4, random_state=42)


sampled_data = pd.concat([sampled_normal, sampled_fraud]).reset_index(drop=True)

sampled_data = sampled_data.sample(frac=1, random_state=42).reset_index(drop=True)

@app.route('/')
def home():
    table_html = sampled_data[['V1', 'V2', 'V3','V4', 'V5', 'Class']].to_html(classes='data', header="true", index=True)
    return render_template('index.html', table=table_html, row_ids=list(sampled_data.index))

@app.route('/predict', methods=['POST'])
def predict():
    row_id = int(request.form['row_id'])
    
    transaction_row = sampled_data.loc[row_id, columns_to_use] 
    values_list = transaction_row.tolist()
    prediction = model.predict([values_list])[0]

    if prediction == 1:
        result = "⚠️ Fraud Detected!"
    else:
        result = "✅ Transaction is Safe!"

    return jsonify({'result': result})

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    manual_input = request.form['manual_input']

    try:
        values = [float(x.strip()) for x in manual_input.split(',')]

        if len(values) != 29:
            return jsonify({'result': '❌ Error: You must enter exactly 29 values!'})

        prediction = model.predict([values])[0]

        if prediction == 1:
            result = "⚠️ Fraud Detected!"
        else:
            result = "✅ Transaction is Safe!"

        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'result': f'❌ Error: Invalid input. Please enter 29 comma-separated values. Error: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
