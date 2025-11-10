import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
import io
import base64
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 

app = Flask(__name__)

model_filename = 'diabetes_pipeline.joblib'
try:
    pipeline = joblib.load(model_filename)
except Exception as e:
    print(f"Error al cargar el pipeline de predicción: {e}")
    pipeline = None

try:
    df_viz = pd.read_csv('diabetes.csv')
    cols_with_zeros_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_viz[cols_with_zeros_as_missing] = df_viz[cols_with_zeros_as_missing].replace(0, np.nan)
    
    features_to_plot = ['Glucose', 'BMI', 'Age']
    viz_averages = df_viz.groupby('Outcome')[features_to_plot].mean()
except Exception as e:
    print(f"Error al cargar 'diabetes.csv' para visualización: {e}")
    viz_averages = None

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
cols_with_zeros_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def create_comparison_plot(user_input_series, averages_df):
    try:
        features_to_plot = ['Glucose', 'BMI', 'Age']
        
        avg_no_diabetes = averages_df.loc[0] 
        avg_diabetes = averages_df.loc[1]            
        user_values = user_input_series[features_to_plot]

        n_features = len(features_to_plot)
        index = np.arange(n_features)
        bar_width = 0.25

        fig, ax = plt.subplots(figsize=(10, 5))

        color_no_diabetes = '#64b5f6' 
        color_diabetes = '#3f51b5'     
        color_user = '#ff9800'         

        bar1 = ax.bar(index, avg_no_diabetes, bar_width, label='Promedio (No Diabético)', color=color_no_diabetes)
        bar2 = ax.bar(index + bar_width, avg_diabetes, bar_width, label='Promedio (Diabético)', color=color_diabetes)
        bar3 = ax.bar(index + 2 * bar_width, user_values, bar_width, label='Su Entrada', color=color_user)

        ax.set_ylabel('Valor')
        ax.set_title('Comparación vs. Promedios')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(features_to_plot)
        ax.legend()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        return f"data:image/png;base64,{image_base64}"

    except Exception as e:
        print(f"Error al crear el gráfico: {e}")
        return None


@app.route('/')
def home():
    default_form_data = {
        'Pregnancies': 1,
        'Glucose': 117,
        'BloodPressure': 72,
        'SkinThickness': 23,
        'Insulin': 30,
        'BMI': 32.0,
        'DiabetesPedigreeFunction': 0.372,
        'Age': 29
    }
    return render_template('index.html', form_data=default_form_data)

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return render_template('index.html', 
                               prediction_label='Error',
                               probability_text='El modelo no se ha cargado correctamente.',
                               prediction_class='no-compra',
                               form_data=request.form 
                               )

    try:
        form_values = [
            request.form.get('Pregnancies', type=int),
            request.form.get('Glucose', type=float),
            request.form.get('BloodPressure', type=float),
            request.form.get('SkinThickness', type=float),
            request.form.get('Insulin', type=float),
            request.form.get('BMI', type=float),
            request.form.get('DiabetesPedigreeFunction', type=float),
            request.form.get('Age', type=int)
        ]

        processed_values = []
        for col_name, value in zip(feature_names, form_values):
            if col_name in cols_with_zeros_as_missing and value == 0:
                processed_values.append(np.nan)
            else:
                processed_values.append(value)
        
        input_df = pd.DataFrame([processed_values], columns=feature_names)

        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)
        prob_diabetes = probability[0][1] * 100

        if prediction[0] == 1:
            prediction_label = 'Diabético'
            prediction_class = 'no-compra' 
            probability_text = f'Probabilidad: {prob_diabetes:.2f}%'
        else:
            prediction_label = 'No Diabético'
            prediction_class = 'compra' 
            probability_text = f'Probabilidad de diabetes: {prob_diabetes:.2f}%'
            
        plot_url = None
        if viz_averages is not None:
            user_input_series = input_df.iloc[0].copy()
            user_input_series.fillna(0, inplace=True) 
            plot_url = create_comparison_plot(user_input_series, viz_averages)

        # 7. Devolver las variables separadas
        return render_template('index.html', 
                               prediction_label=prediction_label,
                               prediction_class=prediction_class,
                               probability_text=probability_text,
                               plot_url=plot_url,
                               form_data=request.form 
                               )

    except Exception as e:
        error_text = f"Error: {e}"
        return render_template('index.html', 
                               prediction_label='Error',
                               probability_text=error_text,
                               prediction_class='no-compra',
                               form_data=request.form 
                               )

if __name__ == "__main__":
    app.run(debug=True)