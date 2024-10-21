import time
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from io import BytesIO
from sqlalchemy import create_engine
from sklearn.metrics import classification_report

# Al principio del script, definir una lista para almacenar todos los resultados
all_results = []

# Función para cargar un modelo
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Función para leer datos desde un archivo
def read_data(file):
    try:
        if file is not None:
            content = file.read()
            if file.name.endswith('.csv'):
                df = pd.read_csv(BytesIO(content))
            elif file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(BytesIO(content))
            else:
                st.error('Formato de archivo no admitido. Por favor, sube un archivo CSV o Excel.')
                return None
            return df
        else:
            st.error('Ningún archivo cargado.')
            return None
    except Exception as e:
        st.error(f"Error al leer el archivo: {str(e)}")
        return None

# Función para leer datos desde MySQL
def read_data_from_mysql():
    try:
        connection_str = "mysql+pymysql://root:root@localhost:3307/detectionfraud"
        engine = create_engine(connection_str)
        query = "SELECT * FROM data"  # Ajusta la consulta según tu estructura
        with engine.connect() as conn:
            df = pd.read_sql(query, con=conn)
        return df
    except Exception as e:
        st.error(f"Error al leer desde MySQL: {str(e)}")
        return None

# Función de preprocesamiento para nuevos datos
def preprocesamiento_nuevos_datos(datos, scaler, reduction_results):
    # Eliminar columnas no relevantes
    datos = datos.drop(['gvkey', 'p_aaer', 'fyear'], axis='columns')

    # Imputar valores perdidos con cero
    columns_to_impute = ['dch_wc', 'ch_rsst', 'dch_rec', 'dch_inv', 'ch_cs', 'ch_cm', 'ch_roa', 'bm', 'reoa', 'EBIT', 'ch_fcf', 'soft_assets', 'dpi']
    datos[columns_to_impute] = datos[columns_to_impute].fillna(0)

    # Guardar 'misstate' en una variable temporal y eliminar la columna si está presente
    misstate_column = datos['misstate'] if 'misstate' in datos.columns else None
    if 'misstate' in datos.columns:
        datos = datos.drop('misstate', axis=1)

    # Escalar datos con el mismo scaler utilizado durante el entrenamiento
    datos_scaled = scaler.transform(datos)

    # Reducir características con las mismas características seleccionadas durante el entrenamiento
    selected_features = reduction_results['selected_features']
    datos_reducidos = datos[selected_features]

    # Agregar 'misstate' nuevamente si estaba presente originalmente
    if misstate_column is not None:
        datos_reducidos.loc[:, 'misstate'] = misstate_column

    return datos_reducidos

# Función para el análisis de modelos no supervisados
def analyze_unsupervised_model(model, preprocessed_data, model_name):
    try:
        if 'misstate' in preprocessed_data.columns:
            X = preprocessed_data.drop('misstate', axis=1)
        else:
            st.error("La columna 'misstate' no está presente en los datos después del preprocesamiento.")
            return

        # Verificar si el modelo LOF tiene la opción novelty=True
        if hasattr(model, 'novelty') and model.novelty:
            # Si se está utilizando LOF para detección de novedades, utiliza fit_predict
            anomalies = model.fit_predict(X)
        else:
            # De lo contrario, utiliza fit_predict en lugar de predict
            anomalies = model.fit_predict(X)

        # Convertir las predicciones (-1 para anomalía, 1 para normal) a 0 y 1 (0 para anomalía, 1 para normal)
        anomalies[anomalies == 1] = 0
        anomalies[anomalies == -1] = 1

        # Mostrar resultados de las métricas
        y_true = preprocessed_data['misstate'].values

        accuracy = round(accuracy_score(y_true, anomalies) * 100, 0)
        precision = round(precision_score(y_true, anomalies) * 100, 0)
        recall = round(recall_score(y_true, anomalies) * 100, 0)
        f1 = round(f1_score(y_true, anomalies) * 100, 0)
        roc_auc = round(roc_auc_score(y_true, anomalies) * 100, 0)

        # Almacenar los resultados en la lista
        result = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        all_results.append(result)

        # Mostrar resultados de las métricas
        st.write("Métricas de evaluación:")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"Precisión: {precision}")
        st.write(f"Recall: {recall}")
        st.write(f"F1 Score: {f1}")

        # Matriz de confusión
        st.write("Matriz de Confusión:")
        st.write(confusion_matrix(y_true, anomalies))

    except Exception as e:
        st.error(f"Error durante el análisis no supervisado: {str(e)}")

# Función para el análisis de modelos supervisados
def analyze_supervised_model(model, preprocessed_data, model_name):
    try:
        if 'misstate' not in preprocessed_data.columns:
            st.error("La columna 'misstate' no está presente en los datos después del preprocesamiento.")
        else:
            # Aquí se realiza la predicción directamente
            anomalies = model.predict(preprocessed_data.drop('misstate', axis=1))

            # Mostrar resultados de las métricas
            y_true = preprocessed_data['misstate'].values
            accuracy = round(accuracy_score(y_true, anomalies) * 100, 0)
            precision = round(precision_score(y_true, anomalies) * 100, 0)
            recall = round(recall_score(y_true, anomalies) * 100, 0)
            f1 = round(f1_score(y_true, anomalies) * 100, 0)
            roc_auc = round(roc_auc_score(y_true, anomalies) * 100, 0)

            # Almacenar los resultados en la lista
            result = {
                'model_name': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            }
            all_results.append(result)

            # Mostrar resultados de las métricas
            st.write("Métricas de evaluación:")
            st.write(f"Accuracy: {accuracy}")
            st.write(f"Precisión: {precision}")
            st.write(f"Recall: {recall}")
            st.write(f"F1 Score: {f1}")

            # Matriz de confusión
            st.write("Matriz de Confusión:")
            st.write(confusion_matrix(y_true, anomalies))

            # AUC-ROC Score (manejando problemas multiclase)
            if len(np.unique(y_true)) > 2:
                st.warning("El problema es multiclase. AUC-ROC no se calcula para problemas multiclase.")
            else:
                st.write(f"AUC-ROC Score: {roc_auc}")

    except Exception as e:
        st.error(f"Error durante el análisis supervisado: {str(e)}")

# Función principal para la página de detección de anomalías
def auto_detection_page():
    st.title("SISTEMA DE DETECCIÓN DE ANOMALÍAS")

    MODELS_UNSUPERVISED = {
        'Modelo Isolation Forest': 'models/if_model.pkl',
        'Modelo Local Outlier Factor': 'models/lof_model.pkl',
        'Modelo One Class Support Vector': 'models/ocsvm_model.pkl',
    }

    MODELS_SUPERVISED = {
        'Modelo Logistic Regresion': 'models/logistic3_model.pkl',
        'Modelo Random Forest': 'models/rf_model.pkl',
        'Modelo Support Vector Machine': 'models/svm_model.pkl',
    }

    # Cargar los modelos y otros objetos necesarios
    with open('preprocessing/scaler.pkl', 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)

    with open('preprocessing/reduction_results.pkl', 'rb') as reduction_file:
        loaded_reduction_results = pickle.load(reduction_file)

    data_option = st.radio("Seleccione origen de datos:", ("Cargar archivo CSV o Excel", "Base de datos MySQL"))

    selected_model = None
    data = None

    if data_option == "Cargar archivo CSV o Excel":
        uploaded_file = st.file_uploader("Cargar archivo CSV o Excel")
        data = read_data(uploaded_file)

    else:
        data = read_data_from_mysql()

    if data is not None and not data.empty:
        # Aplicar preprocesamiento a los nuevos datos
        preprocessed_data = preprocesamiento_nuevos_datos(data, loaded_scaler, loaded_reduction_results)

        model_type = st.radio("Seleccione el tipo de modelo:", ("Modelos No Supervisados", "Modelos Supervisados"))

        if model_type == "Modelos No Supervisados":
            selected_model = st.selectbox("Seleccione un modelo", list(MODELS_UNSUPERVISED.keys()))
            if selected_model and st.button("Realizar Análisis"):
                model = load_model(MODELS_UNSUPERVISED[selected_model])
                analyze_unsupervised_model(model, preprocessed_data, selected_model)

        else:
            selected_model = st.selectbox("Seleccione un modelo", list(MODELS_SUPERVISED.keys()))
            if selected_model and st.button("Realizar Análisis"):
                model = load_model(MODELS_SUPERVISED[selected_model])
                analyze_supervised_model(model, preprocessed_data, selected_model)

    # Al final de la función auto_detection_page(), después de realizar todos los análisis
    # Puedes mostrar o almacenar la lista de resultados
    #st.write("Resultados de todos los modelos analizados:")
    #st.write(all_results)

