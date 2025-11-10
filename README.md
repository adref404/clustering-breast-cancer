# 🏥 Sistema de Clustering para Segmentación de Pacientes - Cáncer de Mama

## 📋 Descripción

Aplicación web interactiva desarrollada con **Streamlit** y **Docker** que implementa técnicas de **Machine Learning No Supervisado** para la segmentación automática de pacientes con cáncer de mama. Utiliza el dataset Wisconsin Diagnostic Breast Cancer para agrupar pacientes en clusters homogéneos mediante algoritmos interpretables de "caja blanca".

## ✨ Características Principales

- **🤖 Algoritmos de Clustering:** K-Means y Clustering Jerárquico Aglomerativo
- **📊 Optimización Automática:** Grid Search con métricas Silhouette Score y Davies-Bouldin Index
- **🔬 Visualización PCA:** Reducción dimensional a 2 componentes para visualización interactiva
- **📈 Gráficos Interactivos:** Implementados con Plotly Express
- **🐳 Dockerizado:** Ejecución con un solo comando (`docker-compose up`)
- **🔧 Preprocesamiento:** Normalización de datos con StandardScaler

## 🛠️ Tecnologías Utilizadas

- Python 3.10
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Plotly
- Docker & Docker Compose

## 🚀 Inicio Rápido

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/clustering-breast-cancer.git
cd clustering-breast-cancer

# Ejecutar con Docker
docker-compose up --build

# Abrir en el navegador
http://localhost:8501
```

## 📊 Dataset

**Wisconsin Diagnostic Breast Cancer Dataset**
- 569 pacientes
- 30 características médicas
- Fuente: UCI Machine Learning Repository / Scikit-learn

## 🎯 Casos de Uso

- Segmentación de pacientes para tratamientos personalizados
- Identificación de grupos de riesgo
- Análisis exploratorio de patrones clínicos
- Investigación médica y oncológica

## 📝 Licencia

MIT License

---

**Desarrollado como parte del proyecto MLNS - Caja Blanca | Machine Learning No Supervisado**