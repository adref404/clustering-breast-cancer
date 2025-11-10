import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Segmentación de Pacientes - Cáncer de Mama",
    page_icon="🏥",
    layout="wide"
)

# Título principal
st.title("🏥 Sistema de Clustering para Segmentación de Pacientes")
st.markdown("### Análisis No Supervisado - Cáncer de Mama (Caja Blanca)")
st.markdown("---")

# Función para cargar datos
@st.cache_data
def load_data():
    """Carga el dataset de Breast Cancer desde sklearn"""
    from sklearn.datasets import load_breast_cancer
    
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target  # 0=malignant, 1=benign (solo para referencia)
    
    return df, data.feature_names

# Cargar datos
df, feature_names = load_data()

# Mostrar información del dataset
st.sidebar.header("⚙️ Configuración del Modelo")
st.sidebar.markdown("---")

# Mostrar dataset
with st.expander("📊 Ver Dataset Original", expanded=False):
    st.dataframe(df, use_container_width=True)
    st.info(f"**Dimensiones:** {df.shape[0]} pacientes × {df.shape[1]} características")

# Preparar datos para clustering (sin la columna target)
X = df.drop('target', axis=1)

# Normalización de datos
st.sidebar.subheader("🔧 Preprocesamiento")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
st.sidebar.success("✓ Datos normalizados con StandardScaler")

# Selección del algoritmo
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Algoritmo de Clustering")
algorithm = st.sidebar.selectbox(
    "Selecciona el algoritmo:",
    ["K-Means", "Clustering Jerárquico Aglomerativo"]
)

# Configuración de hiperparámetros
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Hiperparámetros")

k_range = range(2, 11)
selected_k = st.sidebar.slider(
    "Número de clusters (k):",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)

# Botón para ejecutar análisis
run_analysis = st.sidebar.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True)

if run_analysis:
    # Sección 1: Grid Search para encontrar k óptimo
    st.header("📊 1. Optimización de Hiperparámetros (Grid Search)")
    
    with st.spinner("Calculando métricas para diferentes valores de k..."):
        silhouette_scores = []
        davies_bouldin_scores = []
        
        for k in k_range:
            if algorithm == "K-Means":
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                model = AgglomerativeClustering(n_clusters=k)
            
            labels = model.fit_predict(X_scaled)
            
            # Calcular métricas
            sil_score = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            
            silhouette_scores.append(sil_score)
            davies_bouldin_scores.append(db_score)
    
    # Crear gráficos de métricas
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sil = px.line(
            x=list(k_range),
            y=silhouette_scores,
            markers=True,
            title="Silhouette Score vs Número de Clusters",
            labels={'x': 'Número de Clusters (k)', 'y': 'Silhouette Score'}
        )
        fig_sil.update_traces(line_color='#1f77b4', marker=dict(size=10))
        fig_sil.update_layout(height=400)
        st.plotly_chart(fig_sil, use_container_width=True)
        st.info("📈 **Mayor Silhouette Score = Mejor** (rango: -1 a 1)")
    
    with col2:
        fig_db = px.line(
            x=list(k_range),
            y=davies_bouldin_scores,
            markers=True,
            title="Davies-Bouldin Index vs Número de Clusters",
            labels={'x': 'Número de Clusters (k)', 'y': 'Davies-Bouldin Index'}
        )
        fig_db.update_traces(line_color='#ff7f0e', marker=dict(size=10))
        fig_db.update_layout(height=400)
        st.plotly_chart(fig_db, use_container_width=True)
        st.info("📉 **Menor Davies-Bouldin = Mejor** (≥ 0)")
    
    # Recomendación de k óptimo
    optimal_k_sil = list(k_range)[np.argmax(silhouette_scores)]
    optimal_k_db = list(k_range)[np.argmin(davies_bouldin_scores)]
    
    st.success(f"💡 **K óptimo según Silhouette:** {optimal_k_sil} | **K óptimo según Davies-Bouldin:** {optimal_k_db}")
    
    st.markdown("---")
    
    # Sección 2: Clustering con k seleccionado
    st.header(f"🎯 2. Resultados del Clustering (k={selected_k})")
    
    # Entrenar modelo con k seleccionado
    if algorithm == "K-Means":
        final_model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    else:
        final_model = AgglomerativeClustering(n_clusters=selected_k)
    
    clusters = final_model.fit_predict(X_scaled)
    
    # Calcular métricas finales
    final_silhouette = silhouette_score(X_scaled, clusters)
    final_db = davies_bouldin_score(X_scaled, clusters)
    
    # Mostrar métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("🔵 Algoritmo", algorithm)
    col2.metric("📊 Silhouette Score", f"{final_silhouette:.4f}")
    col3.metric("📉 Davies-Bouldin Index", f"{final_db:.4f}")
    
    st.markdown("---")
    
    # Sección 3: Visualización con PCA
    st.header("🔬 3. Visualización con PCA (2 Componentes)")
    
    with st.spinner("Aplicando PCA y generando visualización..."):
        # Aplicar PCA para reducción a 2D
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Crear DataFrame para visualización
        df_pca = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': clusters.astype(str),
            'Target_Real': df['target'].map({0: 'Maligno', 1: 'Benigno'})
        })
        
        # Varianza explicada
        var_explained = pca.explained_variance_ratio_
        st.info(f"📊 **Varianza explicada:** PC1 = {var_explained[0]:.2%} | PC2 = {var_explained[1]:.2%} | Total = {var_explained.sum():.2%}")
        
        # Gráfico de dispersión interactivo
        fig_pca = px.scatter(
            df_pca,
            x='PC1',
            y='PC2',
            color='Cluster',
            title=f'Visualización de Clusters en Espacio PCA ({algorithm})',
            labels={'PC1': f'PC1 ({var_explained[0]:.1%})', 'PC2': f'PC2 ({var_explained[1]:.1%})'},
            hover_data=['Target_Real'],
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_pca.update_traces(marker=dict(size=8, line=dict(width=0.5, color='white')))
        fig_pca.update_layout(height=600)
        st.plotly_chart(fig_pca, use_container_width=True)
    
    st.markdown("---")
    
    # Sección 4: Distribución de Clusters
    st.header("📦 4. Distribución de Pacientes por Cluster")
    
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(
            pd.DataFrame({
                'Cluster': cluster_counts.index,
                'Pacientes': cluster_counts.values,
                'Porcentaje': (cluster_counts.values / len(clusters) * 100).round(2)
            }),
            use_container_width=True
        )
    
    with col2:
        fig_dist = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title="Número de Pacientes por Cluster",
            labels={'x': 'Cluster', 'y': 'Número de Pacientes'},
            color=cluster_counts.index.astype(str),
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_dist.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown("---")
    
    # Sección 5: Análisis de Características por Cluster
    st.header("🔍 5. Características Promedio por Cluster")
    
    df_analysis = df.copy()
    df_analysis['Cluster'] = clusters
    
    # Top 10 características más importantes
    top_features = list(feature_names[:10])
    cluster_profiles = df_analysis.groupby('Cluster')[top_features].mean()
    
    # Mostrar tabla con formato condicional
    try:
        st.dataframe(cluster_profiles.style.background_gradient(cmap='RdYlGn', axis=1), use_container_width=True)
    except:
        # Fallback sin estilo si hay problemas
        st.dataframe(cluster_profiles, use_container_width=True)
    
    st.success("✅ Análisis completado exitosamente!")

else:
    st.info("👈 Configura los parámetros en el panel lateral y presiona **'Ejecutar Análisis'** para comenzar.")
    
    # Mostrar información sobre el dataset
    st.header("ℹ️ Información del Dataset")
    st.markdown("""
    Este sistema analiza el **Wisconsin Diagnostic Breast Cancer Dataset** que contiene:
    
    - **569 pacientes** con diagnóstico de cáncer de mama
    - **30 características** extraídas de imágenes digitalizadas de aspiración con aguja fina (FNA)
    - Las características incluyen: radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal
    
    **Objetivo:** Segmentar pacientes en grupos homogéneos usando técnicas de clustering no supervisado.
    """)
    
    st.markdown("---")
    st.markdown("**Desarrollado por:** Data Science Team | **Dataset:** UCI Machine Learning Repository")