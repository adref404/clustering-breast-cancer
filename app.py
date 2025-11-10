import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    
    return df, data.feature_names, data.target

# Cargar datos
df, feature_names, targets = load_data()

# Sidebar - Configuración
st.sidebar.header("⚙️ Configuración del Modelo")
st.sidebar.markdown("---")

# Mostrar información del dataset
with st.expander("📊 Ver Dataset Original", expanded=False):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(df, use_container_width=True)
    with col2:
        st.metric("Pacientes", df.shape[0])
        st.metric("Características", df.shape[1])
        st.write("**Distribución Real:**")
        st.write(f"🟢 Benignos: {sum(targets == 1)}")
        st.write(f"🔴 Malignos: {sum(targets == 0)}")

# Preparar datos para clustering
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

if algorithm == "K-Means":
    st.sidebar.info("💡 **K-Means:** Algoritmo rápido que minimiza la varianza intra-cluster. Ideal para clusters esféricos.")
else:
    st.sidebar.info("💡 **Jerárquico:** Construye una jerarquía de clusters. No requiere especificar k inicialmente.")

# Configuración de hiperparámetros
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Hiperparámetros")

k_range = range(2, 11)
selected_k = st.sidebar.slider(
    "Número de clusters (k):",
    min_value=2,
    max_value=10,
    value=3,
    step=1,
    help="Número de grupos en los que se dividirán los pacientes"
)

# Botón para ejecutar análisis
run_analysis = st.sidebar.button("🚀 Ejecutar Análisis", type="primary", use_container_width=True)

if run_analysis:
    # Sección 1: Grid Search
    st.header("📊 1. Optimización de Hiperparámetros (Grid Search)")
    st.markdown("**Objetivo:** Encontrar el número óptimo de clusters evaluando métricas de calidad.")
    
    with st.spinner("🔄 Calculando métricas para diferentes valores de k..."):
        silhouette_scores = []
        davies_bouldin_scores = []
        
        for k in k_range:
            if algorithm == "K-Means":
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
            else:
                model = AgglomerativeClustering(n_clusters=k)
            
            labels = model.fit_predict(X_scaled)
            sil_score = silhouette_score(X_scaled, labels)
            db_score = davies_bouldin_score(X_scaled, labels)
            
            silhouette_scores.append(sil_score)
            davies_bouldin_scores.append(db_score)
    
    # Gráficos de métricas
    col1, col2 = st.columns(2)
    
    with col1:
        fig_sil = px.line(
            x=list(k_range),
            y=silhouette_scores,
            markers=True,
            title="📈 Silhouette Score vs Número de Clusters",
            labels={'x': 'Número de Clusters (k)', 'y': 'Silhouette Score'}
        )
        fig_sil.add_hline(y=max(silhouette_scores), line_dash="dash", line_color="green", 
                          annotation_text="Óptimo")
        fig_sil.update_traces(line_color='#1f77b4', marker=dict(size=10))
        fig_sil.update_layout(height=400)
        st.plotly_chart(fig_sil, use_container_width=True)
        st.info("📈 **Mayor Silhouette Score = Mejor separación** (rango: -1 a 1)\n\n"
                "Valores > 0.5 = excelente | 0.25-0.5 = aceptable | < 0.25 = débil")
    
    with col2:
        fig_db = px.line(
            x=list(k_range),
            y=davies_bouldin_scores,
            markers=True,
            title="📉 Davies-Bouldin Index vs Número de Clusters",
            labels={'x': 'Número de Clusters (k)', 'y': 'Davies-Bouldin Index'}
        )
        fig_db.add_hline(y=min(davies_bouldin_scores), line_dash="dash", line_color="green",
                         annotation_text="Óptimo")
        fig_db.update_traces(line_color='#ff7f0e', marker=dict(size=10))
        fig_db.update_layout(height=400)
        st.plotly_chart(fig_db, use_container_width=True)
        st.info("📉 **Menor Davies-Bouldin = Mejor compactación** (≥ 0)\n\n"
                "Valores < 1 = excelente | 1-2 = aceptable | > 2 = débil")
    
    # Análisis de k óptimo
    optimal_k_sil = list(k_range)[np.argmax(silhouette_scores)]
    optimal_k_db = list(k_range)[np.argmin(davies_bouldin_scores)]
    
    st.markdown("### 🎯 Recomendación de k Óptimo")
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Según Silhouette", optimal_k_sil, 
                delta=f"Score: {max(silhouette_scores):.4f}")
    col2.metric("📉 Según Davies-Bouldin", optimal_k_db,
                delta=f"Index: {min(davies_bouldin_scores):.4f}")
    
    # Análisis inteligente
    if optimal_k_sil == optimal_k_db:
        col3.success(f"✅ **Consenso:** k={optimal_k_sil}\n\nAmbas métricas coinciden")
    else:
        col3.warning(f"⚠️ **Discrepancia:**\nSilhouette prefiere k={optimal_k_sil}\nDavies-Bouldin prefiere k={optimal_k_db}")
    
    # Explicación contextual
    st.markdown("---")
    if optimal_k_sil == 2 and optimal_k_db == 2:
        st.info("💡 **Interpretación Clínica:** Las métricas sugieren **k=2 como óptimo**, lo cual tiene sentido médico "
                "ya que el dataset de Wisconsin está diseñado para clasificar tumores en dos categorías naturales: "
                "**Benignos** y **Malignos**. Usar k > 2 crea subdivisiones dentro de estas categorías.")
    
    st.markdown("---")
    
    # Sección 2: Clustering con k seleccionado
    st.header(f"🎯 2. Resultados del Clustering (k={selected_k})")
    
    # Entrenar modelo
    if algorithm == "K-Means":
        final_model = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    else:
        final_model = AgglomerativeClustering(n_clusters=selected_k)
    
    clusters = final_model.fit_predict(X_scaled)
    
    # Calcular métricas finales
    final_silhouette = silhouette_score(X_scaled, clusters)
    final_db = davies_bouldin_score(X_scaled, clusters)
    
    # Mostrar métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔵 Algoritmo", algorithm)
    col2.metric("📊 Silhouette Score", f"{final_silhouette:.4f}")
    col3.metric("📉 Davies-Bouldin", f"{final_db:.4f}")
    
    # Calidad del clustering
    if final_silhouette > 0.5:
        quality = "Excelente ⭐⭐⭐"
        quality_color = "green"
    elif final_silhouette > 0.25:
        quality = "Aceptable ⭐⭐"
        quality_color = "orange"
    else:
        quality = "Débil ⭐"
        quality_color = "red"
    
    col4.markdown(f"**Calidad:** :{quality_color}[{quality}]")
    
    # Justificación de k seleccionado
    if selected_k != optimal_k_sil:
        st.warning(f"⚠️ **Nota:** Seleccionaste k={selected_k}, pero el k óptimo según Silhouette es k={optimal_k_sil}. "
                   f"Esto puede ser válido si buscas mayor **granularidad clínica** para estratificar tratamientos.")
    else:
        st.success(f"✅ Seleccionaste el k óptimo (k={selected_k}). ¡Excelente elección!")
    
    st.markdown("---")
    
    # Sección 3: Visualización PCA
    st.header("🔬 3. Visualización con PCA (Análisis de Componentes Principales)")
    st.markdown("**Reducción dimensional de 30 características a 2 componentes para visualización.**")
    
    with st.spinner("🔄 Aplicando PCA y generando visualización..."):
        # Aplicar PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        # Crear DataFrame
        df_pca = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': clusters.astype(str),
            'Target_Real': df['target'].map({0: '🔴 Maligno', 1: '🟢 Benigno'})
        })
        
        var_explained = pca.explained_variance_ratio_
        
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Varianza PC1", f"{var_explained[0]:.2%}")
        col2.metric("📊 Varianza PC2", f"{var_explained[1]:.2%}")
        col3.metric("📊 Varianza Total", f"{var_explained.sum():.2%}")
        
        st.info(f"💡 Los 2 componentes principales capturan **{var_explained.sum():.1%}** de la información original "
                f"de las 30 características. Esto permite visualizar los clusters en un espacio 2D.")
        
        # Gráfico de dispersión
        fig_pca = px.scatter(
            df_pca,
            x='PC1',
            y='PC2',
            color='Cluster',
            title=f'🔬 Visualización de Clusters en Espacio PCA ({algorithm}, k={selected_k})',
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
        dist_df = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Pacientes': cluster_counts.values,
            'Porcentaje': (cluster_counts.values / len(clusters) * 100).round(2)
        })
        st.dataframe(dist_df, use_container_width=True, hide_index=True)
        
        # Análisis de balance
        max_pct = dist_df['Porcentaje'].max()
        min_pct = dist_df['Porcentaje'].min()
        
        if max_pct / min_pct > 5:
            st.warning(f"⚠️ **Desbalance:** Cluster más grande es {max_pct/min_pct:.1f}x más grande que el más pequeño")
        else:
            st.success("✅ **Balance aceptable** entre clusters")
    
    with col2:
        fig_dist = px.bar(
            dist_df,
            x='Cluster',
            y='Pacientes',
            title="📊 Número de Pacientes por Cluster",
            text='Porcentaje',
            color='Cluster',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_dist.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_dist.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    st.markdown("---")
    
    # Sección 5: Análisis de Características
    st.header("🔍 5. Perfiles Clínicos por Cluster")
    st.markdown("**Características promedio de cada cluster (Top 10 más relevantes):**")
    
    df_analysis = df.copy()
    df_analysis['Cluster'] = clusters
    
    # Top 10 características
    top_features = list(feature_names[:10])
    cluster_profiles = df_analysis.groupby('Cluster')[top_features].mean()
    
    # Mostrar tabla con formato
    try:
        st.dataframe(
            cluster_profiles.style.background_gradient(cmap='RdYlGn', axis=1).format("{:.2f}"),
            use_container_width=True
        )
    except:
        st.dataframe(cluster_profiles, use_container_width=True)
    
    # Interpretación clínica automática
    st.markdown("### 🏥 Interpretación Clínica de los Clusters")
    
    # Calcular severidad de cada cluster basado en mean radius y mean concavity
    severity_scores = cluster_profiles[['mean radius', 'mean concavity', 'mean concave points']].mean(axis=1)
    severity_ranking = severity_scores.sort_values()
    
    for idx, cluster_id in enumerate(severity_ranking.index):
        severity_level = idx + 1
        patients_count = cluster_counts[cluster_id]
        pct = (patients_count / len(clusters) * 100)
        
        if severity_level == 1:
            emoji = "🟢"
            label = "Riesgo Bajo"
            interpretation = "Tumores pequeños, regulares, probablemente **benignos**. Requieren monitoreo rutinario."
        elif severity_level == len(severity_ranking):
            emoji = "🔴"
            label = "Riesgo Alto"
            interpretation = "Tumores grandes, irregulares, probablemente **malignos agresivos**. Requieren tratamiento inmediato."
        else:
            emoji = "🟡"
            label = f"Riesgo Medio ({severity_level}/{len(severity_ranking)})"
            interpretation = "Tumores con características intermedias. Requieren evaluación detallada y posible intervención."
        
        st.markdown(f"**{emoji} Cluster {cluster_id} - {label}**")
        st.markdown(f"- Pacientes: {patients_count} ({pct:.1f}%)")
        st.markdown(f"- {interpretation}")
        st.markdown("")
    
    st.markdown("---")
    
    # Comparación con diagnóstico real
    st.header("🎯 6. Validación con Diagnóstico Real")
    
    df_validation = pd.DataFrame({
        'Cluster': clusters,
        'Diagnóstico_Real': df['target'].map({0: 'Maligno', 1: 'Benigno'})
    })
    
    validation_table = pd.crosstab(
        df_validation['Cluster'], 
        df_validation['Diagnóstico_Real'],
        margins=True
    )
    
    st.dataframe(validation_table, use_container_width=True)
    
    # Calcular pureza de clusters
    st.markdown("### 📊 Pureza de Clusters")
    purities = []
    for cluster_id in range(selected_k):
        cluster_data = df_validation[df_validation['Cluster'] == cluster_id]
        if len(cluster_data) > 0:
            purity = cluster_data['Diagnóstico_Real'].value_counts().max() / len(cluster_data)
            dominant_class = cluster_data['Diagnóstico_Real'].value_counts().idxmax()
            purities.append({
                'Cluster': cluster_id,
                'Pureza': f"{purity:.1%}",
                'Clase Dominante': dominant_class,
                'Pacientes': len(cluster_data)
            })
    
    st.dataframe(pd.DataFrame(purities), use_container_width=True, hide_index=True)
    
    st.info("💡 **Pureza:** Porcentaje de la clase más frecuente en cada cluster. "
            "Valores > 80% indican que el cluster captura bien una categoría diagnóstica.")
    
    st.markdown("---")
    st.success("✅ **Análisis completado exitosamente!** Los resultados están listos para ser interpretados.")

else:
    st.info("👈 **Instrucciones:** Configura los parámetros en el panel lateral y presiona **'Ejecutar Análisis'** para comenzar.")
    
    # Información del dataset
    st.header("ℹ️ Información del Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Wisconsin Diagnostic Breast Cancer Dataset
        
        **Descripción:**
        - 569 pacientes con diagnóstico de cáncer de mama
        - 30 características extraídas de imágenes FNA (Fine Needle Aspiration)
        - 2 clases: Benigno (357) y Maligno (212)
        
        **Características incluyen:**
        - Radio, textura, perímetro, área
        - Suavidad, compacidad, concavidad
        - Simetría, dimensión fractal
        - Para cada característica: media, error estándar y "worst" (más severo)
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Objetivo del Análisis
        
        **Segmentar pacientes en grupos homogéneos** usando técnicas de clustering no supervisado para:
        
        1. Identificar patrones en los datos sin etiquetas
        2. Agrupar pacientes con características similares
        3. Estratificar niveles de riesgo
        4. Personalizar tratamientos médicos
        5. Validar la capacidad de los algoritmos de descubrir las categorías benigno/maligno
        
        **Algoritmos disponibles:** K-Means y Clustering Jerárquico
        """)
    
    st.markdown("---")
    
    # Guía de uso
    st.header("📖 Guía de Uso")
    
    with st.expander("🔍 ¿Cómo usar esta aplicación?"):
        st.markdown("""
        1. **Selecciona un algoritmo** en el panel lateral (K-Means o Jerárquico)
        2. **Ajusta el número de clusters (k)** usando el slider
        3. **Presiona 'Ejecutar Análisis'** para ver los resultados
        4. **Analiza las métricas** de Silhouette y Davies-Bouldin para evaluar calidad
        5. **Visualiza los clusters** en el espacio PCA
        6. **Interpreta los perfiles clínicos** de cada grupo
        7. **Valida con el diagnóstico real** para verificar precisión
        
        💡 **Recomendación:** Comienza con k=2 o k=3 para ver la estructura básica del dataset.
        """)
    
    st.markdown("---")
    st.markdown("**Desarrollado por:** Data Science Team | **Dataset:** UCI Machine Learning Repository | **Framework:** Streamlit + Docker")