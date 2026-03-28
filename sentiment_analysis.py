# =============================================================================
#  PROYECTO INTEGRADOR - SoftComputing y Computación Afectiva
#  Análisis de Sentimientos en Redes Sociales con VADER
# =============================================================================
#
#  INSTRUCCIONES DE USO:
#  1. Instalar dependencias:
#       pip install vaderSentiment pandas matplotlib seaborn wordcloud scikit-learn
#
#  2. Descargar el dataset Sentiment140 de Kaggle:
#       https://www.kaggle.com/datasets/kazanova/sentiment140
#       Guardar el archivo como: sentiment140.csv (en la misma carpeta que este script)
#
#  3. Ejecutar:
#       python sentiment_analysis.py
#
#  NOTA: Si no tienes el dataset, el script genera datos de ejemplo automáticamente.
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import re
import os
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN GENERAL ────────────────────────────────────────────────────
MUESTRA = 5000          # Número de tweets a analizar (ajustable)
RANDOM_STATE = 42       # Para reproducibilidad
UMBRAL_POS = 0.05       # Score mínimo para clasificar como Positivo
UMBRAL_NEG = -0.05      # Score máximo para clasificar como Negativo

# Colores del proyecto
COLOR_POS = "#2E7D32"   # Verde oscuro → Positivo
COLOR_NEU = "#1565C0"   # Azul oscuro  → Neutro
COLOR_NEG = "#C62828"   # Rojo oscuro  → Negativo
PALETA = [COLOR_POS, COLOR_NEU, COLOR_NEG]

# ─── 1. FUNCIONES DE PREPROCESAMIENTO ─────────────────────────────────────────

def limpiar_texto(texto):
    """
    Limpia el texto eliminando ruido característico de redes sociales.
    Pasos: eliminar URLs → eliminar @menciones → eliminar #hashtags
           → eliminar caracteres no alfabéticos → convertir a minúsculas.
    """
    texto = str(texto)
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto)  # URLs
    texto = re.sub(r'@\w+', '', texto)                      # Menciones
    texto = re.sub(r'#\w+', '', texto)                      # Hashtags
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)               # Solo letras
    texto = re.sub(r'\s+', ' ', texto)                      # Espacios múltiples
    return texto.lower().strip()


def clasificar_sentimiento(score):
    """
    Clasifica el score compuesto de VADER en una categoría de sentimiento.

    VADER devuelve un score 'compound' en el rango [-1.0, +1.0]:
      - Valores cercanos a  1.0 → muy positivo
      - Valores cercanos a  0.0 → neutro
      - Valores cercanos a -1.0 → muy negativo
    """
    if score >= UMBRAL_POS:
        return 'Positivo'
    elif score <= UMBRAL_NEG:
        return 'Negativo'
    else:
        return 'Neutro'


# ─── 2. CARGA DE DATOS ────────────────────────────────────────────────────────

def cargar_dataset():
    """
    Carga el dataset Sentiment140 de Kaggle.
    Si no existe el archivo, genera datos de ejemplo para demostración.
    """
    archivo = 'sentiment140.csv'

    if os.path.exists(archivo):
        print(f"[OK] Cargando dataset '{archivo}'...")
        df = pd.read_csv(archivo, encoding='latin-1', header=None)
        df.columns = ['sentimiento', 'id', 'fecha', 'query', 'usuario', 'texto']
        df = df[['sentimiento', 'texto']]

        # Sentiment140 usa: 0 = negativo, 4 = positivo → convertir a etiquetas
        df['etiqueta_real'] = df['sentimiento'].map({0: 'Negativo', 4: 'Positivo'})
        df = df[['texto', 'etiqueta_real']]

        # Tomar muestra aleatoria
        df = df.sample(min(MUESTRA, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"[OK] {len(df)} tweets cargados.")

    else:
        print("[AVISO] No se encontró 'sentiment140.csv'.")
        print("[INFO] Generando datos de ejemplo para demostración...")
        df = generar_datos_ejemplo()

    return df


def generar_datos_ejemplo():
    """
    Genera un dataset de ejemplo cuando no se dispone del archivo real.
    Contiene 300 tweets sintéticos con sus etiquetas.
    """
    positivos = [
        "I love this product, absolutely amazing quality!",
        "Best purchase I've made this year, highly recommend!",
        "Great customer service and fast shipping, very happy",
        "This exceeded all my expectations, wonderful experience",
        "So happy with my order! Will definitely buy again",
        "Outstanding quality, worth every penny spent here",
        "Fantastic product, exactly as described and beautiful",
        "Really impressed with the quality and fast delivery",
        "Amazing product, my family loves it so much",
        "Perfect gift, arrived on time and well packaged",
        "Incredible service, will be back for more soon",
        "Absolutely love it! Best decision I have made",
        "Super satisfied with this purchase, very good",
        "The product works perfectly, happy customer here",
        "Excellent quality and great value for the price",
    ] * 10  # 150 positivos

    negativos = [
        "Terrible product, broke after just two days of use",
        "Very disappointed, not as described and poor quality",
        "Worst purchase ever, complete waste of my money",
        "Awful customer service, they never responded to me",
        "Product arrived damaged and packaging was destroyed",
        "Completely useless, does not work at all period",
        "Very bad quality, nothing like the photos shown",
        "Extremely disappointed with this purchase overall",
        "Horrible experience, took weeks to arrive broken",
        "Do not buy this, total scam and waste of money",
        "Poor quality materials, fell apart immediately after",
        "Disgusting service, rude staff and no refund given",
        "Broken on arrival, seller refused to help me",
        "Terrible smell and the color was completely wrong",
        "Never again, worst online shopping experience ever",
    ] * 10  # 150 negativos

    textos = positivos + negativos
    etiquetas = ['Positivo'] * 150 + ['Negativo'] * 150

    df = pd.DataFrame({'texto': textos, 'etiqueta_real': etiquetas})
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"[OK] {len(df)} registros de ejemplo generados.")
    return df


# ─── 3. ANÁLISIS DE SENTIMIENTOS CON VADER ────────────────────────────────────

def analizar_sentimientos(df):
    """
    Aplica VADER a cada tweet y genera la clasificación de sentimiento.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) es un modelo de
    Computación Afectiva basado en un lexicón con valencia emocional.
    No requiere entrenamiento: usa reglas lingüísticas y un diccionario
    pre-construido de ~7,500 palabras con sus scores de valencia.
    """
    print("\n[PROCESANDO] Limpiando textos...")
    df['texto_limpio'] = df['texto'].apply(limpiar_texto)

    print("[PROCESANDO] Aplicando análisis VADER...")
    analizador = SentimentIntensityAnalyzer()

    # polarity_scores devuelve: {'neg': x, 'neu': x, 'pos': x, 'compound': x}
    scores = df['texto_limpio'].apply(lambda x: analizador.polarity_scores(x))

    df['score_pos']      = scores.apply(lambda x: x['pos'])
    df['score_neg']      = scores.apply(lambda x: x['neg'])
    df['score_neu']      = scores.apply(lambda x: x['neu'])
    df['score_compound'] = scores.apply(lambda x: x['compound'])

    df['sentimiento_vader'] = df['score_compound'].apply(clasificar_sentimiento)

    print("[OK] Análisis completado.")
    return df


# ─── 4. PRUEBAS CON FRASES MANUALES ───────────────────────────────────────────

def prueba_frases_manuales():
    """
    Prueba el analizador con frases conocidas para validar el comportamiento.
    Esta sección documenta la Prueba 1 del informe.
    """
    analizador = SentimentIntensityAnalyzer()

    frases_prueba = [
        ("I absolutely love this product, it's amazing!",     "Positivo esperado"),
        ("This is terrible, completely useless garbage",       "Negativo esperado"),
        ("The package arrived today",                          "Neutro esperado"),
        ("Not bad, could be better but it's okay I guess",    "Neutro/leve esperado"),
        ("WORST EXPERIENCE EVER!!! Never buying again!!",     "Negativo esperado"),
    ]

    print("\n" + "="*70)
    print("PRUEBA 1: Validación manual del analizador VADER")
    print("="*70)
    print(f"{'Texto':<50} {'Compound':>9} {'Clase'}")
    print("-"*70)

    for texto, etiqueta in frases_prueba:
        scores = analizador.polarity_scores(texto)
        compound = scores['compound']
        clase = clasificar_sentimiento(compound)
        print(f"{texto[:48]:<50} {compound:>9.4f} → {clase}  ({etiqueta})")

    print("="*70)


# ─── 5. MÉTRICAS DE EVALUACIÓN ────────────────────────────────────────────────

def evaluar_modelo(df):
    """
    Evalúa la precisión del modelo VADER comparando con las etiquetas reales.
    Solo aplica si el dataset tiene etiquetas reales (Sentiment140 o ejemplo).
    Esta sección documenta la Prueba 2 del informe.
    """
    # Filtrar solo positivos y negativos (Sentiment140 no tiene neutros reales)
    df_eval = df[df['etiqueta_real'].isin(['Positivo', 'Negativo'])].copy()
    df_eval = df_eval[df_eval['sentimiento_vader'].isin(['Positivo', 'Negativo'])].copy()

    if len(df_eval) < 10:
        print("[AVISO] No hay suficientes datos para evaluación.")
        return

    print("\n" + "="*70)
    print("PRUEBA 2: Métricas de evaluación del modelo")
    print("="*70)
    print(classification_report(
        df_eval['etiqueta_real'],
        df_eval['sentimiento_vader'],
        target_names=['Negativo', 'Positivo']
    ))

    return df_eval


# ─── 6. VISUALIZACIONES ───────────────────────────────────────────────────────

def generar_visualizaciones(df, df_eval=None):
    """
    Genera cuatro visualizaciones clave para el informe:
    1. Distribución de sentimientos (barras)
    2. Distribución del score compound (histograma)
    3. Nube de palabras por categoría
    4. Matriz de confusión (si hay datos de evaluación)
    """
    os.makedirs('resultados', exist_ok=True)
    print("\n[GENERANDO] Visualizaciones...")

    # ── Figura 1: Distribución y score compound ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Análisis de Sentimientos con VADER\nSoftComputing y Computación Afectiva',
                 fontsize=14, fontweight='bold', y=1.02)

    # Gráfico 1a: Distribución de categorías
    conteo = df['sentimiento_vader'].value_counts()
    orden = ['Positivo', 'Neutro', 'Negativo']
    conteo = conteo.reindex([c for c in orden if c in conteo.index])

    bars = axes[0].bar(conteo.index, conteo.values,
                       color=[COLOR_POS if x == 'Positivo' else COLOR_NEU if x == 'Neutro' else COLOR_NEG
                              for x in conteo.index],
                       edgecolor='white', linewidth=1.5, width=0.5)

    for bar, val in zip(bars, conteo.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                     f'{val}\n({val/len(df)*100:.1f}%)',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[0].set_title('Distribución de Sentimientos', fontweight='bold', pad=12)
    axes[0].set_xlabel('Categoría de sentimiento')
    axes[0].set_ylabel('Número de tweets')
    axes[0].spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(0, conteo.max() * 1.25)

    # Gráfico 1b: Histograma del score compound
    axes[1].hist(df['score_compound'], bins=50, color='#1565C0', alpha=0.7, edgecolor='white')
    axes[1].axvline(x=UMBRAL_POS, color=COLOR_POS, linestyle='--', linewidth=1.5,
                    label=f'Umbral positivo ({UMBRAL_POS})')
    axes[1].axvline(x=UMBRAL_NEG, color=COLOR_NEG, linestyle='--', linewidth=1.5,
                    label=f'Umbral negativo ({UMBRAL_NEG})')
    axes[1].axvline(x=df['score_compound'].mean(), color='orange', linestyle='-', linewidth=2,
                    label=f'Media: {df["score_compound"].mean():.3f}')
    axes[1].set_title('Distribución del Score Compound VADER', fontweight='bold', pad=12)
    axes[1].set_xlabel('Score compound [-1, +1]')
    axes[1].set_ylabel('Frecuencia')
    axes[1].legend(fontsize=9)
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('resultados/01_distribucion_sentimientos.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[OK] Gráfico 1 guardado: resultados/01_distribucion_sentimientos.png")

    # ── Figura 2: Nubes de palabras ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Nube de Palabras por Categoría de Sentimiento',
                 fontsize=14, fontweight='bold')

    categorias = ['Positivo', 'Neutro', 'Negativo']
    colores_wc = ['Greens', 'Blues', 'Reds']

    for ax, cat, cmap_name in zip(axes, categorias, colores_wc):
        textos_cat = ' '.join(df[df['sentimiento_vader'] == cat]['texto_limpio'].tolist())

        if len(textos_cat.strip()) < 10:
            ax.text(0.5, 0.5, f'Sin datos\n{cat}', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(cat, fontweight='bold')
            ax.axis('off')
            continue

        wc = WordCloud(
            width=500, height=300,
            background_color='white',
            colormap=cmap_name,
            max_words=80,
            collocations=False
        ).generate(textos_cat)

        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(cat, fontweight='bold', fontsize=13,
                     color=COLOR_POS if cat == 'Positivo' else COLOR_NEU if cat == 'Neutro' else COLOR_NEG)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('resultados/02_nubes_palabras.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("[OK] Gráfico 2 guardado: resultados/02_nubes_palabras.png")

    # ── Figura 3: Matriz de confusión (si aplica) ──
    if df_eval is not None and len(df_eval) >= 10:
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(
            df_eval['etiqueta_real'],
            df_eval['sentimiento_vader'],
            labels=['Negativo', 'Positivo']
        )
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negativo', 'Positivo'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title('Matriz de Confusión\nVADER vs. Etiquetas Reales', fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig('resultados/03_matriz_confusion.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("[OK] Gráfico 3 guardado: resultados/03_matriz_confusion.png")


# ─── 7. ESTADÍSTICAS RESUMEN ──────────────────────────────────────────────────

def imprimir_estadisticas(df):
    """
    Imprime un resumen estadístico de los resultados del análisis.
    """
    print("\n" + "="*70)
    print("RESUMEN ESTADÍSTICO DEL ANÁLISIS")
    print("="*70)
    print(f"Total de tweets analizados:   {len(df):>6,}")
    print(f"Tweets Positivos:             {len(df[df['sentimiento_vader']=='Positivo']):>6,}  "
          f"({len(df[df['sentimiento_vader']=='Positivo'])/len(df)*100:.1f}%)")
    print(f"Tweets Neutros:               {len(df[df['sentimiento_vader']=='Neutro']):>6,}  "
          f"({len(df[df['sentimiento_vader']=='Neutro'])/len(df)*100:.1f}%)")
    print(f"Tweets Negativos:             {len(df[df['sentimiento_vader']=='Negativo']):>6,}  "
          f"({len(df[df['sentimiento_vader']=='Negativo'])/len(df)*100:.1f}%)")
    print(f"\nScore compound promedio:      {df['score_compound'].mean():>+.4f}")
    print(f"Desviación estándar:          {df['score_compound'].std():>.4f}")
    print(f"Score mínimo registrado:      {df['score_compound'].min():>+.4f}")
    print(f"Score máximo registrado:      {df['score_compound'].max():>+.4f}")
    print("="*70)

    # Muestra los 5 tweets más positivos y más negativos
    print("\nTOP 3 TWEETS MÁS POSITIVOS:")
    top_pos = df.nlargest(3, 'score_compound')[['texto', 'score_compound']]
    for _, row in top_pos.iterrows():
        print(f"  [{row['score_compound']:+.4f}] {str(row['texto'])[:80]}...")

    print("\nTOP 3 TWEETS MÁS NEGATIVOS:")
    top_neg = df.nsmallest(3, 'score_compound')[['texto', 'score_compound']]
    for _, row in top_neg.iterrows():
        print(f"  [{row['score_compound']:+.4f}] {str(row['texto'])[:80]}...")
    print("="*70)


# ─── PUNTO DE ENTRADA PRINCIPAL ───────────────────────────────────────────────

if __name__ == '__main__':
    print("="*70)
    print("  ANÁLISIS DE SENTIMIENTOS EN REDES SOCIALES CON VADER")
    print("  Proyecto Integrador — SoftComputing y Computación Afectiva")
    print("="*70)

    # Paso 1: Validación manual del analizador
    prueba_frases_manuales()

    # Paso 2: Cargar datos
    df = cargar_dataset()

    # Paso 3: Analizar sentimientos
    df = analizar_sentimientos(df)

    # Paso 4: Estadísticas resumen
    imprimir_estadisticas(df)

    # Paso 5: Evaluación con métricas formales
    df_eval = evaluar_modelo(df)

    # Paso 6: Visualizaciones
    generar_visualizaciones(df, df_eval)

    # Paso 7: Guardar resultados en CSV
    df.to_csv('resultados/resultados_analisis.csv', index=False)
    print("\n[OK] Resultados guardados en: resultados/resultados_analisis.csv")
    print("[OK] ¡Análisis completado! Revisa la carpeta 'resultados/'")
    print("="*70)
