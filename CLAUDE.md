# CLAUDE.md - Contexto y Reglas de SmartCret

## 1. Visión y Propósito
SmartCret es una plataforma digital de Machine Learning y optimización multiobjetivo para el diseño de mezclas de concreto ecoeficientes. Su meta es predecir resistencia a compresión (MPa) y trabajabilidad (slump en mm) con intervalos de incertidumbre, minimizando simultáneamente costo económico y huella de carbono (kg CO₂/m³).

## 2. Reglas Fundamentales de Ingeniería y Código
1. **Contrato de Datos Primero:**
   - Todo código lee nombres de columnas, unidades y rangos desde `configs/schema.yaml`.
   - Nunca hardcodear nombres de columnas o unidades en el código sin referenciar el esquema.
2. **Separación Estricta de Datos:**
   - Los datos sintéticos se etiquetan explícitamente con `fuente = 'sintetico'`.
   - Nunca mezclar datos sintéticos con datos reales sin trazabilidad.
   - La partición de test reservado (`test_reserved`) se mantiene intacta para evaluación final sin fuga de datos.
3. **Parámetros en Archivos YAML:**
   - Restricciones normativas (NTP 339, ACI 318) residen en `configs/norms.yaml`.
   - Precios, factores de emisión y disponibilidad residen en `configs/materials.yaml`.
   - Hiperparámetros y modelos residen en `configs/model.yaml`.
4. **Reproducibilidad:**
   - Fijar semillas aleatorias (`random_state=42`) en generadores, splits, optimizadores y modelos.
5. **Incertidumbre y Honestidad en ML:**
   - Siempre reportar intervalos de predicción (ej. 90% de confianza) junto a las estimaciones puntuales.
   - En el optimizador, aplicar margen de seguridad derivado de la incertidumbre para asegurar cumplimiento de $f'_c$.
   - Marcar cualquier mezcla propuesta que extrapole el rango observado en el conjunto de entrenamiento.
6. **Asistente Agéntico:**
   - El asistente no inventa cifras. Debe invocar herramientas (`predict`, `optimize`, `simulate`, `carbon_budget`) y reportar exactamente los resultados calculados.
