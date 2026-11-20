# SMARTCRET

**Diseño y optimización de mezclas de concreto ecoeficientes mediante Machine Learning.**
Mezclas inteligentes que equilibran desempeño, costo y menor impacto ambiental.

Plataforma digital que, con Machine Learning y optimización sobre datos abiertos y experimentales de la empresa, estima la **resistencia** y la **trabajabilidad** de mezclas con cementos altamente adicionados y propone **dosificaciones por m³** que equilibran costo, huella de carbono y requisitos técnicos, normativos y de durabilidad.

---

## 1. Alcance del producto

| Componente | Descripción |
|---|---|
| Motor de predicción | Modelos de ML que estiman resistencia a compresión (MPa) y trabajabilidad (slump) con intervalos de incertidumbre |
| Optimizador | Optimización multiobjetivo (costo, CO₂) sujeta a resistencia, trabajabilidad, normas y disponibilidad |
| Simulador de imprevistos | Re-optimiza ante cambios de precios o disponibilidad de materiales |
| Presupuesto de carbono | Control de emisiones por obra frente a una meta, con alertas |
| Comparador de escenarios | Escenarios guardados con fichas resumen exportables |
| Asistente agéntico de IA | LLM con herramientas que interpreta resultados considerando la incertidumbre |

### Metas preliminares (a validar con casos de prueba)

- Reducir 40 % el tiempo de diseño de una mezcla.
- Disminuir 30 % las iteraciones de ensayo en laboratorio.
- Bajar 5-10 % el costo por m³ frente a la mezcla original.
- Reducir 10-15 % las emisiones de CO₂ por m³.
- Error de predicción de resistencia < 10 % en datos reservados.
- Alertar el 100 % de las superaciones de presupuesto de costo o carbono.

> Son metas estimadas: no se reportan como logros hasta medirlas con los casos de estudio.

---

## 2. Estrategia: construir contra un "contrato de datos"

Los datos reales los entregarán los compañeros de Ingeniería Civil más adelante. Por eso:

1. Todo el código lee los nombres de columnas, unidades y rangos desde `configs/schema.yaml`.
2. Se desarrolla y prueba todo el pipeline con **datos sintéticos** (claramente marcados).
3. Cuando lleguen los datos reales, solo se ajusta el YAML, se colocan los archivos en `data/raw/` y se ejecuta `python train.py`.

### Variables típicas (a confirmar con Ingeniería Civil)

- **Entradas:** tipo de cemento, % de adición (ceniza, escoria, puzolana), cemento, agua, relación a/c, agregado fino, agregado grueso, aditivo, edad de curado.
- **Salidas:** resistencia a compresión (MPa), slump (mm).
- **Para el optimizador:** costo por material (S/ por kg), factor de emisión (kg CO₂/kg), disponibilidad.

---

## 3. Estructura del repositorio

```
smartcret/
├── README.md
├── CLAUDE.md                 # contexto y reglas para el agente de código
├── pyproject.toml
├── configs/
│   ├── schema.yaml           # contrato de datos
│   ├── norms.yaml            # restricciones normativas (editable por Ing. Civil)
│   ├── materials.yaml        # costos, factores de emisión, disponibilidad
│   └── model.yaml            # hiperparámetros y modelos a comparar
├── data/
│   ├── raw/                  # datos reales entregados (no se versionan)
│   ├── processed/
│   └── synthetic/            # datos sintéticos para pruebas
├── src/smartcret/
│   ├── data/                 # carga, validación, split
│   ├── models/               # entrenamiento, incertidumbre
│   ├── optimizer/
│   ├── simulator/
│   ├── assistant/
│   └── app/                  # interfaz web (Streamlit)
├── tests/
├── notebooks/                # EDA cuando lleguen los datos
├── models/                   # modelos entrenados versionados (v1, v2...)
├── train.py
└── evaluate.py
```

---

## 4. Plan por fases

Cada fase tiene un **criterio de aceptación**. No se avanza a la siguiente hasta cumplirlo.

### Fase 0: Preparación (sin datos)

- Repositorio con la estructura anterior y entorno reproducible (Python 3.11+).
- `CLAUDE.md` con el contexto del proyecto, las reglas y el contrato de datos.
- **Aceptación:** `pip install -e .` funciona y `pytest` corre (aunque sea con una prueba vacía).

### Fase 1: Datos (listo para recibirlos)

1. `schema.yaml`: columnas, unidades, tipos, rangos válidos.
2. Módulo de carga y validación: lee CSV/Excel, normaliza nombres y unidades, reporta filas inválidas, duplicados y faltantes.
3. Generador de datos sintéticos basado en relaciones físicas simples (por ejemplo, a menor a/c, mayor resistencia). Se marcan con una columna `fuente = sintetico`.
4. Separación train / validación / **test reservado**, con semilla fija y sin fuga de datos. Columna `fuente` para evaluar por separado datos abiertos y de laboratorio.

**Aceptación:** `python -m smartcret.data.generate_synthetic` crea 500 filas, y el cargador las valida y las divide correctamente.

### Fase 2: Modelo (listo para entrenar)

1. Script único: `python train.py --data data/raw/ --config configs/model.yaml`.
2. Modelos a comparar: regresión regularizada (línea base), Random Forest, Gradient Boosting (XGBoost / LightGBM / CatBoost).
3. Validación cruzada y ajuste de hiperparámetros (Optuna).
4. Métricas: RMSE, MAE, R² y error porcentual (meta: < 10 % en datos reservados).
5. **Incertidumbre:** intervalos de predicción (quantile regression o conformal prediction).
6. Registro de experimentos y guardado versionado en `models/vN/` con metadatos (datos usados, métricas, fecha).

**Aceptación:** `python train.py` corre de punta a punta con datos sintéticos y imprime las métricas y los intervalos.

### Fase 3: Optimizador

1. Objetivos: minimizar costo y CO₂.
2. Restricciones: resistencia mínima (con margen según incertidumbre), slump objetivo, límites normativos (a/c máximo, cemento mínimo, etc.) leídos de `norms.yaml`.
3. Algoritmo: NSGA-II (`pymoo`) para generar un frente de Pareto.
4. La búsqueda se limita al rango observado en el entrenamiento; las propuestas fuera de rango se marcan.

**Aceptación:** dado un requisito de obra (por ejemplo, f'c y slump), devuelve un conjunto de mezclas por m³ que cumplen las restricciones, con costo y CO₂ calculados.

### Fase 4: Funciones de la plataforma

- Simulador de imprevistos: cambia precios o disponibilidad y re-optimiza.
- Presupuesto de carbono por obra, con alertas al superar costo o CO₂.
- Comparador de escenarios con fichas resumen exportables (PDF).
- Persistencia de escenarios en SQLite (migrable a Postgres / Supabase).

**Aceptación:** se puede guardar, recargar y comparar al menos dos escenarios, y las alertas se disparan en los casos de prueba.

### Fase 5: Asistente agéntico

- LLM con *tool calling* que invoca el predictor, el optimizador y el simulador.
- **Regla:** el asistente no inventa números; solo reporta lo que devuelven las herramientas e indica la incertidumbre.

**Aceptación:** responde "¿qué mezcla me conviene si sube el precio del cemento?" llamando a las herramientas y explicando el resultado.

### Fase 6: Interfaz y demostración

- Interfaz web (Streamlit) con: ingreso de requisitos, mezcla propuesta vs. original, escenarios y asistente.
- Caso de demostración guionizado y evaluación con datos reservados.

**Aceptación:** la demo completa corre en una laptop con internet en menos de 5 minutos.

---

## 5. Instrucciones para el agente de código

Trabajar **fase por fase**, no con un solo prompt gigante. Ejemplo para la Fase 1:

> Implementa la Fase 1 según el README. Criterio de aceptación: `pytest` pasa; `python -m smartcret.data.generate_synthetic` crea 500 filas sintéticas; `python train.py` corre de punta a punta con ellas. No avances a la Fase 2 hasta que esto funcione.

### Reglas para `CLAUDE.md`

- Nunca inventar datos reales ni mezclarlos con sintéticos.
- Todo parámetro de dominio (normas, factores de emisión, precios) va en YAML, no en el código.
- Cada módulo con pruebas mínimas.
- Si algo del dominio del concreto es ambiguo, dejar un `TODO` o preguntar; no suponer.
- Fijar semillas aleatorias para que los resultados sean reproducibles.
- El test reservado no se usa para ajustar hiperparámetros.

---

## 6. Checklist: lo que deben entregar los compañeros

- [ ] Un CSV o Excel por fuente, una fila por ensayo, con encabezados claros.
- [ ] Unidades explícitas (kg/m³, MPa, mm) y **edad del ensayo** (7, 28 días, etc.).
- [ ] Tipo de cemento y porcentaje de adición de cada mezcla.
- [ ] Costos actuales por material.
- [ ] Factores de emisión por material, con la fuente de cada uno.
- [ ] Normas aplicables (NTP / ACI) y límites a usar como restricciones.
- [ ] Origen de cada dato (abierto vs. laboratorio de la empresa).

---

## 7. Cuando lleguen los datos reales

1. Colocar los archivos en `data/raw/`.
2. Ajustar `configs/schema.yaml` a las columnas reales.
3. Ejecutar análisis exploratorio en `notebooks/` (distribuciones, faltantes, valores atípicos).
4. Ejecutar `python train.py --data data/raw/ --config configs/model.yaml`.
5. Revisar métricas en el test reservado con `python evaluate.py`.
6. Reentrenar y versionar el modelo (`models/v2/`).

---

## 8. Riesgos y limitaciones

- **Pocos datos:** probablemente cientos de filas, no miles. Preferir modelos simples y reportar la incertidumbre con honestidad.
- **Extrapolación:** el optimizador puede proponer mezclas fuera del rango de entrenamiento; limitar y marcar.
- **Calidad de datos:** mezclar fuentes abiertas y de laboratorio puede introducir sesgos; evaluar por separado.
- **Validación en laboratorio:** las mezclas propuestas siempre deben confirmarse con ensayos.

---

## 9. Equipo

| Integrante | Rol | Institución |
|---|---|---|
| Fabrizio Salvador Elias Perez Peralta | Desarrollo de la plataforma, ML, optimizador y asistente de IA | Universidad Privada de Tacna (Ingeniería de Sistemas) |
| Vanessa de los Angeles Olanda Ccari | Variables de mezcla, depuración de datos, normas y validación técnica | Universidad Nacional de San Agustín (Ingeniería Civil) |
| Daniel Alejandro Patiño Puma | Variables de mezcla, depuración de datos, normas y validación técnica | Universidad Nacional de San Agustín (Ingeniería Civil) |
| Israel Nazareth Chaparro Cruz | Guía metodológica y evaluación del prototipo | Universidad Privada de Tacna (docente) |

> La distribución de roles es una propuesta; conviene ajustarla a lo acordado por el equipo.