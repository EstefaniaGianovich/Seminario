# Trabajo Práctico: Problema de Ruteo de Vehículos con Capacidad (CVRP)
## Algoritmos: Fuerza Bruta vs. Optimización por Colonia de Hormigas (ACO)

Este repositorio contiene la solución completa para el Trabajo Práctico de **Ruteo de Vehículos con Capacidad (Capacitated Vehicle Routing Problem - CVRP)**, desarrollada en Python 3 puro con soporte gráfico en `matplotlib`.

---

## 1. Estructura del Proyecto

```text
Colonia de hormigas/
├── app_interactiva.py           # Interfaz gráfica interactiva web local (Streamlit)
├── cvrp_solver.py               # Script con algoritmos, benchmarks y consola interactiva
├── cvrp_n5_fuerza_bruta.png     # Gráfico de rutas para N = 5 (Fuerza Bruta)
├── cvrp_n5_aco.png              # Gráfico de rutas para N = 5 (ACO)
├── cvrp_n10_fuerza_bruta.png    # Gráfico de rutas para N = 10 (Fuerza Bruta)
├── cvrp_n10_aco.png             # Gráfico de rutas para N = 10 (ACO)
├── cvrp_n20_aco.png             # Gráfico de rutas para N = 20 (ACO)
└── README.md                    # Documentación del proyecto
```

---

## 2. Requisitos e Instalación

- Python 3.10 o superior (probado en Python 3.14).
- Librerías: `streamlit`, `matplotlib`, `numpy`, `pandas`.

Instalación rápida:
```bash
pip install streamlit matplotlib numpy pandas
```

---

## 3. Modos de Ejecución

### Opción A: Interfaz Gráfica Interactiva (Streamlit)
Ejecuta la aplicación web local completa en el navegador con controles deslizantes, mapa panorámico central, tabla de Gap (%), botones de simulación independientes y modal en lenguaje 100% natural:
```bash
streamlit run app_interactiva.py
```
*(o alternativamente: `python -m streamlit run app_interactiva.py`)*

- **Botón "📖 Descripción del problema":** Abre un modal interactivo en lenguaje 100% natural y conversacional (sin fórmulas complejas ni notación matemática) que explica la logística de distribución, las restricciones y por qué las hormigas superan a la fuerza bruta.
- **Botones Independientes sobre el Mapa:**
  - *`🐜 Simular Algoritmo de Hormigas`:* Ejecuta la animación con **6 hormigas simultáneas**, donde 4 recorren la ruta óptima verde y 2 divergen por caminos secundarios grises evaporados para luego reincorporarse al circuito principal.
  - *`🚚 Simular Despacho de Camiones`:* Ejecuta a demanda el despacho fluido de la flota con un color diferenciado por cada vehículo.
- **Métricas Panorámicas:** Las tarjetas de métricas, la tabla comparativa con Gap (%) y el detalle de carga se ubican de forma limpia y amplia **debajo del mapa**.
- **Gráficos Integrados:** Emplea únicamente caracteres Unicode nativos (`'🐜'` y `'🚚'`) sin depender de archivos de imagen externos.




### Opción B: Batería de Pruebas Automáticas (Consola)
Ejecuta automáticamente las instancias $N = 5$, $N = 10$ y $N = 20$, cronometra tiempos de CPU y genera los gráficos PNG:
```bash
python cvrp_solver.py
```

### Opción C: Menú de Consola Interactivo
```bash
python cvrp_solver.py --menu
```
