"""
================================================================================
APLICACIÓN WEB INTERACTIVA (STREAMLIT): CVRP CON COLONIA DE HORMIGAS (ACO)
================================================================================
Ajustes aplicados:
  - Desactivación de Image.MAX_IMAGE_PIXELS para prevenir DecompressionBombError
    cuando se alinean múltiples ciudades en la misma coordenada X o Y.
  - Fijación de límites de ejes para evitar colapsos visuales de Matplotlib.
  - Sin emoticones en la ventana modal 'Descripción del Problema'.
================================================================================
"""

import math
import os
import time
from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import streamlit as st

# Desactivación del límite de píxeles de Pillow para evitar DecompressionBombError
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from cvrp_solver import (
    CVRPInstance,
    CVRPSolution,
    solve_cvrp_brute_force,
    AntColonyCVRP
)

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN DE PÁGINA STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CVRP Solver | Colonia de Hormigas",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: -15px;
        margin-bottom: 2px;
    }
    .sub-title {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 12px;
    }
    .route-card-compact {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

ANT_PNG_PATH = "hormiga.png"

TRUCK_COLORS = [
    '#2563EB', '#DC2626', '#EAB308', '#9333EA', '#06B6D4',
    '#F97316', '#10B981', '#EC4899', '#6366F1', '#14B8A6'
]
COLOR_OPTIMAL_ACO = '#10B981'
COLOR_EVAPORATED  = '#D1D5DB'


# ------------------------------------------------------------------------------
# 2. MODAL DESCRIPCIÓN DEL PROBLEMA (SIN EMOTICONES)
# ------------------------------------------------------------------------------
@st.dialog("Descripción del Problema y Guía del TP", width="large")
def show_problem_description_modal():
    t_tab1, t_tab2 = st.tabs([
        "Consigna, Reglas y Operación de la Flota",
        "Formulación Matemática y Lógica del Algoritmo ACO"
    ])

    with t_tab1:
        st.markdown("""
        ### El Desafío Logístico de Distribución

        Una empresa de logística debe repartir pedidos a un conjunto de clientes o ciudades distribuidas geográficamente, partiendo desde un único **Depósito Central** (identificado como Nodo 0).

        Este problema pertenece a la familia de modelos conocidos como **CVRP** (*Capacitated Vehicle Routing Problem* o Problema de Ruteo de Vehículos con Capacidad).

        ---

        ### Reglas Operativas del Reparto

        1. **Depósito Central Único (Nodo 0):** Cada ruta debe iniciar obligatoriamente en el depósito central y finalizar en el mismo punto tras completar las entregas.
        2. **Capacidad Limitada por Vehículo ($K$ paquetes):** La suma de las demandas de las ciudades asignadas a un mismo camión no puede exceder la capacidad máxima de su bodega ($K$).
        3. **Visita Única por Cliente:** Cada ciudad cliente debe ser visitada exactamente una sola vez por un único vehículo.
        4. **Continuidad de Ruta:** Todo vehículo que ingresa a un nodo cliente debe salir de él hacia el siguiente destino o de regreso al depósito.
        5. **Objetivo Global:** Minimizar la distancia total combinada recorrida por la totalidad de la flota utilizada.

        ---

        ### Planificación y Salida Simultánea

        El modelo asume una operación planificada en paralelo: todos los vehículos asignados salen simultáneamente al inicio de la jornada laboral para realizar sus itinerarios independientes sin traslapes.
        """)

    with t_tab2:
        st.markdown("""
        ### Formulación Matemática y Lógica del Algoritmo

        #### Significado de CVRP
        **CVRP** es la sigla en inglés de *Capacitated Vehicle Routing Problem*. Es un problema de optimización combinatoria de clase NP-duro, lo que significa que el tiempo requerido para encontrar la solución exacta mediante inspección exhaustiva (Fuerza Bruta) crece exponencialmente ($O(N!)$), volviéndose incomputable para valores de $N \\ge 15$.

        ---

        #### Formulación Formal del Modelo

        Dado un grafo completo $G = (V, E)$ donde $V = \\{0, 1, \\dots, N\\}$ (siendo 0 el Depósito Central) y $d_{ij}$ es la matriz de distancias euclidianas entre nodos:

        **Función Objetivo (Minimización de Distancia Total):**

        $$\\min Z = \\sum_{k=1}^{M} \\sum_{j=1}^{|R_k|-1} d_{R_k[j], R_k[j+1]}$$

        Donde $R_k$ representa la secuencia de nodos visitados por el vehículo $k$.

        **Restricciones:**
        - Inicio y fin en Depósito: $v_{k,0} = 0$ y $v_{k,m_k} = 0, \\quad \\forall k \\in \\{1, \\dots, M\\}$
        - Respeto de Capacidad: $\\sum_{i \\in R_k} d_i \\le K, \\quad \\forall k \\in \\{1, \\dots, M\\}$
        - Cobertura Completa sin Duplicados: $R_a \\cap R_b = \\{0\\}, \\quad \\forall a \\neq b$

        ---

        #### Lógica del Algoritmo de Colonia de Hormigas (ACO)

        El algoritmo de Optimización basada en Colonia de Hormigas (ACO) es una metaheurística inspirada en el comportamiento colectivo de las hormigas biológicas para encontrar caminos óptimos entre su nido y las fuentes de alimento.

        1. **Regla de Transición Probabilística:**
           Una hormiga $k$ ubicada en la ciudad $i$ elige desplazarse a la ciudad $j$ no visitada según la probabilidad:

           $$P_{ij}^k = \\frac{[\\tau_{ij}]^\\alpha \\cdot [\\eta_{ij}]^\\beta}{\\sum_{l \\in N_k} [\\tau_{il}]^\\alpha \\cdot [\\eta_{il}]^\\beta}$$

           - $\\tau_{ij}$: Intensidad de **feromona** acumulada en el tramo $(i, j)$.
           - $\\eta_{ij} = \\frac{1}{d_{ij}}$: **Visibilidad** o deseabilidad heurística (inverso de la distancia).
           - $\\alpha$: Peso relativo de la memoria colectiva (feromona).
           - $\\beta$: Peso relativo de la codicia local (distancia corta).

        2. **Evaporación de Feromona:**
           Para evitar la convergencia prematura en soluciones subóptimas, la feromona se evapora gradualmente tras cada ciclo:

           $$\\tau_{ij} \\leftarrow (1 - \\rho) \\cdot \\tau_{ij}$$

           Donde $\\rho \\in (0, 1]$ es la tasa de evaporación.

        3. **Depósito de Feromona (Refuerzo Positivo):**
           Las hormigas depositan feromona en las aristas recorridas proporcionalmente a la calidad de su solución (cuanto menor es la distancia total del viaje, mayor cantidad de feromona deposita).
        """)


# ------------------------------------------------------------------------------
# 3. HELPERS DE DIBUJO
# ------------------------------------------------------------------------------
def draw_ant_marker(ax, x: float, y: float, zoom: float = 0.045):
    if os.path.exists(ANT_PNG_PATH):
        try:
            arr_img = plt.imread(ANT_PNG_PATH)
            imagebox = OffsetImage(arr_img, zoom=zoom)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False, zorder=10)
            ax.add_artist(ab)
            return
        except Exception:
            pass
    ax.scatter(x, y, s=120, color='#D97706', edgecolors='black', zorder=10)


def draw_nodes_base(ax, instance: CVRPInstance):
    depot_x, depot_y = instance.coords[0]
    ax.scatter(depot_x, depot_y, c='#FBBF24', s=380, marker='*', edgecolors='black', linewidth=1.5, zorder=8)
    ax.text(depot_x, depot_y - 4.5, "DEPÓSITO", fontsize=8.0, fontweight='bold', ha='center', va='top', color='#92400E', zorder=9)

    for city_id in range(1, instance.num_cities + 1):
        x, y = instance.coords[city_id]
        ax.scatter(x, y, c='white', s=160, edgecolors='black', linewidth=1.4, zorder=7)
        ax.text(x, y, f"{city_id}", fontsize=8.0, ha='center', va='center', fontweight='bold', color='black', zorder=8)
        ax.text(x, y + 3.2, f"({instance.demands[city_id]}p)", fontsize=7.0, ha='center', va='bottom', color='#4B5563', fontweight='bold', zorder=8)

    # Fijar rangos fijos de ejes para prevenir que la figura colapse
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_xlabel("X (km)", fontsize=8.5)
    ax.set_ylabel("Y (km)", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.35)


def render_solution_map_compact(instance: CVRPInstance, solution: Optional[CVRPSolution]) -> plt.Figure:
    plt.close('all')
    fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=80)

    if solution and solution.routes:
        for r_idx, route in enumerate(solution.routes):
            color = TRUCK_COLORS[r_idx % len(TRUCK_COLORS)]
            xs = [instance.coords[node][0] for node in route]
            ys = [instance.coords[node][1] for node in route]

            ax.plot(xs, ys, color=color, linewidth=2.2, alpha=0.9, zorder=4)

            for k in range(len(route) - 1):
                u, v = route[k], route[k + 1]
                x_mid = (instance.coords[u][0] + instance.coords[v][0]) / 2.0
                y_mid = (instance.coords[u][1] + instance.coords[v][1]) / 2.0
                dx = (instance.coords[v][0] - instance.coords[u][0]) * 0.08
                dy = (instance.coords[v][1] - instance.coords[u][1]) * 0.08
                ax.annotate('', xy=(x_mid + dx, y_mid + dy), xytext=(x_mid, y_mid),
                            arrowprops=dict(arrowstyle="->", color=color, lw=1.3, alpha=0.95), zorder=5)

    draw_nodes_base(ax, instance)
    
    title_str = f"Mapa de la Solución ({solution.algorithm_name if solution else 'Estático'})"
    if solution:
        title_str += f"\nDistancia Total: {solution.total_distance:.2f} km"
    
    ax.set_title(title_str, fontsize=9.5, fontweight='bold', pad=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------
# 4. BARRA LATERAL (SIDEBAR) & PARÁMETROS
# ------------------------------------------------------------------------------
st.sidebar.title("Parámetros del Sistema")

if st.sidebar.button("Descripción del problema", use_container_width=True):
    show_problem_description_modal()

st.sidebar.markdown("---")

max_vehicles_available = st.sidebar.slider("Flota Máxima de Vehículos", 0, 10, 5, 1)
num_cities = st.sidebar.slider("Cantidad de Ciudades (N)", 0, 50, 0, 1)
vehicle_capacity = st.sidebar.slider("Capacidad por Camión (K paquetes)", 8, 25, 12, 1)
seed_val = st.sidebar.number_input("Semilla Aleatoria (Seed)", value=42, step=1)

st.sidebar.subheader("Parámetros ACO")
num_ants = st.sidebar.slider("Número de Hormigas", 10, 100, 30, 5)
num_iterations = st.sidebar.slider("Iteraciones (Ciclos)", 20, 150, 60, 10)

with st.sidebar.expander("Opciones Avanzadas ACO"):
    alpha = st.slider("Alpha (Feromona)", 0.0, 3.0, 1.0, 0.1)
    beta = st.slider("Beta (Visibilidad)", 1.0, 5.0, 2.5, 0.1)
    rho = st.slider("Rho (Evaporación)", 0.01, 0.50, 0.10, 0.01)

algorithm_choice = st.sidebar.radio(
    "Modo de resolución:",
    ["Comparar Ambos", "Colonia de Hormigas (ACO)", "Fuerza Bruta"]
)

# Encabezado Principal
st.markdown('<div class="main-title">Optimización de Ruteo de Vehículos con Capacidad (CVRP)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Plataforma interactiva de resolución y simulación estocástica</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 5. LÓGICA DE EJECUCIÓN CONDICIONAL Y VALIDACIONES
# ------------------------------------------------------------------------------

if num_cities == 0:
    st.info("Seleccioná una Cantidad de Ciudades (N > 0) en el panel izquierdo para comenzar a trabajar.")

else:
    if "last_params" not in st.session_state or st.session_state.last_params != (num_cities, seed_val):
        st.session_state.last_params = (num_cities, seed_val)
        gen_inst = CVRPInstance.generate(num_cities=num_cities, capacity=vehicle_capacity, seed=seed_val)
        
        rows = [{"Nodo": "Depósito (0)", "X (km)": gen_inst.coords[0][0], "Y (km)": gen_inst.coords[0][1], "Demanda (paq)": 0}]
        for c_id in range(1, num_cities + 1):
            rows.append({
                "Nodo": f"Ciudad {c_id}",
                "X (km)": gen_inst.coords[c_id][0],
                "Y (km)": gen_inst.coords[c_id][1],
                "Demanda (paq)": gen_inst.demands[c_id]
            })
        st.session_state.nodes_df = pd.DataFrame(rows)

    with st.expander("Posicionamiento Manual de Ciudades y Demandas (Editar coordenadas)", expanded=False):
        st.caption("Podés cambiar los valores de X, Y o las Demandas. Se actualizará automáticamente el mapa y la solución.")
        edited_df = st.data_editor(
            st.session_state.nodes_df,
            num_rows="fixed",
            use_container_width=True,
            key="city_coords_editor"
        )
        st.session_state.nodes_df = edited_df

    coords_list = [(float(row["X (km)"]), float(row["Y (km)"])) for _, row in edited_df.iterrows()]
    demands_list = [int(row["Demanda (paq)"]) for _, row in edited_df.iterrows()]

    instance = CVRPInstance(
        num_cities=len(coords_list) - 1,
        capacity=vehicle_capacity,
        coords=coords_list,
        demands=demands_list
    )

    total_demand = sum(instance.demands)
    min_trucks_required = math.ceil(total_demand / instance.capacity) if instance.capacity > 0 else 0
    fleet_is_valid = (max_vehicles_available >= min_trucks_required) and (max_vehicles_available > 0)

    if not fleet_is_valid:
        st.error(
            f"Flota insuficiente: La demanda total ({total_demand} paq.) requiere como mínimo "
            f"{min_trucks_required} camiones de {instance.capacity} paq. Disponibles: {max_vehicles_available}."
        )

    else:
        fb_solution: Optional[CVRPSolution] = None
        aco_solution: Optional[CVRPSolution] = None
        fb_infeasible = (instance.num_cities >= 15 and algorithm_choice in ["Comparar Ambos", "Fuerza Bruta"])

        if algorithm_choice in ["Comparar Ambos", "Fuerza Bruta"] and not fb_infeasible:
            fb_solution = solve_cvrp_brute_force(instance, timeout_seconds=5.0)

        if algorithm_choice in ["Comparar Ambos", "Colonia de Hormigas (ACO)"]:
            aco_solver = AntColonyCVRP(
                instance=instance, num_ants=num_ants, num_iterations=num_iterations,
                alpha=alpha, beta=beta, rho=rho, q=100.0, seed=seed_val
            )
            aco_solution = aco_solver.solve()

        sol_active = aco_solution if aco_solution and aco_solution.routes else fb_solution

        trigger_static = False
        trigger_trucks = False
        trigger_ants = False

        if algorithm_choice == "Fuerza Bruta" and fb_infeasible:
            st.error(
                f"Fuerza Bruta Incomputable (N = {instance.num_cities}): La cantidad de permutaciones "
                f"(O(N!) = {math.factorial(instance.num_cities):,} combinaciones) requeriría un tiempo inalcanzable."
            )
        elif algorithm_choice == "Fuerza Bruta":
            st.info("Mostrando la solución estática exacta obtenida por Fuerza Bruta.")
            trigger_static = True
        else:
            if instance.num_cities <= 10:
                st.markdown("#### Selección de Modo de Visualización")
                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    trigger_static = st.button("Ver Mapa de la Solución", use_container_width=True)
                with btn_col2:
                    trigger_trucks = st.button("Simular Despacho de Camiones", use_container_width=True)
                with btn_col3:
                    trigger_ants = st.button("Simular Algoritmo de Hormigas", use_container_width=True)
            else:
                st.info(
                    "Simulaciones paso a paso no disponibles para más de 10 ciudades "
                    "debido al costo de procesamiento. Se despliega directamente la solución convergida de Colonia de Hormigas."
                )
                trigger_static = True

        # ----------------------------------------------------------------------
        # 6. LAYOUT PRINCIPAL: MAPA Y PANEL DERECHO
        # ----------------------------------------------------------------------
        if sol_active and sol_active.routes:
            col_map_area, col_trucks_area = st.columns([65, 35], gap="medium")
        else:
            col_map_area = st.container()
            col_trucks_area = None

        with col_map_area:
            map_placeholder = st.empty()

        if col_trucks_area is not None:
            with col_trucks_area:
                st.markdown("#### Flota Asignada")
                if sol_active and sol_active.routes:
                    for idx, route in enumerate(sol_active.routes):
                        load = sum(instance.demands[c] for c in route)
                        route_km = sum(instance.dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
                        color = TRUCK_COLORS[idx % len(TRUCK_COLORS)]

                        st.markdown(
                            f"""<div class="route-card-compact" style="border-left: 6px solid {color};">
                                <b style="color:{color}; font-size:1.0rem;">Camión #{idx + 1}</b><br/>
                                <b>Carga:</b> {load}/{instance.capacity} paquetes<br/>
                                <b>Distancia:</b> {route_km:.2f} km
                            </div>""",
                            unsafe_allow_html=True
                        )

        # ----------------------------------------------------------------------
        # 7. EJECUCIÓN DE ANIMACIONES Y MAPAS
        # ----------------------------------------------------------------------
        if trigger_ants and instance.num_cities <= 10 and sol_active and sol_active.routes:
            opt_edges = set()
            for r in sol_active.routes:
                for idx in range(len(r) - 1):
                    opt_edges.add((min(r[idx], r[idx + 1]), max(r[idx], r[idx + 1])))

            ant_routes = []
            for i in range(6):
                base_r = sol_active.routes[i % len(sol_active.routes)]
                legs = [(base_r[j], base_r[j + 1]) for j in range(len(base_r) - 1)]
                if i in [4, 5] and len(legs) >= 2:
                    u, v = legs[0]
                    alt = ((v + i) % instance.num_cities) + 1
                    legs[0] = (u, alt)
                    legs.insert(1, (alt, v))
                ant_routes.append(legs + legs)

            max_legs = max(len(r) for r in ant_routes)
            n_substeps = 10

            for leg_idx in range(max_legs):
                for s in range(n_substeps):
                    t = s / n_substeps
                    ant_coords = []
                    for r in ant_routes:
                        curr_leg = r[min(leg_idx, len(r) - 1)]
                        p1 = instance.coords[curr_leg[0]]
                        p2 = instance.coords[curr_leg[1]]
                        cx = p1[0] + t * (p2[0] - p1[0])
                        cy = p1[1] + t * (p2[1] - p1[1])
                        ant_coords.append((cx, cy))

                    plt.close('all')
                    fig_h, ax_h = plt.subplots(figsize=(6.5, 4.8), dpi=80)
                    n_nodes = instance.num_cities + 1
                    for i in range(n_nodes):
                        for j in range(i + 1, n_nodes):
                            if (i, j) not in opt_edges:
                                ax_h.plot([instance.coords[i][0], instance.coords[j][0]], [instance.coords[i][1], instance.coords[j][1]], color=COLOR_EVAPORATED, alpha=0.40, linewidth=0.8, zorder=2)

                    for (i, j) in opt_edges:
                        ax_h.plot([instance.coords[i][0], instance.coords[j][0]], [instance.coords[i][1], instance.coords[j][1]], color=COLOR_OPTIMAL_ACO, linewidth=2.4, alpha=0.90, zorder=4)

                    for hx, hy in ant_coords:
                        draw_ant_marker(ax_h, hx, hy, zoom=0.038)

                    draw_nodes_base(ax_h, instance)
                    ax_h.set_title(f"Simulación simplificada de exploración estocástica y depósito de feromonas\nVerde: Ruta Óptima ({sol_active.total_distance:.2f} km) | Gris: Caminos Evaporados", fontsize=8.5, fontweight='bold', pad=8)
                    fig_h.tight_layout()
                    map_placeholder.pyplot(fig_h)
                    plt.close(fig_h)
                    time.sleep(0.008)

        elif trigger_trucks and instance.num_cities <= 10 and sol_active and sol_active.routes:
            routes = sol_active.routes
            num_trucks = len(routes)
            truck_legs = [[(r[i], r[i + 1]) for i in range(len(r) - 1)] for r in routes]
            max_legs = max(len(l) for l in truck_legs)
            n_substeps_truck = 12

            for leg_idx in range(max_legs):
                for s in range(n_substeps_truck):
                    t = s / n_substeps_truck
                    plt.close('all')
                    fig_t, ax_t = plt.subplots(figsize=(6.5, 4.8), dpi=80)

                    for r_idx in range(num_trucks):
                        t_color = TRUCK_COLORS[r_idx % len(TRUCK_COLORS)]
                        legs = truck_legs[r_idx]

                        for k in range(min(leg_idx, len(legs))):
                            u_k, v_k = legs[k]
                            ax_t.plot([instance.coords[i][0] for i in (u_k, v_k)], [instance.coords[i][1] for i in (u_k, v_k)], color=t_color, linewidth=2.2, alpha=0.9, zorder=4)

                        if leg_idx < len(legs):
                            u_curr, v_curr = legs[leg_idx]
                            p1 = instance.coords[u_curr]
                            p2 = instance.coords[v_curr]
                            tx = p1[0] + t * (p2[0] - p1[0])
                            ty = p1[1] + t * (p2[1] - p1[1])

                            ax_t.plot([p1[0], tx], [p1[1], ty], color=t_color, linewidth=2.4, alpha=0.95, zorder=5)
                            ax_t.scatter(tx, ty, s=140, color=t_color, edgecolors='black', linewidth=1.2, zorder=9)
                            ax_t.text(tx, ty - 3.2, "🚚", fontsize=12, ha='center', va='top', zorder=10)

                    draw_nodes_base(ax_t, instance)
                    ax_t.set_title(f"Despacho Simultáneo de la Flota ({num_trucks} Camiones en Paralelo)\nAvanzando en simultáneo desde el Depósito...", fontsize=9.0, fontweight='bold', pad=8)
                    fig_t.tight_layout()
                    map_placeholder.pyplot(fig_t)
                    plt.close(fig_t)
                    time.sleep(0.008)

            fig_final = render_solution_map_compact(instance, sol_active)
            map_placeholder.pyplot(fig_final)
            plt.close(fig_final)

        else:
            if sol_active and sol_active.routes:
                fig_static = render_solution_map_compact(instance, sol_active)
                map_placeholder.pyplot(fig_static)
                plt.close(fig_static)
            elif not fb_infeasible:
                fig_empty = render_solution_map_compact(instance, None)
                map_placeholder.pyplot(fig_empty)
                plt.close(fig_empty)

        # ----------------------------------------------------------------------
        # 8. MÉTRICAS Y TABLA
        # ----------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### Indicadores Clave de Desempeño (KPIs)")

        if sol_active and sol_active.routes:
            loads = [sum(instance.demands[c] for c in r) for r in sol_active.routes]
            avg_efficiency = (sum(loads) / (len(loads) * instance.capacity)) * 100.0
        else:
            avg_efficiency = 0.0

        m1, m2, m3 = st.columns(3)
        with m1:
            dist_val = f"{sol_active.total_distance:.2f} km" if sol_active and sol_active.total_distance < float('inf') else "Incomputable"
            st.metric("Distancia Total Recorrida", dist_val)
        with m2:
            time_val = f"{sol_active.execution_time:.4f} s" if sol_active else "0.0 s"
            st.metric("Tiempo de Cómputo CPU", time_val)
        with m3:
            eff_val = f"{avg_efficiency:.1f}%" if sol_active else "N/A"
            st.metric("Eficiencia de Carga Promedio", eff_val)

        st.markdown("#### Tabla Comparativa de Rendimiento")

        table_rows = []
        if fb_solution:
            fb_dist_str = f"{fb_solution.total_distance:.2f} km" if fb_solution.total_distance < float('inf') else "Incomputable"
            fb_time_str = f"{fb_solution.execution_time:.4f} s" if fb_solution.status != "Cancelado (N >= 15)" else "Inviable"
            table_rows.append({
                "Algoritmo": "Fuerza Bruta (O(N!))",
                "Distancia (km)": fb_dist_str,
                "Tiempo (s)": fb_time_str,
                "Estado": fb_solution.status,
                "Gap Relativo (%)": "0.00% (Referencia)" if fb_solution.total_distance < float('inf') else "N/A"
            })
        elif fb_infeasible:
            table_rows.append({
                "Algoritmo": "Fuerza Bruta (O(N!))",
                "Distancia (km)": "Incomputable",
                "Tiempo (s)": "Inviable",
                "Estado": f"Cancelado (N = {instance.num_cities} >= 15)",
                "Gap Relativo (%)": "N/A"
            })

        if aco_solution:
            gap_str = "N/A"
            if fb_solution and fb_solution.total_distance < float('inf') and fb_solution.total_distance > 0:
                gap_val = ((aco_solution.total_distance - fb_solution.total_distance) / fb_solution.total_distance) * 100.0
                gap_str = f"{gap_val:+.2f}%"
            elif instance.num_cities >= 15:
                gap_str = "N/A (FB Incomputable)"

            table_rows.append({
                "Algoritmo": "Colonia de Hormigas (ACO)",
                "Distancia (km)": f"{aco_solution.total_distance:.2f} km",
                "Tiempo (s)": f"{aco_solution.execution_time:.4f} s",
                "Estado": aco_solution.status,
                "Gap Relativo (%)": gap_str
            })

        df_comparison = pd.DataFrame(table_rows)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)