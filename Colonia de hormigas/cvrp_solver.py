"""
================================================================================
TRABAJO PRÁCTICO: PROBLEMA DE RUTEO DE VEHÍCULOS CON CAPACIDAD (CVRP)
Algoritmos: Fuerza Bruta vs. Optimización por Colonia de Hormigas (ACO)
================================================================================
Autores: Cátedra de Algoritmos & Estructuras de Datos / Inteligencia Artificial
Python: 3.10+ (Compatible con Python 3.14)
Dependencias: matplotlib, numpy
================================================================================
"""

import math
import random
import time
import itertools
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np


# ==============================================================================
# 1. MODELO DE DATOS Y GENERACIÓN DE INSTANCIAS DEL PROBLEMA
# ==============================================================================

@dataclass
class CVRPInstance:
    """
    Representa una instancia del Problema de Ruteo de Vehículos con Capacidad (CVRP).
    
    Atributos:
        num_cities: Cantidad de clientes/ciudades (N, excluyendo el depósito).
        capacity: Capacidad máxima de carga por vehículo (K).
        coords: Lista de tuplas (x, y) con las coordenadas de cada nodo (0 es depósito).
        demands: Lista de demandas de cada nodo (nodo 0 siempre demanda 0).
        dist_matrix: Matriz de distancias euclidianas entre todos los pares de nodos.
    """
    num_cities: int
    capacity: int
    coords: List[Tuple[float, float]]
    demands: List[int]
    dist_matrix: List[List[float]] = field(init=False)

    def __post_init__(self):
        total_nodes = self.num_cities + 1
        self.dist_matrix = [[0.0] * total_nodes for _ in range(total_nodes)]
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    dx = self.coords[i][0] - self.coords[j][0]
                    dy = self.coords[i][1] - self.coords[j][1]
                    self.dist_matrix[i][j] = math.hypot(dx, dy)

    @classmethod
    def generate(cls, num_cities: int, capacity: int = 12, seed: int = 42) -> "CVRPInstance":
        """
        Genera una instancia sintética reproducible (seed=42).
        El depósito (Nodo 0) se ubica en el centro (50, 50).
        Las N ciudades se distribuyen aleatoriamente en el plano [0, 100] x [0, 100].
        Las demandas son enteros entre 1 y 4 (para K=12, implica 3 a 5 paradas por camión).
        """
        rng = random.Random(seed)
        coords = [(50.0, 50.0)]  # Nodo 0: Depósito Central
        demands = [0]            # El depósito no tiene demanda

        for _ in range(num_cities):
            x = round(rng.uniform(5.0, 95.0), 2)
            y = round(rng.uniform(5.0, 95.0), 2)
            coords.append((x, y))
            demands.append(rng.randint(1, 4))

        return cls(num_cities=num_cities, capacity=capacity, coords=coords, demands=demands)


@dataclass
class CVRPSolution:
    """
    Representa una solución al problema CVRP.
    
    Atributos:
        routes: Lista de rutas. Cada ruta es una lista de nodos que empieza y termina en 0.
        total_distance: Suma de las longitudes de todas las rutas.
        execution_time: Tiempo de cómputo en segundos medido con time.perf_counter().
        algorithm_name: Nombre del algoritmo utilizado.
        status: Estado de la ejecución ('Óptimo', 'Aproximado', 'Timeout/Cancelado', etc.)
        notes: Notas o estadísticas adicionales (ej. permutaciones analizadas).
    """
    routes: List[List[int]]
    total_distance: float
    execution_time: float
    algorithm_name: str
    status: str = "Completado"
    notes: str = ""


# ==============================================================================
# 2. FUNCIONES DE EVALUACIÓN Y PARTICIONAMIENTO
# ==============================================================================

def compute_route_distance(route: List[int], dist_matrix: List[List[float]]) -> float:
    """Calcula la distancia recorrida en una sola ruta cerrada."""
    dist = 0.0
    for idx in range(len(route) - 1):
        dist += dist_matrix[route[idx]][route[idx + 1]]
    return dist


def compute_total_distance(routes: List[List[int]], dist_matrix: List[List[float]]) -> float:
    """Calcula la distancia total recorrida por toda la flota de vehículos."""
    return sum(compute_route_distance(r, dist_matrix) for r in routes)


def split_permutation_into_routes(permutation: Tuple[int, ...], instance: CVRPInstance) -> List[List[int]]:
    """
    Particiona vorazmente (greedy split) una secuencia ordenada de ciudades en
    rutas válidas que respetan la capacidad máxima K del vehículo.
    Cada ruta inicia en 0 y finaliza en 0.
    """
    routes: List[List[int]] = []
    current_route = [0]
    current_load = 0

    for city in permutation:
        demand = instance.demands[city]
        if current_load + demand <= instance.capacity:
            current_route.append(city)
            current_load += demand
        else:
            current_route.append(0)
            routes.append(current_route)
            current_route = [0, city]
            current_load = demand

    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)

    return routes


# ==============================================================================
# 3. ALGORITMO 1: FUERZA BRUTA / EXHAUSTIVO (O(N!))
# ==============================================================================

def solve_cvrp_brute_force(instance: CVRPInstance, timeout_seconds: float = 5.0) -> CVRPSolution:
    """
    Resuelve el CVRP explorando todas las permutaciones posibles de las N ciudades.
    
    Para N >= 15 (o si se supera el timeout de 5 segundos), se interrumpe y evidencia
    la imposibilidad de cómputo por explosión combinatoria factorial O(N!).
    """
    n = instance.num_cities
    cities = list(range(1, n + 1))
    total_permutations = math.factorial(n)

    # Si N >= 15, la explosión factorial hace inviable siquiera empezar:
    # 15! = 1.307.674.368.000 (1.3 billones de combinaciones)
    if n >= 15:
        return CVRPSolution(
            routes=[],
            total_distance=float('inf'),
            execution_time=0.0,
            algorithm_name="Fuerza Bruta",
            status="Cancelado (N >= 15)",
            notes=(f"Incomputable por explosión factorial: {n}! = {total_permutations:,} permutaciones. "
                   f"A 1.000.000 ops/seg tomaría ~{total_permutations / 1e6 / 86400:.1f} días.")
        )

    best_distance = float('inf')
    best_routes: List[List[int]] = []
    perms_evaluated = 0

    start_time = time.perf_counter()

    for perm in itertools.permutations(cities):
        perms_evaluated += 1

        # Control de tiempo periódico (cada 20.000 iteraciones para no degradar rendimiento)
        if perms_evaluated % 20000 == 0:
            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout_seconds:
                return CVRPSolution(
                    routes=best_routes,
                    total_distance=best_distance,
                    execution_time=elapsed,
                    algorithm_name="Fuerza Bruta",
                    status=f"Interrumpido por Timeout ({timeout_seconds}s)",
                    notes=f"Evaluadas {perms_evaluated:,} de {total_permutations:,} permutaciones ({perms_evaluated/total_permutations*100:.2f}%)."
                )

        candidate_routes = split_permutation_into_routes(perm, instance)
        candidate_dist = compute_total_distance(candidate_routes, instance.dist_matrix)

        if candidate_dist < best_distance:
            best_distance = candidate_dist
            best_routes = candidate_routes

    elapsed = time.perf_counter() - start_time

    return CVRPSolution(
        routes=best_routes,
        total_distance=best_distance,
        execution_time=elapsed,
        algorithm_name="Fuerza Bruta",
        status="Óptimo Global",
        notes=f"Espacio de búsqueda completo explorado: {perms_evaluated:,} permutaciones analizadas."
    )


# ==============================================================================
# 4. ALGORITMO 2: OPTIMIZACIÓN POR COLONIA DE HORMIGAS (ACO)
# ==============================================================================

class AntColonyCVRP:
    """
    Implementación del algoritmo de Colonia de Hormigas adaptado a CVRP.
    
    Cada hormiga construye una solución factible respetando en tiempo real
    la capacidad máxima K del vehículo. Si el camión actual se llena, regresa
    al depósito y continúa con un nuevo vehículo hasta visitar todos los clientes.
    """
    def __init__(
        self,
        instance: CVRPInstance,
        num_ants: int = 30,
        num_iterations: int = 60,
        alpha: float = 1.0,      # Ponderación del rastro de feromona
        beta: float = 2.5,       # Ponderación de la información heurística (visibilidad = 1/dist)
        rho: float = 0.1,        # Tasa de evaporación de feromona
        q: float = 100.0,        # Factor constante de depósito de feromona
        elitism_weight: float = 2.0,  # Refuerzo extra a la mejor solución global
        seed: int = 42
    ):
        self.instance = instance
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.elitism_weight = elitism_weight
        self.rng = random.Random(seed)

        n = instance.num_cities + 1
        # Inicialización de matriz de visibilidad heurística: eta_ij = 1 / d_ij
        self.visibility = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and self.instance.dist_matrix[i][j] > 0:
                    self.visibility[i][j] = 1.0 / self.instance.dist_matrix[i][j]
                else:
                    self.visibility[i][j] = 0.0

        # Feromona inicial tau_0 (uniforme)
        self.initial_pheromone = 1.0 / (instance.num_cities * 10.0)
        self.pheromone = [[self.initial_pheromone] * n for _ in range(n)]

    def _select_next_city(self, current: int, feasible_cities: List[int]) -> int:
        """Selección estocástica mediante rueda de ruleta ponderada por feromona y heurística."""
        probabilities: List[float] = []
        for city in feasible_cities:
            tau = self.pheromone[current][city] ** self.alpha
            eta = self.visibility[current][city] ** self.beta
            probabilities.append(tau * eta)

        total_prob = sum(probabilities)
        if total_prob <= 0.0:
            return self.rng.choice(feasible_cities)

        # Muestreo con ruleta acumulada
        pick = self.rng.uniform(0.0, total_prob)
        cumulative = 0.0
        for city, prob in zip(feasible_cities, probabilities):
            cumulative += prob
            if pick <= cumulative:
                return city
        return feasible_cities[-1]

    def _construct_solution(self) -> Tuple[List[List[int]], float]:
        """Una hormiga construye un conjunto completo de rutas que satisfacen la demanda."""
        unvisited = set(range(1, self.instance.num_cities + 1))
        routes: List[List[int]] = []

        while unvisited:
            route = [0]
            current_node = 0
            current_load = 0

            while True:
                # Filtrar ciudades que aún no fueron visitadas y entran en la capacidad restante
                feasible = [
                    c for c in unvisited
                    if current_load + self.instance.demands[c] <= self.instance.capacity
                ]

                if not feasible:
                    # El vehículo no puede cargar más: debe regresar al depósito central
                    break

                next_city = self._select_next_city(current_node, feasible)
                route.append(next_city)
                current_load += self.instance.demands[next_city]
                unvisited.remove(next_city)
                current_node = next_city

            route.append(0)
            routes.append(route)

        total_dist = compute_total_distance(routes, self.instance.dist_matrix)
        return routes, total_dist

    def solve(self) -> CVRPSolution:
        """Ejecuta el ciclo de vida de la Colonia de Hormigas."""
        start_time = time.perf_counter()

        best_routes: List[List[int]] = []
        best_distance = float('inf')

        for _ in range(self.num_iterations):
            iteration_solutions: List[Tuple[List[List[int]], float]] = []

            # 1. Fase de construcción de todas las hormigas
            for _ in range(self.num_ants):
                routes, dist = self._construct_solution()
                iteration_solutions.append((routes, dist))
                if dist < best_distance:
                    best_distance = dist
                    best_routes = routes

            # 2. Fase de evaporación de feromonas: tau = (1 - rho) * tau
            n = self.instance.num_cities + 1
            for i in range(n):
                for j in range(n):
                    self.pheromone[i][j] *= (1.0 - self.rho)
                    # Mantener un límite inferior para evitar el congelamiento de caminos
                    if self.pheromone[i][j] < 1e-6:
                        self.pheromone[i][j] = 1e-6

            # 3. Fase de depósito de feromonas por las hormigas de la iteración
            for routes, dist in iteration_solutions:
                delta = self.q / dist
                for route in routes:
                    for idx in range(len(route) - 1):
                        u, v = route[idx], route[idx + 1]
                        self.pheromone[u][v] += delta
                        self.pheromone[v][u] += delta

            # 4. Refuerzo elitista: la mejor solución global deposita feromona extra
            if best_routes:
                elite_delta = (self.q / best_distance) * self.elitism_weight
                for route in best_routes:
                    for idx in range(len(route) - 1):
                        u, v = route[idx], route[idx + 1]
                        self.pheromone[u][v] += elite_delta
                        self.pheromone[v][u] += elite_delta

        elapsed = time.perf_counter() - start_time

        return CVRPSolution(
            routes=best_routes,
            total_distance=best_distance,
            execution_time=elapsed,
            algorithm_name="Colonia de Hormigas (ACO)",
            status="Metaheurística Convergida",
            notes=f"{self.num_iterations} iteraciones, {self.num_ants} hormigas por ciclo."
        )


# ==============================================================================
# 5. VISUALIZACIÓN GRÁFICA CON MATPLOTLIB
# ==============================================================================

def plot_cvrp_solution(
    instance: CVRPInstance,
    solution: CVRPSolution,
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Genera un gráfico bidimensional con las rutas resultantes diferenciadas por color.
    Cada ruta sale y regresa al Depósito Central (Nodo 0).
    """
    if not solution.routes:
        print(f"[Aviso] No hay rutas para graficar en {solution.algorithm_name} ({solution.status}).")
        return

    plt.figure(figsize=(9, 8))
    ax = plt.subplot(111)

    # Paleta de colores distinguible para cada vehículo
    color_palette = [
        '#E6194B', '#3CB44B', '#FFE119', '#4363D8', '#F58231',
        '#911EB4', '#42D4F4', '#F032E6', '#BFEF45', '#FABEBE',
        '#469990', '#DCBEFF', '#9A6324', '#FFFAC8', '#800000'
    ]

    # Graficar las rutas de los vehículos
    for r_idx, route in enumerate(solution.routes):
        color = color_palette[r_idx % len(color_palette)]
        xs = [instance.coords[node][0] for node in route]
        ys = [instance.coords[node][1] for node in route]
        load = sum(instance.demands[node] for node in route)

        ax.plot(
            xs, ys,
            color=color,
            linewidth=2.2,
            alpha=0.85,
            marker='o',
            markersize=6,
            label=f"Vehículo #{r_idx + 1} (Carga: {load}/{instance.capacity} paq.)"
        )

        # Flechas de dirección en los tramos
        for k in range(len(route) - 1):
            u, v = route[k], route[k + 1]
            x_mid = (instance.coords[u][0] + instance.coords[v][0]) / 2.0
            y_mid = (instance.coords[u][1] + instance.coords[v][1]) / 2.0
            dx = (instance.coords[v][0] - instance.coords[u][0]) * 0.1
            dy = (instance.coords[v][1] - instance.coords[u][1]) * 0.1
            ax.annotate(
                '', xy=(x_mid + dx, y_mid + dy), xytext=(x_mid, y_mid),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5, alpha=0.9)
            )

    # Graficar clientes
    for city_id in range(1, instance.num_cities + 1):
        x, y = instance.coords[city_id]
        ax.scatter(x, y, c='dodgerblue', s=130, edgecolors='black', zorder=5)
        ax.text(
            x, y + 2.0,
            f"C{city_id}\n(d={instance.demands[city_id]})",
            fontsize=8, ha='center', va='bottom',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="gray", lw=0.5)
        )

    # Graficar Depósito Central
    depot_x, depot_y = instance.coords[0]
    ax.scatter(depot_x, depot_y, c='gold', s=300, marker='*', edgecolors='black', linewidth=1.5, zorder=6, label="Depósito (Nodo 0)")
    ax.text(depot_x, depot_y - 4.5, "DEPÓSITO CENTRAL", fontsize=9, fontweight='bold', ha='center', va='top')

    # Configuración de ejes y título
    ax.set_title(
        f"Rutas CVRP: {solution.algorithm_name}\n"
        f"N = {instance.num_cities} ciudades | Capacidad K = {instance.capacity} | Distancia Total = {solution.total_distance:.2f} km\n"
        f"Tiempo de Cómputo: {solution.execution_time:.4f} s | {solution.status}",
        fontsize=11, fontweight='bold', pad=12
    )
    ax.set_xlabel("Coordenada X (km)", fontsize=10)
    ax.set_ylabel("Coordenada Y (km)", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0), fontsize=8, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[Gráfico guardado exitosamente]: {save_path}")

    if show:
        plt.show()
    plt.close()


# ==============================================================================
# 6. BANCO DE PRUEBAS AUTOMÁTICAS (N = 5, 10, 20)
# ==============================================================================

def run_benchmarks(save_plots: bool = True):
    """
    Ejecuta el protocolo comparativo formal para N = 5, 10 y 20 ciudades.
    Mide tiempos de CPU con time.perf_counter() y contrasta la calidad de ruta.
    """
    test_cases = [
        {"n": 5, "capacity": 10, "aco_ants": 20, "aco_iters": 40},
        {"n": 10, "capacity": 12, "aco_ants": 25, "aco_iters": 60},
        {"n": 20, "capacity": 15, "aco_ants": 35, "aco_iters": 80},
    ]

    print("\n" + "=" * 95)
    print(" PROTOCOLO DE EXPERIMENTACIÓN: FUERZA BRUTA vs. COLONIA DE HORMIGAS (ACO)")
    print("=" * 95)

    results_table = []

    for test in test_cases:
        n = test["n"]
        cap = test["capacity"]
        print(f"\n>>> EVALUANDO INSTANCIA: N = {n} CIUDADES | CAPACIDAD K = {cap} (Seed = 42)")
        print("-" * 95)

        instance = CVRPInstance.generate(num_cities=n, capacity=cap, seed=42)

        # 1. Fuerza Bruta
        print(f"[*] Ejecutando Fuerza Bruta (O(N!))...")
        fb_sol = solve_cvrp_brute_force(instance, timeout_seconds=5.0)
        print(f"    - Estado: {fb_sol.status}")
        print(f"    - Tiempo: {fb_sol.execution_time:.6f} segundos")
        print(f"    - Distancia: {fb_sol.total_distance:.2f} km")
        if fb_sol.notes:
            print(f"    - Detalle: {fb_sol.notes}")

        # 2. Colonia de Hormigas (ACO)
        print(f"[*] Ejecutando Colonia de Hormigas (ACO)...")
        aco = AntColonyCVRP(
            instance,
            num_ants=test["aco_ants"],
            num_iterations=test["aco_iters"],
            alpha=1.0,
            beta=2.5,
            rho=0.1,
            q=100.0,
            seed=42
        )
        aco_sol = aco.solve()
        print(f"    - Estado: {aco_sol.status}")
        print(f"    - Tiempo: {aco_sol.execution_time:.6f} segundos")
        print(f"    - Distancia: {aco_sol.total_distance:.2f} km")
        print(f"    - Rutas asignadas: {len(aco_sol.routes)} vehículos")

        # Diferencia de calidad si FB llegó a solución óptima
        gap = "N/A"
        if fb_sol.total_distance < float('inf') and fb_sol.total_distance > 0:
            diff = ((aco_sol.total_distance - fb_sol.total_distance) / fb_sol.total_distance) * 100
            gap = f"{diff:+.2f}%"

        results_table.append({
            "N": n,
            "K": cap,
            "FB_Dist": f"{fb_sol.total_distance:.2f}" if fb_sol.total_distance < float('inf') else "Incomputable",
            "FB_Time": f"{fb_sol.execution_time:.4f}s" if fb_sol.status != "Cancelado (N >= 15)" else "Inviable",
            "ACO_Dist": f"{aco_sol.total_distance:.2f}",
            "ACO_Time": f"{aco_sol.execution_time:.4f}s",
            "Gap": gap
        })

        # Generar y guardar gráficos comparativos
        if save_plots:
            if fb_sol.routes:
                plot_cvrp_solution(instance, fb_sol, save_path=f"cvrp_n{n}_fuerza_bruta.png", show=False)
            plot_cvrp_solution(instance, aco_sol, save_path=f"cvrp_n{n}_aco.png", show=False)

    # Imprimir Cuadro Resumen
    print("\n" + "=" * 95)
    print(f"{'RESUMEN COMPARATIVO FINAL':^95}")
    print("=" * 95)
    print(f"{'N':<5} | {'K':<4} | {'FB Distancia':<14} | {'FB Tiempo':<12} | {'ACO Distancia':<14} | {'ACO Tiempo':<12} | {'Gap ACO vs FB':<14}")
    print("-" * 95)
    for r in results_table:
        print(f"{r['N']:<5} | {r['K']:<4} | {r['FB_Dist']:<14} | {r['FB_Time']:<12} | {r['ACO_Dist']:<14} | {r['ACO_Time']:<12} | {r['Gap']:<14}")
    print("=" * 95)
    print("[INFO] Los gráficos PNG de cada instancia han sido guardados en el directorio actual.\n")


# ==============================================================================
# 7. MENÚ INTERACTIVO EN CONSOLA
# ==============================================================================

def interactive_menu():
    """Interfaz por consola para el usuario."""
    while True:
        print("\n" + "=" * 60)
        print(" SISTEMA DE RESOLUCIÓN DE CVRP - CÁTEDRA DE ALGORITMOS")
        print("=" * 60)
        print("1. Ejecutar batería automática de pruebas (N = 5, 10, 20)")
        print("2. Resolver caso personalizado (elegir N, K y algoritmo)")
        print("3. Salir")
        print("-" * 60)
        choice = input("Seleccione una opción (1-3): ").strip()

        if choice == "1":
            run_benchmarks(save_plots=True)
        elif choice == "2":
            try:
                n_str = input("Ingrese cantidad de ciudades N (ej: 8): ").strip()
                n = int(n_str)
                k_str = input("Ingrese capacidad de los vehículos K (ej: 12): ").strip()
                k = int(k_str)

                instance = CVRPInstance.generate(num_cities=n, capacity=k, seed=42)
                print(f"\nInstancia generada: {n} ciudades, Depósito Central en (50, 50).")
                print("Demandas:", instance.demands[1:])

                print("\nSeleccione Algoritmo:")
                print("a) Fuerza Bruta Exhaustiva (O(N!))")
                print("b) Colonia de Hormigas (ACO)")
                print("c) Ambos y comparar")
                alg_choice = input("Opción (a/b/c): ").strip().lower()

                if alg_choice in ('a', 'c'):
                    print("\n[*] Ejecutando Fuerza Bruta...")
                    fb_res = solve_cvrp_brute_force(instance, timeout_seconds=6.0)
                    print(f"Resultado FB: Distancia = {fb_res.total_distance:.2f} km | Tiempo = {fb_res.execution_time:.4f} s | {fb_res.status}")
                    if fb_res.routes:
                        plot_cvrp_solution(instance, fb_res, show=True)

                if alg_choice in ('b', 'c'):
                    print("\n[*] Ejecutando Colonia de Hormigas (ACO)...")
                    aco = AntColonyCVRP(instance, num_ants=30, num_iterations=60, seed=42)
                    aco_res = aco.solve()
                    print(f"Resultado ACO: Distancia = {aco_res.total_distance:.2f} km | Tiempo = {aco_res.execution_time:.4f} s | {aco_res.status}")
                    plot_cvrp_solution(instance, aco_res, show=True)

            except ValueError:
                print("[Error] Ingrese números enteros válidos.")
            except Exception as e:
                print(f"[Error inesperado]: {e}")
        elif choice == "3":
            print("\nFinalizando programa. ¡Hasta luego!")
            break
        else:
            print("[Opción no válida] Por favor elija 1, 2 o 3.")


if __name__ == "__main__":
    import sys
    if "--menu" in sys.argv:
        interactive_menu()
    else:
        run_benchmarks(save_plots=True)
        print("\nPara acceder al menú interactivo ejecute: python cvrp_solver.py --menu")

