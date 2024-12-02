import re
import numpy as np
import math
from tqdm.notebook import tqdm
from scipy.optimize import curve_fit
import networkx as nx
import cartopy.crs as ccrs


def dms_to_decimal(coord):
    # Usar re para extraer grados, minutos, segundos y dirección
    match = re.match(r'(\d+)°(\d+)\'(\d+)"\s*([NSEO])', coord)
    if match:
        degrees = int(match.group(1))
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        direction = match.group(4)
        # Convertir a decimal
        decimal = degrees + minutes / 60 + seconds / 3600

        # Hacer negativo si es S o O
        if direction in ['S', 'O']:
            decimal = -decimal

        return decimal
    else:
        raise ValueError("Invalid DMS format.")


def pl(x, A, a):
    return A*x**a

def XY_fit(x_data, y_data, func=pl, x_range=[1, 1e5], n_points=1000):
    # Fit using curve_fit
    params, _ = curve_fit(func, x_data, y_data)

    # Generate x values in the specified range
    x_fit = np.linspace(x_range[0], x_range[1], n_points)
    # Calculate y values of the fitted function with obtained parameters
    y_fit = func(x_fit, *params)

    return x_fit, y_fit, params

def lonlat_to_cartesian(lat, lon, h=0):
    """
    Convierte coordenadas geográficas (latitud, longitud, altitud) 
    en coordenadas cartesianas (X, Y, Z).

    Parámetros:
    - lat: Latitud en grados.
    - lon: Longitud en grados.
    - h: Altitud en kilómetros (opcional, por defecto 0).

    Retorna:
    - Una tupla (x, y, z) con las coordenadas cartesianas.
    """
    R = 6371  # Radio promedio de la Tierra en km
    phi = math.radians(lat)  # Convertir latitud a radianes
    lam = math.radians(lon)  # Convertir longitud a radianes

    # Cálculos de coordenadas cartesianas
    x = (R + h) * math.cos(phi) * math.cos(lam)
    y = (R + h) * math.cos(phi) * math.sin(lam)
    z = (R + h) * math.sin(phi)

    return (x, y, z)

def distance_between_points(point1, point2):
    """
    Calcula la distancia euclidiana entre dos puntos en coordenadas cartesianas.

    Parámetros:
    - point1: Tupla (x1, y1, z1) del primer punto.
    - point2: Tupla (x2, y2, z2) del segundo punto.

    Retorna:
    - Distancia euclidiana entre los dos puntos.
    """
    x1, y1, z1 = point1
    x2, y2, z2 = point2

    # Fórmula de la distancia euclidiana
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

    return distance

def N_i(coords, eps, j=0):
    count = 0
    for i, coord in enumerate(coords):
        if i!=j :
            dist = distance_between_points(coord, coords[j])
            if eps > dist:
                count += 1
    return count

def C_q(coords, eps, q=2):
    N = len(coords)
    total = 0
    for j, coord in enumerate(tqdm(coords)):
        total += (N_i(coords, eps, j)/(N-1))**(q-1)

    total = total/N
    total = total**(1/(q-1))

    return total

def D_q(coords, eps=1, q=2):
    return np.log10(C_q(coords, eps, q=q))/np.log10(eps)


# -------------- Creating Graph ------------ #

def create_graph_map(lon_range = [], lat_range = [], cell_size=0.1):
    """
    Crea un grafo basado en una cuadrícula regular donde cada nodo es identificado
    directamente por su tupla (lon_idx, lat_idx).

    Parámetros:
    ----------
    lon_range : list o tuple
        Rango de longitudes [min_lon, max_lon].
    lat_range : list o tuple
        Rango de latitudes [min_lat, max_lat].
    cell_size : float
        Tamaño de cada celda en grados.

    Retorna:
    -------
    G : networkx.Graph
        Grafo donde los nodos son identificados por tuplas (lon_idx, lat_idx),
        y cada nodo tiene como atributo su posición central.
    """
    # Crear arrays para las coordenadas centrales de las celdas
    min_lon, max_lon = lon_range
    min_lat, max_lat = lat_range
    lon_centers = np.arange(min_lon + cell_size / 2, max_lon, cell_size)
    lat_centers = np.arange(min_lat + cell_size / 2, max_lat, cell_size)

    # Crear un grafo vacío
    G = nx.Graph()

    # Agregar nodos con las posiciones centrales como atributos
    for lat_idx, lat in enumerate(lat_centers):
        for lon_idx, lon in enumerate(lon_centers):
            G.add_node((lon_idx, lat_idx), position=(lon, lat))

    return G


def create_connected_graph_map(df, cell_size=0.1, lat_col="dLat", lon_col="dLon", date_col="Inicio"):
    """
    Crea un grafo basado en una cuadrícula y agrega conexiones entre nodos
    según la secuencialidad de eventos en el dataframe.

    Parámetros:
    ----------
    df : pandas.DataFrame
        DataFrame con columnas 'lon', 'lat', y 'Inicio' (tiempo).
    cell_size : float
        Tamaño de cada celda en grados.

    Retorna:
    -------
    G : networkx.Graph
        Grafo con nodos representando celdas y conexiones entre nodos por eventos secuenciales.
    """
    # Calcular rangos de latitud y longitud automáticamente
    min_lon, max_lon = df[lon_col].min(), df[lon_col].max()
    min_lat, max_lat = df[lat_col].min(), df[lat_col].max()

    # Ajustar los rangos para incluir completamente los extremos
    lon_range = [min_lon - cell_size / 2, max_lon + cell_size / 2]
    lat_range = [min_lat - cell_size / 2, max_lat + cell_size / 2]

    # Crear el grafo base
    G = create_graph_map(lon_range, lat_range, cell_size)

    # Función auxiliar para asignar eventos a celdas
    def assign_to_cell(lon, lat):
        lon_idx = int((lon - lon_range[0]) // cell_size)
        lat_idx = int((lat - lat_range[0]) // cell_size)
        return (lon_idx, lat_idx)

    # Agregar una columna con la celda correspondiente a cada evento
    df['cell'] = df.apply(lambda row: assign_to_cell(row[lon_col], row[lat_col]), axis=1)

    # Ordenar por tiempo para evaluar eventos secuenciales
    df.sort_values(by=date_col, ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Iterar por los eventos y conectar nodos
    for i in range(len(df) - 1):
        cell_current = df.loc[i, 'cell']
        cell_next = df.loc[i + 1, 'cell']

        # Conectar nodos si son distintos
        if cell_current != cell_next:
            G.add_edge(cell_current, cell_next)

    return G


def plot_graph_on_map(ax, G, base_size=5, pondered_size=5, edge_width=0.1, alpha_edge=0.2):
    """
    Plotea un grafo sobre un mapa usando las posiciones (lon, lat) de los nodos,
    escalando el tamaño de los nodos según su grado. Excluye nodos sin conexiones.

    Parámetros:
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        El eje del mapa donde se superpone el grafo.
    G : networkx.Graph
        El grafo a superponer, cuyos nodos tienen el atributo 'position' como (lon, lat).
    """
    # Filtrar nodos con grado mayor a 0
    connected_nodes = [(node, degree) for node, degree in G.degree() if degree > 0]

    # Iterar sobre nodos conectados para plotearlos
    for node, degree in connected_nodes:
        lon, lat = G.nodes[node]['position']

        # Escalar el tamaño del nodo según su grado
        node_size = base_size + degree * pondered_size # Tamaño base + escalamiento por grado

        ax.scatter(
            lon, lat,
            s=node_size, color='red', edgecolor='black', zorder=5, transform=ccrs.PlateCarree()
        )

    # Iterar sobre edges para dibujarlos
    for u, v in G.edges():
        lon_u, lat_u = G.nodes[u]['position']
        lon_v, lat_v = G.nodes[v]['position']
        ax.plot(
            [lon_u, lon_v], [lat_u, lat_v],
            color='black', linewidth=edge_width, zorder=4, transform=ccrs.PlateCarree(), alpha=alpha_edge
        )