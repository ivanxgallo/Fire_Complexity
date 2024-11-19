import re
import numpy as np
import math
from tqdm.notebook import tqdm
from scipy.optimize import curve_fit


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