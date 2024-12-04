import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

def is_detect_window(point1, point2, point0, r):
    distance1 = np.linalg.norm(point1 - point0)
    distance2 = np.linalg.norm(point2 - point0)
    detect, contained, transects = 0, 0, 0
    line_vec = point2 - point1
    point_vec = point0 - point1
    distance_line = np.abs(np.cross(line_vec, point_vec)) / np.linalg.norm(line_vec)
    if distance_line < r:
        detect = 1
        if distance1 < r and distance2 < r:
            contained = 1
        elif distance2 >= r and distance1 >= r:
            transects = 1
    return detect, contained, transects

def load_lines(filename):
    df = pd.read_csv(filename)
    line_segments = []
    lengths = []
    for i in range(len(df)):
        segment = {
            'point1': np.array([df.loc[i, 'X'], df.loc[i, 'Y']]),
            'point2': np.array([df.loc[i, 'BX'], df.loc[i, 'BY']]),
            'length': np.array([df.loc[i, 'Length']]) / 2
        }
        line_segments.append(segment)
        lengths.append(df.loc[i, 'Length'])
    lengths_array = np.array(lengths)
    return line_segments, lengths_array

def window_method(line_segments, window_center, window_radius):
    n, nc, nt = 0, 0, 0
    for i, segment in enumerate(line_segments):
        point1 = segment['point1']
        point2 = segment['point2']
        detect, contained, transects = is_detect_window(point1, point2, window_center, window_radius)
        n = n + detect
        nc = nc + contained
        nt = nt + transects
    m = n + nc - nt
    nn = n - nc + nt
    dens = m / (2 * np.pi * window_radius ** 2)
    leng = np.pi * window_radius * nn / (2 * m)

    return dens, leng

def plot_radius_density_length(data_dict):
    plt.figure()

    for i, data in data_dict.items():
        radius_values = data['radius_values']
        density_values = data['density_values']
        plt.plot(radius_values, density_values, marker='o', label=f'Number of windows = {i}')
    plt.xlabel("Radius (m)", fontsize=14, fontproperties=font_prop)
    plt.ylabel("Density ($1/m^2$)", fontsize=14, fontproperties=font_prop)
    plt.legend(fontsize=12, prop=font_prop)
    plt.savefig('density_vs_radius.png', dpi=300)
    plt.show()
    plt.close()

    # Plot for length against radius
    plt.figure()
    for i, data in data_dict.items():
        radius_values = data['radius_values']
        length_values = data['length_values']
        plt.plot(radius_values, length_values, marker='o', label=f'Number of windows = {i}')
    plt.xlabel("Radius (m)", fontsize=14,fontproperties=font_prop)
    plt.ylabel("Length (m)", fontsize=14, fontproperties=font_prop)
    plt.legend(fontsize=12, prop=font_prop)
    plt.savefig('length_vs_radius.png', dpi=300)
    # plt.close()
    plt.show()

if __name__ == '__main__':
    lines, _ = load_lines('points.csv')
    results_dict = {}
    for i in [1, 2, 5]:
        radius_values = []
        density_values = []
        length_values = []
        rng = np.random.default_rng(seed=i+193)
        x_coords = rng.uniform(400, 600, i)
        y_coords = rng.uniform(400, 600, i)
        points = np.column_stack((x_coords, y_coords))
        scale = 0.15 / 360
        for radius in np.arange(50, 501, 25):
            d, l = [], []
            for point in points:
                density, length = window_method(lines, point, radius)
                d.append(density)
                l.append(length)
            d = np.array(d)
            l = np.array(l)
            radius_values.append(radius * scale)
            density_values.append(d.mean() / scale**2)
            length_values.append(l.mean() * scale)
        results_dict[i] = {
            'radius_values': radius_values,
            'density_values': density_values,
            'length_values': length_values
        }
    plot_radius_density_length(results_dict)
