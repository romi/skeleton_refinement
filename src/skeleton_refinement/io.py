from pathlib import Path

import numpy as np


def load_xyz(filename):
    """Load a point cloud or skeleton file saved as a series of space-separated XYZ coordinates.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the point cloud or skeleton file to parse.

    Returns
    -------
    numpy.ndarray
        The XYZ coordinates of the point cloud or skeleton.
    """
    f = open(filename, "r")
    lines = f.readlines()
    org_x = []
    org_y = []
    org_z = []
    for l in lines:
        org_x.append(float(l.split(' ')[0]))
        org_y.append(float(l.split(' ')[1]))
        org_z.append(float(l.split(' ')[2]))
    f.close()
    X = np.column_stack((org_x, org_y, org_z))
    return X


def load_ply(filename):
    """Load a point cloud coordinates.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the point cloud ofile to parse.

    Returns
    -------
    numpy.ndarray
        The XYZ coordinates of the point cloud or skeleton.
    """
    from plyfile import PlyData
    plydata = PlyData.read(filename)
    X = np.array([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    return X


def load_json(filename, key=None):
    """Load a point cloud or skeleton file from a json file.
    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the point cloud or skeleton file to parse.
    key : str, optional
        The key of the JSON dictionary containing the point cloud or skeleton coordinates to load.

    Returns
    -------
    numpy.ndarray
        The XYZ coordinates of the point cloud or skeleton.
    """
    import json
    with open(filename, mode='rb') as f:
        X = json.load(f)

    if key is not None:
        X = X[key]
    return np.array(X)


def load_nx(filename, key='position'):
    """Load a tree graph from a pickled networkx file.
    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the tree graph file to parse.

    Returns
    -------
    numpy.ndarray
        The XYZ coordinates of the point cloud or skeleton.
    """
    import pickle
    with open(filename, mode='rb') as f:
        G = pickle.load(f)

    X = []
    for node in G.nodes:
        X.append(G.nodes[node][key])

    return np.array(X)


def save_json(filename, G, **kwargs):
    """Save a tree graph to a JSON file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the JSON file to write.
    G : networkx.Graph
        Graph to write.
    """
    import json
    data = {
        "points": [G.nodes[node]['position'] for node in G.nodes],
        "lines": list(G.edges),
    }
    if 'indent' not in kwargs:
        kwargs.update({'indent': 2})
    with open(filename, 'w') as f:
        f.writelines(json.dumps(data, **kwargs))
    return


def save_nx(filename, G, **kwargs):
    """Save a tree graph to a pickle file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the pickle file to write.
    G : networkx.Graph
        Graph to write.
    """
    import pickle
    if 'protocol' not in kwargs:
        kwargs['protocol'] = pickle.HIGHEST_PROTOCOL
    if isinstance(filename, str):
        filename = Path(filename)
    with filename.open(mode='wb') as f:
        pickle.dump(G, f, **kwargs)
    return
