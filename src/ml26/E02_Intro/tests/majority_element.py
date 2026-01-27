"""
Test cases para el ejercicio de elemento mayoritario.

El elemento mayoritario es aquel que aparece más veces en un arreglo.
Si todos aparecen la misma cantidad de veces, se regresa el primero.
"""

test_cases = [
    {
        'input': [3, 2, 3],
        'expected_output': 3,
        'description': 'Caso básico: 3 aparece 2 veces'
    },
    {
        'input': [2, 2, 1, 1, 1, 2, 2],
        'expected_output': 2,
        'description': 'Caso ejemplo: 2 aparece 4 veces, 1 aparece 3 veces'
    },
    {
        'input': [1],
        'expected_output': 1,
        'description': 'Un solo elemento'
    },
    {
        'input': [1, 1, 2, 2],
        'expected_output': 1,
        'description': 'Empate: se regresa el primero'
    },
    {
        'input': [6, 5, 5],
        'expected_output': 5,
        'description': 'Caso simple: 5 es mayoritario'
    },
    {
        'input': [1, 2, 3, 4, 5, 5, 5, 5],
        'expected_output': 5,
        'description': 'Muchos elementos diferentes, 5 es mayoritario'
    },
    {
        'input': [-1, -1, -1, 0, 0],
        'expected_output': -1,
        'description': 'Números negativos: -1 aparece más veces'
    },
    {
        'input': [10, 10, 10, 20, 20, 30],
        'expected_output': 10,
        'description': 'Diferentes frecuencias: 10 aparece 3 veces'
    }
]
