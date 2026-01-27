"""
Test cases para el ejercicio de one-hot encoding de strings.

Dado un texto y un vocabulario, regresa un vector indicando cuántas veces
aparece cada palabra del vocabulario en el texto.
"""

test_cases = [
    {
        'vocab': ['feliz', 'perrito', 'casa', 'comida', 'semestre', 'puerta', 'vuelo'],
        'messages': [
            'Mañana vuelo a francia',
            'Feliz inicio de semestre',
            'Tengo un perrito en casa'
        ],
        'expected_output': [
            [0, 0, 0, 0, 0, 0, 1],  # solo 'vuelo' aparece
            [1, 0, 0, 0, 1, 0, 0],  # 'feliz' y 'semestre' aparecen
            [0, 1, 1, 0, 0, 0, 0]   # 'perrito' y 'casa' aparecen
        ],
        'description': 'Casos del ejemplo original'
    },
    {
        'vocab': ['hola', 'mundo', 'python'],
        'messages': [
            'Hola mundo',
            'Python es genial',
            'Hola hola python'
        ],
        'expected_output': [
            [1, 1, 0],  # 'hola' y 'mundo'
            [0, 0, 1],  # solo 'python'
            [2, 0, 1]   # 'hola' aparece 2 veces, 'python' 1 vez
        ],
        'description': 'Palabras repetidas: hola aparece dos veces'
    },
    {
        'vocab': ['gato', 'perro', 'ave'],
        'messages': [
            'Tengo un gato y un perro',
            'No tengo mascotas',
            'El gato y el perro juegan'
        ],
        'expected_output': [
            [1, 1, 0],  # 'gato' y 'perro'
            [0, 0, 0],  # ninguna palabra del vocabulario
            [1, 1, 0]   # 'gato' y 'perro'
        ],
        'description': 'Mensaje sin palabras del vocabulario'
    },
    {
        'vocab': ['aprender', 'programar', 'machine', 'learning'],
        'messages': [
            'Quiero aprender machine learning',
            'Programar es divertido',
            'Machine learning y programar van juntos'
        ],
        'expected_output': [
            [1, 0, 1, 1],  # 'aprender', 'machine', 'learning'
            [0, 1, 0, 0],  # solo 'programar'
            [0, 1, 1, 1]   # 'programar', 'machine', 'learning'
        ],
        'description': 'Vocabulario técnico'
    },
    {
        'vocab': ['el', 'la', 'los', 'las'],
        'messages': [
            'El gato y la casa',
            'Los perros y las aves',
            'El sol'
        ],
        'expected_output': [
            [1, 1, 0, 0],  # 'el' y 'la'
            [0, 0, 1, 1],  # 'los' y 'las'
            [1, 0, 0, 0]   # solo 'el'
        ],
        'description': 'Artículos'
    }
]
