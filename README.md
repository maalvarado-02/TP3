# TP3 - Estado plano de tensión (MEF) - Presa de gravedad

Código en Python para modelar el equilibrio estático de un sólido elástico
lineal en 2D (estado plano de tensión) con elementos triangulares y
cuadrangulares, lineales y cuadráticos. La aplicación es el análisis de una
presa de gravedad (tensiones, momento estabilizante y desestabilizante).

## Uso rápido

```
python check_shape.py     # verifica funciones de forma y cuadratura
python bar_roof.py        # ejemplo de la cátedra (barras 2D, techo)
python dam.py             # modelo de la presa (se activa por etapas)
```

Dependencias: `numpy`, `meshio` (ver `requirements.txt`).

## Estructura

```
source/
  elements/
    quadrature.py        reglas de Gauss (triángulo, cuadrilátero, línea)
    shape_functions.py   elementos de referencia: TRI3, TRI6, QUAD4, QUAD8
    bar2d.py             barra 2D (código de la cátedra)
    solid2d.py           elemento sólido 2D           [PENDIENTE]
  materials/
    linear_elastic.py    material elástico lineal (E, nu, rho, matriz D)
  mesh/
    mesh.py, node.py     lectura de malla y conectividades
  systems/
    dense_matrix.py      matriz de rigidez densa y ensamble
    boundary_conditions.py   condiciones de Dirichlet
  solvers/
    linear.py            resolución K u = f
  output/
    paraview.py          escritura de resultados .vtu (meshio)
bar_roof.py              ejemplo de la cátedra (techo)
check_shape.py           verificación de elementos de referencia
dam.py                   script principal de la presa, por etapas
meshes/                  mallas de entrada
results/                 salidas para ParaView
```

## Elementos de referencia (`shape_functions.py`)

Cada elemento hereda de `ReferenceElement` y define:

- `node_coords`: coordenadas de los nodos en el elemento de referencia.
- `shape(xi, eta)`: devuelve `N` (nn,) y `dN` (nn, 2) = [dN/dxi, dN/deta].
- `gauss_rule()`: puntos (ng, 2) y pesos (ng,) de integración.
- `meshio_type`: nombre de celda para escribir en meshio / ParaView.
- `reference_area()`: 1/2 para triángulos, 4 para cuadriláteros.

Dominio de referencia: triángulos en (0,0)-(1,0)-(0,1); cuadriláteros en
[-1, 1] x [-1, 1].

### Orden de los nodos (igual a meshio / VTK)

| Elemento | Esquinas | Nodos de lado (medios)             |
|----------|----------|------------------------------------|
| TRI3     | 0 1 2    | -                                  |
| TRI6     | 0 1 2    | 3 (0-1), 4 (1-2), 5 (2-0)          |
| QUAD4    | 0 1 2 3  | -                                  |
| QUAD8    | 0 1 2 3  | 4 (0-1), 5 (1-2), 6 (2-3), 7 (3-0) |

Las esquinas van en sentido antihorario.

## Cuadratura (`quadrature.py`)

| Elemento | Regla                  | Exactitud |
|----------|------------------------|-----------|
| TRI3     | 1 punto                | grado 1   |
| TRI6     | 3 puntos               | grado 2   |
| QUAD4    | Gauss 2 x 2            | -         |
| QUAD8    | Gauss 3 x 3            | -         |
| aristas  | `line_rule(n)` 1D      | para cargas de borde |

## Verificación (`check_shape.py`)

Para cada elemento comprueba: delta de Kronecker en los nodos, partición de
la unidad, suma de derivadas nula, derivadas contra diferencias finitas,
reproducción de campo lineal y suma de pesos igual al área de referencia.
Todos los errores deben dar del orden de 1e-10 o menos.

## Formato de malla (propuesto, a implementar)

Texto plano con secciones. El tipo de elemento, el orden y el material se
indican en cada elemento:

```
# n_nodes n_elements dim
MATERIALS
# id name E nu rho
1 concrete 21e9 0.2 2400
NODES
# id x y
1 0.0 0.0
ELEMENTS
# id type order material n1 n2 ...
1 QUAD4 1 1 1 2 6 5
NODESETS
# name  node ids
base 1 2 3 4
EDGESETS
# name  element local_edge
upstream 5 1
```

La malla del techo (`meshes/roof.mesh`, elementos `BAR2D`) se mantiene como
caso de prueba de que el lector de mallas sigue funcionando.

## Etapas de `dam.py`

1. Elementos de referencia (hecho)
2. Material con matriz D (plane stress)
3. Malla y elementos `Solid2D`
4. Ensamble de K
5. Cargas: peso propio y presión hidrostática
6. Condiciones de borde en la base y solución
7. Postproceso: reacciones, tensión principal máxima, momentos, factor de seguridad

## Pendientes

- [ ] Unificar imports (`source.` en todos los archivos)
- [ ] Aristas locales por elemento (para cargas de borde)
- [ ] `LinearElastic` con `rho` y `D_plane_stress()`
- [ ] `Solid2D` (K, fuerza de volumen, carga de arista, tensiones)
- [ ] Lector de malla con secciones
- [ ] Patch test por tipo de elemento
- [ ] Generador de malla estructurada de la presa (nx, ny)
- [ ] Visualización sobre la deformada con factor de escala
- [ ] Momentos y factor de seguridad
- [ ] Bonus: condición de borde Robin en la base