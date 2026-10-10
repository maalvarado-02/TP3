# TP3 - Estado plano de tensión (MEF) - Presa de gravedad

Código en Python para modelar el equilibrio estático de un sólido elástico
lineal en 2D (estado plano de tensión) con elementos triangulares y
cuadrangulares, lineales y cuadráticos (T3, T6, Q4 y Q8). La aplicación es el 
análisis de una presa de gravedad (tensiones, momento estabilizante y desestabilizante).

Dependencias: `numpy`, `meshio` (ver `requirements.txt`).

## Estado

# Listo:
- Elementos de referencia y cuadratura (T3, T6, Q4, Q8)
- Material elástico lineal con matriz D (plane stress)
- Solid2D: matrices N, B y K para los 4 tipos de elementos
- Lectura de malla y ensamble
- Patch test con los 4 elementos (tensión σx constante)

# Falta:
- Patch test para σy y τxy (opcional)
- Cargas: peso propio y presión hidrostática (módulo aparte de Solid2D)
- Malla de la presa
- Condiciones de borde en la base y solución
- Postproceso: reacciones, tensión principal máxima sobre la deformada (con factor de escala), momentos estabilizante y desestabilizante, factor de seguridad
- Hipótesis de la base, variaciones de geometría
- Bonus: base con condición de borde Robin


## Estructura

```
source/
  elements/
    quadrature.py        reglas de Gauss (triángulo, cuadrilátero, línea)
    shape_functions.py   elementos de referencia (N, dN): TRI3, TRI6, QUAD4, QUAD8
    bar2d.py             barra 2D (código de la cátedra)
    solid2d.py           elemento sólido 2D (B, detJ, Ke)
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
bar_roof.py              ejemplo de la cátedra (techo de barras)
check_shape.py           verificación de elementos de referencia
dam.py                   script principal de la presa, por etapas
patch_test.py            verificacion patch-test de los 4 elementos (σx cte)
meshes/                  mallas de entrada (patch test y presa)
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


## Elemento sólido (solid2d.py)

Solid2D(global_id, node_global_ids, mesh, material, thickness, element_type) sirve para los 4 tipos: toma N y dN del elemento de referencia, calcula el jacobiano y B en cada punto de Gauss y arma Ke = sum(B^T D B detJ t w).

- B_and_detJ(xi, eta): matriz B (3 x 2nn) y det(J); lanza error si detJ <= 0 (revisar el orden de los nodos).
- stiffness_matrix() y global_dofs(): misma interfaz que Bar2D.
- Los vectores de fuerza se arman aparte (no están en Solid2D).


## Formato de malla (meshes/*.mesh)

Texto plano; las líneas que empiezan con # son comentarios.

n_nodes n_elements dim
node_number x y                       (numeración desde 1)
element_number element_type connectivity   (TRI3, TRI6, QUAD4, QUAD8)


## Patch test (patch_test.py)

Parche de elementos irregulares con un nodo interno y cargas nodales equivalentes a σx constante (figura de Cook, sec. 6.13). Se elige el elemento con ELEMENT ("Q4", "T3", "T6" o "Q8") y se lee meshes/patchtest_<ELEMENT>.mesh.

- Apoyos: nodo 1 fijo (ux, uy) y nodo 4 con rodillo (ux = 0).
- Se verifica que σx sea constante y valga 2F/(H t) en las esquinas y en los puntos de Gauss de cada elemento, y que los autovalores de K reducida sean positivos.
- Reacciones esperadas en x (nodos 1 y 4): -F y -2F para T3 y Q4; -F/3 y -2F/3 para T6 y Q8 (la carga consistente en lados cuadráticos se reparte L/6, 4L/6, L/6).


## Etapas de `dam.py`

1. Elementos de referencia (hecho)
2. Material con matriz D (plane stress) (hecho)
3. Malla y elementos `Solid2D` (hecho para el patch test; falta la malla de la presa)
4. Ensamble de K
5. Cargas: peso propio y presión hidrostática
6. Condiciones de borde en la base y solución
7. Postproceso: reacciones, tensión principal máxima, momentos, factor de seguridad
