# TP3 - Estado plano de deformación - Presa de gravedad

Código en Python para modelar el equilibrio estático de un sólido elástico
lineal en 2D (estado plano de deformación) con elementos triangulares y
cuadrangulares, lineales y cuadráticos (T3, T6, Q4 y Q8). La aplicación es el 
análisis de una presa de gravedad (tensiones, momento estabilizante y desestabilizante).

Dependencias: `numpy`, `meshio` (ver `requirements.txt`).

## Estado

# Listo:
- Elementos de referencia y cuadratura (T3, T6, Q4, Q8)
- Material elástico lineal con matriz D (plane strain o plane stress)
- Solid2D: matrices N, B y K para los 4 tipos de elementos
- Formato de malla con tipo de elemento, orden de interpolación y material
- Generador de mallas estructuradas de la presa (make_dam_mesh.py)
- Cargas: peso propio y presión hidrostática (loads.py)
- Condiciones de borde: empotramiento (Dirichlet)
- Patch test con los 4 elementos (tensión σx constante)
- Postproceso: reacciones, tensión principal máxima sobre la deformada (con factor de escala), momentos estabilizante y desestabilizante, FS

# Falta:
- Patch test para σy y τxy (opcional)
- Hipótesis de la base, variaciones de geometría
- Condiciones de borde: resortes distribuidos (Robin) (BONUS)
- Estudio de convergencia de malla y de variaciones de geometría (FS)

## Estructura

```
source/
  elements/
    quadrature.py           reglas de Gauss (triángulo, cuadrilátero, línea)
    shape_functions.py      elementos de referencia (N, dN): TRI3, TRI6, QUAD4, QUAD8
    solid2d.py              elemento sólido 2D (B, detJ, Ke)
  materials/
    linear_elastic.py       material elástico lineal (E, nu, rho, matriz D)
  mesh/
    mesh.py, node.py        lectura de malla y conectividades
  systems/
    dense_matrix.py         matriz de rigidez densa y ensamble
    boundary_conditions.py  Dirichlet (apply_dirichlet_bc) y Robin (assemble_robin)
    loads.py                aplicar cargas de peso e hidrostaticas
  solvers/
    linear.py               resolución K u = f
  output/
    paraview.py             escritura de resultados .vtu (meshio)
check_shape.py              verificación de elementos de referencia
dam.py                      script principal de la presa, por etapas
patch_test.py               verificacion patch-test de los 4 elementos (σx cte)
make_dam_mesh.py            crea las mallas de la presa para los 4 elementos
meshes/                     mallas de entrada (patch test y presa)
results/                    salidas para ParaView
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

| Elemento | Regla              | Exactitud               |
|----------|--------------------|-------------------------|
| TRI3     | 1 punto            | grado 1                 |
| TRI6     | 3 puntos           | grado 2                 |
| QUAD4    | Gauss 2 x 2        | -                       |
| QUAD8    | Gauss 3 x 3        | -                       |
| aristas  | `line_rule(n)` 1D  | cargas de borde y Robin |


## Elemento sólido (solid2d.py)

Solid2D(global_id, node_global_ids, mesh, material, thickness, element_type) sirve para los 4 tipos: toma N y dN del elemento de referencia, calcula el jacobiano y B en cada punto de Gauss y arma Ke = sum(B^T D B detJ t w).

- B_and_detJ(xi, eta): matriz B (3 x 2nn) y det(J); lanza error si detJ <= 0 (revisar el orden de los nodos).
- stiffness_matrix() y global_dofs(): misma interfaz que Bar2D.
- stress_at(xi, eta, u), stress_at_gauss(u), average_stress(u): tensión [σx, σy, τxy] en un punto, en los puntos de Gauss y promedio del elemento.
- principal_stress(u) y vonMises_stress(u): sobre la tensión promedio.
- Los vectores de fuerza se arman aparte (loads.py).

## Formato de malla (meshes/*.mesh)

Texto plano; las líneas que empiezan con # son comentarios.

n_nodes | n_elements | dim | n_materials
material_id | type | E | nu | rho   (type: solo linear_elastic)
node_number | x | y                       (numeración desde 1)
element_number | element_type | material_id | connectivity

## Cargas (loads.py)
- assemble_body_force(f, elements, b): peso propio, b = (0, -rho*g).
assemble_pressure(f, elements, on_edge, pressure): presión normal de compresión sobre las aristas donde on_edge(p, q) sea verdadero. - Usa funciones de forma 1D de 2 o 3 nodos y Gauss de 3 puntos, por lo que sirve para lineales y cuadráticos.
- EDGES: aristas locales de cada tipo de elemento.

En la presa: peso propio en todo el hormigón y presión sobre la cara aguas arriba por debajo del nivel del agua

## Condiciones de borde (boundary_conditions.py)




## Patch test (patch_test.py)

Parche de elementos irregulares con un nodo interno y cargas nodales equivalentes a σx constante (figura de Cook, sec. 6.13). Se elige el elemento con ELEMENT ("Q4", "T3", "T6" o "Q8") y se lee meshes/patchtest_<ELEMENT>.mesh.

- Apoyos: nodo 1 fijo (ux, uy) y nodo 4 con rodillo (ux = 0).
- Se verifica que σx sea constante y valga 2F/(H t) en las esquinas y en los puntos de Gauss de cada elemento, y que los autovalores de K reducida sean positivos.
- Reacciones esperadas en x (nodos 1 y 4): -F y -2F para T3 y Q4; -F/3 y -2F/3 para T6 y Q8 (la carga consistente en lados cuadráticos se reparte L/6, 4L/6, L/6).


## Etapas de `dam.py`

1. Datos de geometria (hecho)
2. Material con matriz D (plane strain) (hecho)
3. Malla y elementos `Solid2D` (hecho)
4. Ensamble de K (hecho)
5. Cargas: peso propio y presión hidrostática (hecho)
6. Condiciones de borde en la base (empotrada o Robin)
7. Solución del sistema K.u=f y chequeo de singularidad de K (hecho)
8. Postproceso: reacciones, tensiones, tensión principal máxima, momentos, FS, desplazamiento maximo (hecho)
