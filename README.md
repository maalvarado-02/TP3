# TP3 - Estado plano de deformación - Presa de gravedad

Código en Python para modelar el equilibrio estático de un sólido elástico
lineal en 2D (estado plano de deformación) con elementos triangulares y
cuadrangulares, lineales y cuadráticos (T3, T6, Q4 y Q8). La aplicación es el 
análisis de una presa de gravedad de hormigón H°21 bajo peso propio y presión hidrostática (tensiones, momento estabilizante y desestabilizante, FS).

Basado en la estructura bar_2d de la cátedra (msc-itba/bar_2d).

Dependencias: `numpy`, `meshio`, `scipy`,`matplotlib` (ver `requirements.txt`).


## Estructura

```
source/
  elements/
    quadrature.py           reglas de Gauss (triángulo, cuadrilátero, línea)
    shape_functions.py      elementos de referencia (N, dN): TRI3, TRI6, QUAD4, QUAD8 y aristas 1D
    solid2d.py              elemento sólido 2D (B, detJ, Ke, tensiones)
  materials/
    linear_elastic.py       material elástico lineal (E, nu, rho, matriz D)
  mesh/
    mesh.py, node.py        lectura de malla y conectividades
  systems/
    matrix.py               DenseMatrix y SparseMatrix (ensamble de K)
    boundary_conditions.py  Dirichlet (apply_dirichlet_bc) y Robin (apply_robin_bc)
    loads.py                cargas de peso propio y presión sobre aristas
  solvers/
    linear.py               resolución K u = f - solve (denso) y solve_sparse (disperso)
  output/
    paraview.py             escritura de resultados .vtu (meshio)
dam.py                      script principal de la presa, por etapas
solve_dam_aux.py            solve_dam(): la misma resolución como función (para convergencia)
convergence.py              estudio de convergencia (T3, T6, Q4, Q8; empotrada y Robin)
patch_test.py               patch test de los 4 elementos (σx constante)
make_dam_mesh.py            crea las mallas de la presa para los 4 elementos
meshes/                     mallas de entrada (patch test y presa)
results/                    salidas para ParaView y gráficos
```

## Elementos de referencia (`shape_functions.py`)

Cada elemento hereda de `ReferenceElement` y define:

- `node_coords`: coordenadas de los nodos en el elemento de referencia.
- `shape(xi, eta)`: devuelve `N` (nn,) y `dN` (nn, 2) = [dN/dxi, dN/deta].
- `gauss_rule()`: puntos (ng, 2) y pesos (ng,) de integración.
- `meshio_type`: nombre de celda para escribir en meshio / ParaView.
- `edges`: índices locales de los nodos de cada arista (antihorarias)
- `reference_area()`: 1/2 para triángulos, 4 para cuadriláteros.

Dominio de referencia: triángulos en (0,0)-(1,0)-(0,1); cuadriláteros en [-1, 1] x [-1, 1].


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

Solid2D(global_id, node_global_ids, mesh, material, thickness, element_type, formulation) sirve para los 4 tipos: toma N y dN del elemento de referencia, calcula el jacobiano y B en cada punto de Gauss y arma Ke = sum(B^T D B detJ t w).

- `B_and_detJ(xi, eta)`: matriz B (3 x 2nn) y det(J); lanza error si detJ <= 0 (revisar el orden de los nodos).
- `stiffness_matrix()` y `global_dofs()`: misma interfaz que `Bar2D`.
- `stress_at(xi, eta, u)`, `stress_at_gauss(u)`, `average_stress(u)`: tensión [σx, σy, τxy] en un
  punto, en los puntos de Gauss y promedio del elemento.
- `principal_stress(u)` y `vonMises_stress(u)`: sobre la tensión promedio (σz = ν(σx+σy) en
  deformación plana, 0 en tensión plana).


## Formato de malla (meshes/*.mesh)

Texto plano; las líneas que empiezan con # son comentarios.

n_nodes | n_elements | dim | n_materials
material_id | type | E | nu | rho   (type: solo linear_elastic)
node_number | x | y                       (numeración desde 1)
element_number | element_type | material_id | connectivity

`element_type` es `TRI3`, `TRI6`, `QUAD4` o `QUAD8`: define a la vez la familia y el orden de interpolación.


## Cargas (loads.py)
- `assemble_body_force(f, elements, b)`: peso propio, `b = (0, -rho*g)`.
- `assemble_pressure(f, elements, on_edge, pressure)`: presión normal de compresión sobre las
  aristas donde `on_edge(p, q)` sea verdadero. Usa funciones de forma 1D de 2 o 3 nodos y Gauss
  de 3 puntos, por lo que sirve para lineales y cuadráticos.

En la presa: peso propio en todo el hormigón y presión sobre la cara aguas arriba por debajo del nivel del agua


## Condiciones de borde (boundary_conditions.py)

- **Empotramiento** (`apply_dirichlet_bc`): ux = uy = 0 en los nodos de la base (y = 0).
  Corresponde a una presa sobre roca mucho más rígida que el hormigón (sin deslizamiento, despegue ni asentamiento). 
  Transmite tracciones que una junta real no admite y genera una singularidad de tensiones en el talón.
- **Robin** (`apply_robin_bc(K, elements, on_edge, k_normal, k_tangential)`): resortes de Winkler
  independientes en las aristas de la base (t = -(kn n n^T + kt s s^T) u),. Se suma una matriz de rigidez de borde
   a K, no se fija ningún GDL.
  `k -> 0` recupera el borde libre y `k -> inf` el empotramiento.


## Patch test (patch_test.py)

Parche de elementos irregulares con un nodo interno y cargas nodales equivalentes a σx constante
(figura de Cook, sec. 6.13). Se elige el elemento con `ELEMENT` ("Q4", "T3", "T6" o "Q8")y se
 lee `meshes/patchtest_<ELEMENT>.mesh`.

- Apoyos: nodo 1 fijo (ux, uy) y nodo 4 con rodillo (ux = 0).
- Se verifica que σx sea constante y valga 2F/(H t) en las esquinas y en los puntos de Gauss de
 cada elemento, y que los autovalores de K reducida sean positivos.
- Reacciones esperadas en x (nodos 1 y 4): -F y -2F para T3 y Q4; -F/3 y -2F/3 para T6 y Q8
 (la carga consistente en lados cuadráticos se reparte L/6, 4L/6, L/6).


## Etapas de `dam.py`

1. Datos de geometría y malla (`make_dam_mesh`)
2. Malla y elementos `Solid2D`
3. Ensamble de K (matriz sparse)
4. Cargas: peso propio y presión hidrostática
5. Condiciones de borde en la base (empotrada o Robin)
6. Solución del sistema K.u=f y chequeo de autovalores de K reducida
7. Postproceso: reacciones, equilibrio global, momentos estabilizante y desestabilizante, FS,
   tensión principal máxima, desplazamiento máximo, archivo `.vtu`