import matplotlib
matplotlib.use("Agg")   # backend sin interfaz gráfica
import matplotlib.pyplot as plt

from solve_dam_aux import solve_dam

#  Ejecuta un estudio de convergencia variando nx (ny se calcula como ny_factor * nx). Devuelve una lista de resultados y errores relativos.

elements=["Q4", "T3", "T6", "Q8"]
formulation="plane_strain"   # "plane_stress" o "plane_strain"
base_bc="fixed"              # "fixed" (empotrada, Dirichlet) o "robin" (resortes de Winkler)
nx_list=[5, 10, 20, 30, 40] # 5, 10, 20, 30
ny_factor = 2   # Para que los elementos tengan distintas proporciones (con 2 quedan parejo las dimensiones en x e y)
robin_pairs = [
    (1e8, 1e6),   # Base casi articulada
    (1e10, 1e8),
    (1e11, 1e9),
    (1e12, 1e10),
    (1e14, 1e12)  # Base casi empotrada
]

def run_convergence(elements=("Q4",), formulation="plane_strain", base_bc="fixed", H=30, b=18, c=3, d=2, t=0, 
                    nx_list=(4, 8, 16, 32), ny_factor=2, robin_pairs=[(1e8, 1e6),]):

    all_results = {}
    all_errors = {}

    # --- Ejecutar FEM para cada nivel de malla ---
    print("\n===============================================================================================================")
    print("                          ANALISIS DE CONVERGENCIA PARA TODOS LOS ELEMENTOS")
    print("===============================================================================================================")
    print("|  nx   |  ny   | nodos  || σ1_max [MPa] |   err_σ1   || σ2_min [MPa] |   err_σ2   || umax [mm]  |  err_umax  |")
    print("|-------|-------|--------||--------------|------------||--------------|------------||------------|------------|")
    for elem in elements:
        print(f"|---  ELEMENTO: {elem} ---")

        # CASO 1: BASE FIJA → convergencia normal
        if base_bc == "fixed":
            results = []

            for nx in nx_list:
                ny = ny_factor * nx
                ny_top = max(1, round(ny * d/H))

                r = solve_dam(element=elem, formulation=formulation, base_bc="fixed", 
                                nx=nx, ny=ny, ny_top=ny_top, H=H, b=b, c=c, d=d, t=t)

                results.append(r)

            all_results[elem] = results
        
        # --- Calcular errores relativos ---
            errors = []

            def rel_err(qk, qkm1):
                return abs(qk - qkm1) / abs(qk)

            for i in range(len(results)):
                if i == 0:
                    errors.append(dict(err_sigma1="", err_sigma2="", err_umax=""))
                else:
                    r0 = results[i-1]
                    r1 = results[i]

                    err_s1 = rel_err(r1["sigma1_max"], r0["sigma1_max"])
                    err_s2 = rel_err(r1["sigma2_min"], r0["sigma2_min"])
                    err_umax = rel_err(r1["umax"], r0["umax"])

                    errors.append(dict(
                        err_sigma1=f"{err_s1:.3e}",
                        err_sigma2=f"{err_s2:.3e}",
                        err_umax=f"{err_umax:.3e}"
                    ))

                    all_errors[elem] = errors
    
            for r, e in zip(results, errors):
                nx = r["nx"]
                ny = r["ny"]
                nodes = r["n_nodes"]

                s1 = r["sigma1_max"] / 1e6
                s2 = r["sigma2_min"] / 1e6
                umax = 1e3 * r["umax"]

                print(f"| {nx:5d} | {ny:5d} | {nodes:6d} || "
                    f"{s1:8.3f} MPa | {e['err_sigma1']:10s} || "
                    f"{s2:8.3f} MPa | {e['err_sigma2']:10s} || "
                    f"{umax:10.4f} | {e['err_umax']:10s} |")
                                
                
        # CASO 2: BASE ROBIN → convergencia doble (nx + KN + KT)
        elif base_bc == "robin":

            all_results[elem] = {}

            for KN, KT in robin_pairs:

                print(f"\n|---  ROBIN: KN={KN:.1e}, KT={KT:.1e}  ---")

                results = []

                # refinación de malla
                for nx in nx_list:
                    ny = ny_factor * nx
                    ny_top = max(1, round(ny * d/H))

                    r = solve_dam(element=elem, formulation=formulation, base_bc="robin",
                                    nx=nx, ny=ny, ny_top=ny_top,
                                    H=H, b=b, c=c, d=d, t=t,
                                    KN=KN, KT=KT)

                    results.append(r)

                all_results[elem][(KN, KT)] = results

                # errores
                errors = []
                for i in range(len(results)):
                    if i == 0:
                        errors.append(dict(err_sigma1="", err_sigma2="", err_umax=""))
                    else:
                        r0 = results[i-1]
                        r1 = results[i]

                        err_s1 = abs(r1["sigma1_max"] - r0["sigma1_max"]) / abs(r1["sigma1_max"])
                        err_s2 = abs(r1["sigma2_min"] - r0["sigma2_min"]) / abs(r1["sigma2_min"])
                        err_umax = abs(r1["umax"] - r0["umax"]) / abs(r1["umax"])

                        errors.append(dict(
                            err_sigma1=f"{err_s1:.3e}",
                            err_sigma2=f"{err_s2:.3e}",
                            err_umax=f"{err_umax:.3e}"
                        ))

                # imprimir tabla
                for r, e in zip(results, errors):
                    nx = r["nx"]
                    ny = r["ny"]
                    nodes = r["n_nodes"]

                    s1 = r["sigma1_max"] / 1e6
                    s2 = r["sigma2_min"] / 1e6
                    umax = 1e3 * r["umax"]

                    print(f"| {nx:5d} | {ny:5d} | {nodes:6d} || "
                            f"{s1:8.3f} MPa | {e['err_sigma1']:10s} || "
                            f"{s2:8.3f} MPa | {e['err_sigma2']:10s} || "
                            f"{umax:10.4f} | {e['err_umax']:10s} |")

        else:
            raise ValueError("base_bc debe ser 'fixed' o 'robin'")

    return all_results, all_errors

# Genera gráficos de convergencia para: sigma1_max, sigma2_min y umax
def plot_convergence(all_results, nx_list, base_bc):
    # σ1_max ----------------------------
    plt.figure(figsize=(8,5))
    for elem, results in all_results.items():
        s1_vals = [r["sigma1_max"]/1e6 for r in results]  # MPa
        plt.plot(nx_list, s1_vals, marker='o', label=elem)

    plt.title("Convergencia de σ1_max")
    plt.xlabel("nx")
    plt.ylabel("σ1_max [MPa]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/convergencia/conv__sigma1_{base_bc}.png", dpi=200)
    plt.close()

    # σ2_min ----------------------------
    plt.figure(figsize=(8,5))
    for elem, results in all_results.items():
        s2_vals = [r["sigma2_min"]/1e6 for r in results]  # MPa
        plt.plot(nx_list, s2_vals, marker='o', label=elem)

    plt.title("Convergencia de σ2_min")
    plt.xlabel("nx")
    plt.ylabel("σ2_min [MPa]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/convergencia/conv__sigma2_{base_bc}.png", dpi=200)
    plt.close()

    # umax ----------------------------
    plt.figure(figsize=(8,5))
    for elem, results in all_results.items():
        umax_vals = [1e3*r["umax"] for r in results]  # mm
        plt.plot(nx_list, umax_vals, marker='o', label=elem)

    plt.title("Convergencia de desplazamiento máximo")
    plt.xlabel("nx")
    plt.ylabel("umax [mm]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/convergencia/conv_umax_{base_bc}.png", dpi=200)
    plt.close()

    print("\nGraficos guardado en results/")


def plot_convergence_robin(all_results, nx_list, base_bc):
    import os
    os.makedirs("results", exist_ok=True)

    # Para cada elemento
    for elem, result_sets in all_results.items():

        # σ1_max ----------------------------
        plt.figure(figsize=(8,5))

        for (KN, KT), results in result_sets.items():
            s1_vals = [r["sigma1_max"]/1e6 for r in results]
            plt.plot(nx_list, s1_vals, marker='o',
                     label=f"KN={KN:.1e}, KT={KT:.1e}")

        plt.title(f"Convergencia σ1_max — {elem}")
        plt.xlabel("nx")
        plt.ylabel("σ1_max [MPa]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/convergencia/conv_{elem}_sigma1_robin.png", dpi=200)
        plt.close()

        # σ2_min ----------------------------
    
        plt.figure(figsize=(8,5))

        for (KN, KT), results in result_sets.items():
            s2_vals = [r["sigma2_min"]/1e6 for r in results]
            plt.plot(nx_list, s2_vals, marker='o',
                     label=f"KN={KN:.1e}, KT={KT:.1e}")

        plt.title(f"Convergencia σ2_min — {elem}")
        plt.xlabel("nx")
        plt.ylabel("σ2_min [MPa]")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/convergencia/conv_{elem}_sigma2_robin.png", dpi=200)
        plt.close()

    print("\nGráficos Robin guardados en results/")

results, errors = run_convergence(elements=elements,
                                  formulation=formulation,
                                  base_bc=base_bc, 
                                  nx_list=nx_list,
                                  robin_pairs=robin_pairs)
if base_bc == "fixed":
    plot_convergence(results, nx_list, base_bc)
else: 
    plot_convergence_robin(results, nx_list, base_bc)