import numpy as np
import random
import matplotlib.pyplot as plt

debug = False

truss1_points = np.array([[0,0],[0,0.5],[0.5,0],[0.5,0.5],[1,0],[1,0.5]])
truss3_points = np.array([[0,0],[0,0.5],[0,1],[0.5,0.25],[0.5,0.75],[1,0.5]])
truss4_points = np.array([[0,0.25],[0,0.75],[0.5,0],[0.5,1],[1,0.5]])
truss5_points = np.array([[0,0.25],[0,0.75],[0.5,0],[0.5,0.5],[0.5,1],[1,0.5]])
truss6_points = np.array([[0,1/3],[0,2/3],[0.5,0],[0.5,1],[1,1/3],[1,2/3]])

truss1_connections = np.array([[0,2],[0,3],[1,2],[1,3],[2,3],[2,4],[2,5],[3,4],[3,5],[4,5]])
truss3_connections = np.array([[0,1],[0,3],[1,2],[1,3],[1,4],[1,5],[2,4],[3,4],[3,5],[4,5]])
truss4_connections = np.array([[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]])
truss5_connections = np.array([[0,1],[0,2],[0,3],[1,3],[1,4],[2,3],[2,5],[3,4],[3,5],[4,5]])
truss6_connections = np.array([[0,1],[0,2],[0,4],[0,5],[1,3],[1,4],[1,5],[2,4],[3,5],[4,5]])

base_trusses = [
    (truss1_points, truss1_connections),
    # (truss3_points, truss3_connections),
    # # (truss4_points, truss4_connections),
    # (truss5_points, truss5_connections),
    # (truss6_points, truss6_connections),
]

base_force_range = (-500,500) # Base force between -500 and 500 N before scaling
force_scale_exp = 1.5
base_area_range = (1e-5,5e-4) # Base area between 50 and 300 mm^2 before scaling
area_scale_exp = 1

def visualize_trusses(trusses):
    for i, (pts, conns) in enumerate(trusses):
        plt.figure()
        for c in conns:
            p1, p2 = pts[c[0]], pts[c[1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-')
        plt.scatter(pts[:,0], pts[:,1], c='red')
        plt.title(f"Truss {i+1}")
        plt.axis('equal')
        plt.grid(True)
        plt.show()


def generate_data():
    truss = random.choice(base_trusses)
    base_points = truss[0]
    edges = truss[1]
    num_nodes = base_points.shape[0]
    num_bars = edges.shape[0]

    perm = np.array([i for i in range(num_nodes)])
    # perm = np.random.permutation(num_nodes)
    base_points = base_points[perm]
    index_map = {old: new for new, old in enumerate(perm)}
    edges = np.array([[index_map[i], index_map[j]] for i, j in edges])

    scale = np.random.uniform(1,10)

    perturbations = scale * np.random.uniform(0,0.1,(num_nodes,2))
    points = scale*base_points + perturbations


    areas = (scale**area_scale_exp) * np.random.uniform(*base_area_range,num_bars)
    forces = (scale**force_scale_exp) * np.random.uniform(*base_force_range,(2,random.randint(1,2)))
    force_nodes = np.random.choice(range(2,num_nodes),forces.shape[1],replace=False)

    node_forces = np.zeros((2,num_nodes))
    node_forces[:,force_nodes] = forces
    node_forces = node_forces[:,perm]

    force_nodes = np.where(node_forces[0] != 0)

    supports = np.array([[1,1,0,0,0,0],[1,1,0,0,0,0]])
    supports = supports[:,perm]

    force_flags = np.zeros(num_nodes)
    force_flags[force_nodes] = 1
    force_flags = force_flags[perm]

    truss_dat = np.concatenate([points.T,node_forces,supports,force_flags.reshape(1,num_nodes)],axis=0)

    # FEM calculation
    E = 2e11  # Young's modulus in Pa (e.g., for steel)
    K_global = np.zeros((2 * num_nodes, 2 * num_nodes))
    F_global = np.zeros(2 * num_nodes)

    # Assemble global stiffness matrix
    for idx, (n1, n2) in enumerate(edges):
        x1, y1 = points[n1]
        x2, y2 = points[n2]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        C = (x2 - x1) / L
        S = (y2 - y1) / L
        A = areas[idx]

        k_local = (E * A / L) * np.array([
            [ C*C,  C*S, -C*C, -C*S],
            [ C*S,  S*S, -C*S, -S*S],
            [-C*C, -C*S,  C*C,  C*S],
            [-C*S, -S*S,  C*S,  S*S]
        ])

        dof = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
        for i in range(4):
            for j in range(4):
                K_global[dof[i], dof[j]] += k_local[i, j]

    # Apply external forces
    for f_node, force in zip(force_nodes, np.transpose(forces)):
        F_global[2 * f_node] = force[0]
        F_global[2 * f_node + 1] = force[1]

    # Apply boundary conditions
    constrained_dofs = [0, 1, 2, 3]  # node 0 Y, node 1 X and Y
    free_dofs = np.setdiff1d(np.arange(2 * num_nodes), constrained_dofs)

    # Reduce system
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    F_reduced = F_global[free_dofs]

    # Solve for displacements
    displacements = np.zeros(2 * num_nodes)
    displacements[free_dofs] = np.linalg.solve(K_reduced, F_reduced)

    # Compute axial stress in each member
    stresses = []
    for idx, (n1, n2) in enumerate(edges):
        x1, y1 = points[n1]
        x2, y2 = points[n2]
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        C = (x2 - x1) / L
        S = (y2 - y1) / L
        u = displacements[[2*n1, 2*n1+1, 2*n2, 2*n2+1]]

        # Axial strain (delta length / original length)
        strain = (-C)*u[0] + (-S)*u[1] + (C)*u[2] + (S)*u[3]
        stress = E * strain
        stresses.append(stress)

    stresses = np.array(stresses)
    max_disp = np.max(np.abs(displacements))
    max_stress = np.max(stresses)
    min_stress = np.min(stresses)

    if debug:
        print(f"Max displacement: {max_disp:.6f} m")
        print(f"Max stress: {max_stress/1e6:.2f} MPa")
        print(f"Min stress: {min_stress/1e6:.2f} MPa")
        print(f"Forces: {forces}")
        print(f"Force location: {force_nodes}")
        print(f"Max area: {np.max(areas)}")
        print(f"Scale: {scale}")
        print(f"Supports: {supports}")
        visualize_trusses([(points,edges)])
    return (truss_dat.T,areas*E,edges,displacements)

num_points = 20000

if debug: num_points = 1

trusses = []
truss_areas = []
truss_edges = []
targets = []

for i in range(num_points):
    truss,areas,edges,target = generate_data()
    trusses.append(truss)
    truss_areas.append(areas)
    truss_edges.append(edges)
    targets.append(target)
    if i%1000 == 0: print(i)
    
if not debug:
    np.save("Truss_dat.npy",np.array(trusses))
    np.save("Truss_areas.npy",np.array(truss_areas))
    np.save("Truss_edges.npy",np.array(truss_edges))
    np.save("Stress_Disp_dat.npy",np.array(targets))