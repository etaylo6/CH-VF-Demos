"""
Timoshenko beam subgraph analysis script with overdefinition fix.
This script runs comprehensive table tests across all subgraph configurations.
"""

import numpy as np
from constrainthg.hypergraph import Hypergraph, Node
from itertools import combinations
from threading import Thread
import queue
import io
from contextlib import redirect_stdout


def timoshenko_subgraph(a_bool, m_bool, s_bool, target_node_name=None):
    """
    Create a fresh Timoshenko beam subgraph with the specified configuration.
    Each call creates completely new Node and Hypergraph objects to ensure isolation.
    """
    # Create completely fresh hypergraph
    hg = Hypergraph()
    
    # Create fresh core nodes (always present) - each call gets new Node objects
    P = Node("point load")
    k = Node("kappa")
    E = Node("youngs modulus")
    I = Node("moment of inertia")
    G = Node("shear modulus")
    A = Node("area")
    L = Node("length")
    theta = Node("theta")
    
    # Create fresh conditional nodes
    if a_bool == 0: 
        radius = Node("radius")
    if s_bool == 0:
        V = Node("poisson")

    ### E,V,G Relations ###
    if s_bool == 0:
        def shear_modulus_from_elastic_poisson(E, V):
            return E / (2 * (1 + V))

        def elastic_modulus_from_shear_poisson(G, V):
            return 2 * G * (1 + V)

        def poisson_from_elastic_shear(E, G):
            return (E / (2 * G)) - 1

    if a_bool == 0:
        ### R,A Relations ###
        def area_from_radius(radius):
            return np.pi * radius**2
        def radius_from_area(A):
            return np.sqrt(A / np.pi)

    if m_bool == 0:
        ### R,I Relations ###
        def moment_of_inertia_from_radius(radius):
            return (np.pi / 4) * radius**4

        def radius_from_moment_of_inertia(I):
            return (4 * I / np.pi)**(1/4)
        
        # Also create direct moment of inertia relations when radius is not available
        def moment_of_inertia_from_area(A):
            # Assuming circular cross-section: A = π*r², so r = √(A/π)
            # I = π*r⁴/4 = π*(A/π)²/4 = A²/(4π)
            return A**2 / (4 * np.pi)
        
        def area_from_moment_of_inertia(I):
            # I = A²/(4π), so A = √(4πI)
            return np.sqrt(4 * np.pi * I)

    ### Timoshenko Relations ###
    def timoshenko_beam_deflection(P, E, I, L, k, G, A):
        # Full Timoshenko beam deflection including length dependency
        delta_bending = P * L**3 / (3 * E * I)
        delta_shear = P * L / (k * G * A)
        return delta_bending + delta_shear

    def solve_for_P_from_timoshenko(theta, E, I, L, k, G, A):
        return theta / (L**3/(3*E*I) + L/(k*G*A))

    def solve_for_E_from_timoshenko(theta, P, I, L, k, G, A):
        return P * L**3 / (3 * I * (theta - P*L/(k*G*A)))

    def solve_for_I_from_timoshenko(theta, P, E, L, k, G, A):
        return P * L**3 / (3 * E * (theta - P*L/(k*G*A)))

    def solve_for_L_from_timoshenko(theta, P, E, I, k, G, A):
        # Solve cubic equation: P*L^3/(3*E*I) + P*L/(k*G*A) - theta = 0
        # Rearrange: (P/(3*E*I))*L^3 + (P/(k*G*A))*L - theta = 0
        coeffs = [P/(3*E*I), 0, P/(k*G*A), -theta]
        roots = np.roots(coeffs)
        # Return the first real positive root
        real_positive_roots = roots[np.isreal(roots) & (roots > 0)]
        if len(real_positive_roots) > 0:
            return real_positive_roots[0]
        else:
            return 0  # Fallback if no positive real root found

    def solve_for_k_from_timoshenko(theta, P, E, I, L, G, A):
        return P * L / (G * A * (theta - P*L**3/(3*E*I)))

    def solve_for_G_from_timoshenko(theta, P, E, I, L, k, A):
        return P * L / (k * A * (theta - P*L**3/(3*E*I)))

    def solve_for_A_from_timoshenko(theta, P, E, I, L, k, G):
        return P * L / (k * G * (theta - P*L**3/(3*E*I)))


    # E,V,G Edges - only add if s_bool == 0 (V node exists)
    if s_bool == 0:
        if target_node_name is None or target_node_name == "shear modulus":
            hg.add_edge([E, V], G, shear_modulus_from_elastic_poisson, label='E,V->G')
        if target_node_name is None or target_node_name == "youngs modulus":
            hg.add_edge([G, V], E, elastic_modulus_from_shear_poisson, label='G,V->E')
        if target_node_name is None or target_node_name == "poisson":
            hg.add_edge([E, G], V, poisson_from_elastic_shear, label='E,G->V')

    # R,A Edges - only add if a_bool == 0 (radius node exists)
    if a_bool == 0:
        if target_node_name is None or target_node_name == "area":
            hg.add_edge([radius], A, area_from_radius, label='radius->A')
        if target_node_name is None or target_node_name == "radius":
            hg.add_edge([A], radius, radius_from_area, label='A->radius')

    # R,I Edges - add if m_bool == 0 (moment of inertia relations available)
    if m_bool == 0:
        # If radius node exists, add radius-I relations
        if a_bool == 0:
            if target_node_name is None or target_node_name == "moment of inertia":
                hg.add_edge([radius], I, moment_of_inertia_from_radius, label='radius->I')
            if target_node_name is None or target_node_name == "radius":
                hg.add_edge([I], radius, radius_from_moment_of_inertia, label='I->radius')
        
        # Always add direct A-I relations when m_bool == 0
        if target_node_name is None or target_node_name == "moment of inertia":
            hg.add_edge([A], I, moment_of_inertia_from_area, label='A->I')
        if target_node_name is None or target_node_name == "area":
            hg.add_edge([I], A, area_from_moment_of_inertia, label='I->A')

    # Timoshenko Edges - only add the necessary edge for the target
    if target_node_name is None or target_node_name == "theta":
        hg.add_edge([P, E, I, L, k, G, A], theta, timoshenko_beam_deflection, label='Timoshenko')
    if target_node_name is None or target_node_name == "point load":
        hg.add_edge([theta, E, I, L, k, G, A], P, solve_for_P_from_timoshenko, label='Timoshenko->P')
    if target_node_name is None or target_node_name == "youngs modulus":
        hg.add_edge([theta, P, I, L, k, G, A], E, solve_for_E_from_timoshenko, label='Timoshenko->E')
    if target_node_name is None or target_node_name == "moment of inertia":
        hg.add_edge([theta, P, E, L, k, G, A], I, solve_for_I_from_timoshenko, label='Timoshenko->I')
    if target_node_name is None or target_node_name == "length":
        hg.add_edge([theta, P, E, I, k, G, A], L, solve_for_L_from_timoshenko, label='Timoshenko->L')
    if target_node_name is None or target_node_name == "kappa":
        hg.add_edge([theta, P, E, I, L, G, A], k, solve_for_k_from_timoshenko, label='Timoshenko->k')
    if target_node_name is None or target_node_name == "shear modulus":
        hg.add_edge([theta, P, E, I, L, k, A], G, solve_for_G_from_timoshenko, label='Timoshenko->G')
    if target_node_name is None or target_node_name == "area":
        hg.add_edge([theta, P, E, I, L, k, G], A, solve_for_A_from_timoshenko, label='Timoshenko->A')

    return hg


# Default values dictionary for Timoshenko beam parameters
DEFAULT_VALUES = {
    # Core Timoshenko parameters (using actual node names)
    'point load': 1000.0,      # Point load (N)
    'youngs modulus': 200e9,   # Young's modulus (Pa) - typical for steel
    'moment of inertia': 1e-6, # Moment of inertia (m^4)
    'length': 1.0,             # Length (m)
    'kappa': 5/6,              # Shear correction factor (rectangular cross-section)
    'shear modulus': 80e9,     # Shear modulus (Pa) - typical for steel
    'area': 1e-4,              # Cross-sectional area (m^2)
    'theta': 0.01,             # Deflection (m)
    
    # Optional parameters (when nodes are included)
    'poisson': 0.3,            # Poisson's ratio (typical for steel)
    'radius': 0.005,           # Radius (m) - for circular cross-section
}


def comprehensive_subgraph_analysis():
    """
    Analyze all possible subgraph configurations and test every possible input combination
    for each target variable.
    """
    # Test all 8 possible subgraph configurations (2^3 = 8)
    subgraph_configs = [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
    ]
    
    results = []
    
    for config_idx, config in enumerate(subgraph_configs):
        a_bool, m_bool, s_bool = config
        
        # Create hypergraph for this configuration
        hg = timoshenko_subgraph(a_bool, m_bool, s_bool)
        
        # Get available nodes and their labels
        available_nodes = []
        node_labels = []
        for node in hg.nodes.keys():
            if hasattr(node, 'label'):
                label = node.label
            else:
                label = str(node)
            available_nodes.append(node)
            node_labels.append(label)
        
        # Test each node as a potential target (silently)
        total_successful = 0
        total_tested = 0
        combo_counter = 0
        
        for target_idx, target_label in enumerate(node_labels):
            # Get all other nodes as potential inputs
            input_nodes = [label for label in node_labels if label != target_label]
            
            if not input_nodes:
                continue
            
            # Test all possible combinations of input nodes
            # Start from n-1 input nodes (maximum) and work backwards
            target_tested = 0
            target_successful = 0
            
            for r in range(len(input_nodes), 0, -1):  # Start from largest, work backwards, minimum 2 inputs
                combinations_list = list(combinations(input_nodes, r))
                
                for combo_idx, input_combo in enumerate(combinations_list):
                    total_tested += 1
                    target_tested += 1
                    combo_counter += 1
                    
                    try:
                        # Create a completely fresh hypergraph for this specific test
                        # This ensures no state persistence between tests
                        test_hg = timoshenko_subgraph(a_bool, m_bool, s_bool, target_label)
                        
                        # Validate the hypergraph was created properly
                        if not test_hg.nodes or not test_hg.edges:
                            continue
                        
                        # Create input values dictionary with fresh node references
                        input_values = {}
                        for input_label in input_combo:
                            # Find the corresponding node object in the new hypergraph
                            for node in test_hg.nodes.keys():
                                if hasattr(node, 'label') and node.label == input_label:
                                    if input_label in DEFAULT_VALUES:
                                        input_values[node] = DEFAULT_VALUES[input_label]
                                    break
                                elif str(node) == input_label:
                                    if input_label in DEFAULT_VALUES:
                                        input_values[node] = DEFAULT_VALUES[input_label]
                                    break
                        
                        # Find target node in the new hypergraph
                        target_node = None
                        for node in test_hg.nodes.keys():
                            if hasattr(node, 'label') and node.label == target_label:
                                target_node = node
                                break
                            elif str(node) == target_label:
                                target_node = node
                                break
                        
                        # Validate we have everything we need
                        if target_node is None or not input_values or len(input_values) < 2:
                            continue
                        
                        # Additional validation: ensure target node is not in input values
                        if target_node in input_values:
                            continue
                        
                        # Try to solve with timeout using threading
                        result = None
                        success = False
                        
                        def solve_worker(result_queue):
                            try:
                                # Capture stdout to check for "No solutions found"
                                captured_output = io.StringIO()
                                with redirect_stdout(captured_output):
                                    # Ensure we're using fresh references
                                    result = test_hg.solve(target_node, input_values, to_print=True)
                                
                                output_text = captured_output.getvalue()
                                result_queue.put(('success', result, output_text))
                            except Exception as e:
                                result_queue.put(('error', str(e), ''))
                            finally:
                                # Clean up any potential state
                                captured_output.close()
                        
                        # Create queue and thread
                        result_queue = queue.Queue()
                        solve_thread = Thread(target=solve_worker, args=(result_queue,))
                        solve_thread.daemon = True
                        solve_thread.start()
                        
                        # Wait for result with 3-second timeout
                        try:
                            result_type, result_data, output_text = result_queue.get(timeout=3)
                            
                            if result_type == 'success':
                                result = result_data
                                
                                # Check if solve actually worked by looking for "No solutions found"
                                success = False
                                if result is not None and "No solutions found" not in output_text:
                                    success = True  # Only consider successful if no "No solutions found" message
                                
                                if success:
                                    total_successful += 1
                                    target_successful += 1
                                
                        except queue.Empty:
                            pass  # Timeout - skip silently
                        except (ValueError, TypeError):
                            pass  # Error in unpacking - skip silently
                        except Exception as e:
                            pass  # Error - skip silently
                        
                        # Clean up references to ensure no state persistence
                        test_hg = None
                        target_node = None
                        input_values = None
                        result = None
                            
                    except Exception as e:
                        continue
            
        # Store results with exact count and additional info
        results.append([config, total_successful, total_tested])
    
    # Configuration labels for better readability
    config_labels = {
        (0, 0, 0): "Full",
        (0, 0, 1): "No Poisson",
        (0, 1, 0): "No Moment", 
        (0, 1, 1): "No Moment/Radius",
        (1, 0, 0): "No Area",
        (1, 0, 1): "No Poisson/Radius",
        (1, 1, 0): "No Area/Moment",
        (1, 1, 1): "Base"
    }
    
    # Print detailed table
    print(f"{'LABEL':<20} | {'SUCCESSFUL':<12} | {'TOTAL TESTED':<15}")
    print("-" * 50)
    for config, successful, total in results:
        label = config_labels.get(config, "Unknown")
        print(f"{label:<20} | {successful:<12} | {total:<15}")
    
    return results


if __name__ == "__main__":
    print("Timoshenko Beam Subgraph Analysis")
    print("=" * 60)
    print("Running across all subgraph configurations...")
    print()
    
    comprehensive_results = comprehensive_subgraph_analysis()