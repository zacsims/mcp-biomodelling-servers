import asyncio
from fastmcp import Client

async def main():
    client = Client("server.py")
    async with client:
        result = await client.call_tool("create_session", {"session_name": "breast_cancer_hypoxia_immune", "set_as_default": "true"})
        print(f"output: {result}")

        biological_scenario =  'Breast cancer cells growing in a 3D hypoxic tumor microenvironment with immune cell infiltration. The tumor experiences low oxygen conditions (hypoxia) which affects both cancer cell behavior and immune cell function. Immune cells (cytotoxic T cells) infiltrate the tumor to attack cancer cells. Hypoxia may reduce immune cell effectiveness and increase cancer cell survival and aggression.'

        result = await client.call_tool("analyze_biological_scenario", {"biological_scenario": biological_scenario})
        print(f"output: {result}")

        result = await client.call_tool("create_simulation_domain", {"domain_x":2000, "domain_y":2000, "domain_z":2000, "dx":20, "max_time":10080})
        print(f"output: {result}")

        result = await client.call_tool("add_single_substrate", {'units': 'mmHg','decay_rate': 0.1, 'substrate_name': 'oxygen', 'dirichlet_value': 10, 'dirichlet_enabled': True, 'initial_condition': 10, 'diffusion_coefficient': 100000})
        print(f"output: {result}")

        result = await client.call_tool("add_single_substrate", {'units': 'dimensionless', 'decay_rate': 0.01, 'substrate_name': 'glucose', 'dirichlet_value': 1, 'dirichlet_enabled': True, 'initial_condition': 1, 'diffusion_coefficient': 80000})
        print(f"output: {result}")

        result = await client.call_tool("add_single_cell_type", {'cycle_model': 'Ki67_basic', 'cell_type_name': 'cytotoxic_T_cell'})
        print(f"output: {result}")

        result = await client.call_tool("add_single_cell_type", {'cycle_model': 'Ki67_basic', 'cell_type_name': 'breast_cancer'})
        print(f"output: {result}")

        result = await client.call_tool("configure_cell_parameters", {'cell_type': 'cytotoxic_T_cell', 'volume_total': 1000, 'necrosis_rate': 0.00001, 'apoptosis_rate': 0.0001, 'fluid_fraction': 0.75, 'motility_speed': 2,'volume_nuclear': 250, 'persistence_time': 5})
        print(f"output: {result}")

        result = await client.call_tool("configure_cell_parameters", {'cell_type': 'breast_cancer', 'volume_total': 2500, 'necrosis_rate': 0.0001, 'apoptosis_rate': 0.00001, 'fluid_fraction': 0.75, 'motility_speed': 0.2, 'volume_nuclear': 500, 'persistence_time': 10})
        print(f"output: {result}")

        result = await client.call_tool("set_substrate_interaction", {'cell_type': 'breast_cancer', 'substrate': 'glucose', 'uptake_rate': 5, 'secretion_rate': 0})
        print(f"output: {result}")

        result = await client.call_tool("set_substrate_interaction", {'cell_type': 'cytotoxic_T_cell', 'substrate': 'oxygen', 'uptake_rate': 20, 'secretion_rate': 0})
        print(f"output: {result}")

        result = await client.call_tool("list_all_available_signals", {})
        print(f"output: {result}")

        result = await client.call_tool("add_single_cell_rule", {'signal': 'oxygen', 'behavior': 'cycle entry', 'half_max': 5, 'cell_type': 'breast_cancer', 'direction': 'increases', 'hill_power': 4, 'max_signal': 20, 'min_signal': 0})
        print(f"output: {result}")

        result = await client.call_tool("add_single_cell_rule", {'signal': 'oxygen', 'behavior': 'necrosis', 'half_max': 3, 'cell_type': 'breast_cancer', 'direction': 'decreases', 'hill_power': 4, 'max_signal': 15, 'min_signal': 0})
        print(f"output: {result}")

        result = await client.call_tool("add_single_cell_rule", {'signal': 'oxygen', 'behavior': 'migration speed', 'half_max': 8, 'cell_type': 'breast_cancer', 'direction': 'decreases', 'hill_power': 3, 'max_signal': 20, 'min_signal': 0})
        print(f"output: {result}")

        result = await client.call_tool("add_single_cell_rule", {'signal': 'oxygen', 'behavior': 'chemotactic response to oxygen', 'half_max': 5, 'cell_type': 'breast_cancer', 'direction': 'decreases', 'hill_power': 2, 'max_signal': 15, 'min_signal': 0})
        print(f"output: {result}")

        result = await client.call_tool("add_single_cell_rule", {'signal': 'contact with breast_cancer', 'behavior': 'attack breast_cancer', 'half_max': 0.5, 'cell_type': 'cytotoxic_T_cell', 'direction': 'increases', 'hill_power': 4, 'max_signal': 1, 'min_signal': 0})
        print(f"output: {result}")

        result = await client.call_tool("add_single_cell_rule", {'signal': 'oxygen', 'behavior': 'attack damage rate', 'half_max': 8, 'cell_type': 'cytotoxic_T_cell', 'direction': 'increases', 'hill_power': 3, 'max_signal': 20, 'min_signal': 0})
        print(f"output: {result}")

        result = await client.call_tool("add_single_cell_rule", {'signal': 'oxygen', 'behavior': 'migration speed', 'half_max': 10, 'cell_type': 'cytotoxic_T_cell', 'direction': 'increases', 'hill_power': 2, 'max_signal': 20, 'min_signal': 0})
        print(f"output: {result}")

        result = await client.call_tool("add_single_cell_rule", {'signal': 'damage', 'behavior': 'apoptosis', 'half_max': 50, 'cell_type': 'breast_cancer', 'direction': 'increases', 'hill_power': 4, 'max_signal': 100, 'min_signal': 0})
        print(f"output: {result}")

        result = await client.call_tool("get_simulation_summary", {})
        print(f"output: {result}")

        result = await client.call_tool("export_xml_configuration", {'filename': 'breast_cancer_hypoxia_immune.xml'})
        print(f"output: {result}")

        result = await client.call_tool("export_cell_rules_csv", {'filename': 'breast_cancer_cell_rules.csv'})
        print(f"output: {result}")

        # ================================================================
        # Test new project creation and execution tools
        # ================================================================

        print("\n" + "="*60)
        print("Testing Project Creation and Execution Tools")
        print("="*60 + "\n")

        # Create a PhysiCell project
        result = await client.call_tool("create_physicell_project", {'project_name': 'test_breast_cancer_sim'})
        print(f"create_physicell_project output: {result}")

        # Compile the project
        result = await client.call_tool("compile_physicell_project", {'project_name': 'test_breast_cancer_sim', 'clean_first': True})
        print(f"compile_physicell_project output: {result}")

        # List simulations (should be empty initially)
        result = await client.call_tool("list_simulations", {})
        print(f"list_simulations output: {result}")

        # Note: Running simulation is a long operation, uncomment to test
        result = await client.call_tool("run_simulation", {'project_name': 'test_breast_cancer_sim'})
        print(f"run_simulation output: {result}")

        # Extract simulation_id from result
        import re
        text = result.content[0].text
        match = re.search(r'\*\*Simulation ID:\*\* ([a-f0-9]+)', text)
        simulation_id = match.group(1) if match else None
        print(f"Extracted simulation_id: {simulation_id}")

        # Poll simulation status until completed or timeout
        if simulation_id:
            max_wait = 300  # 5 minutes max
            poll_interval = 10  # check every 10 seconds
            elapsed = 0

            while elapsed < max_wait:
                result = await client.call_tool("get_simulation_status", {'simulation_id': simulation_id})
                status_text = result.content[0].text

                # Extract status from response
                if "**Status:** completed" in status_text:
                    print(f"Simulation completed after {elapsed}s!")
                    print(f"get_simulation_status output: {result}")
                    break
                elif "**Status:** failed" in status_text or "**Status:** stopped" in status_text:
                    print(f"Simulation ended with non-success status after {elapsed}s")
                    print(f"get_simulation_status output: {result}")
                    break

                # Extract file counts for progress indication
                svg_match = re.search(r'SVG snapshots: (\d+)', status_text)
                svg_count = svg_match.group(1) if svg_match else "?"
                print(f"Still running... ({elapsed}s elapsed, {svg_count} SVG files)")

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
            else:
                print(f"Timeout after {max_wait}s - simulation still running")
                result = await client.call_tool("get_simulation_status", {'simulation_id': simulation_id})
                print(f"Final status: {result}")

        # Test output file listing
        result = await client.call_tool("get_simulation_output_files", {'output_folder': '/Users/simsz/PhysiCell/output'})
        print(f"get_simulation_output_files output: {result}")

        # Test GIF generation
        result = await client.call_tool("generate_simulation_gif", {'output_folder': '/Users/simsz/PhysiCell/output', 'max_frames': 10})
        print(f"generate_simulation_gif output: {result}")


if __name__ == "__main__":
    asyncio.run(main())
