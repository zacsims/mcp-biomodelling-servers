#!/usr/bin/env python3
"""
PhysiCell Configuration Pre-Flight Validator

Checks an exported XML configuration + rules CSV for common issues before
running a simulation. No PhysiCell dependencies required — uses only
standard library (xml.etree, csv).

Usage:
    python validate_config.py PhysiCell_settings.xml [cell_rules.csv]

Exit codes:
    0 — All checks passed
    1 — Warnings found (simulation may have issues)
    2 — Errors found (simulation will likely fail or produce wrong results)
"""

import xml.etree.ElementTree as ET
import csv
import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


class ValidationResult:
    """Collects PASS/WARN/FAIL results."""

    def __init__(self):
        self.passes: List[str] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def passed(self, msg: str):
        self.passes.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def error(self, msg: str):
        self.errors.append(msg)

    def print_report(self):
        print("\n" + "=" * 60)
        print("PhysiCell Configuration Validation Report")
        print("=" * 60)

        if self.passes:
            print(f"\nPASS ({len(self.passes)}):")
            for msg in self.passes:
                print(f"  [PASS] {msg}")

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for msg in self.warnings:
                print(f"  [WARN] {msg}")

        if self.errors:
            print(f"\nERRORS ({len(self.errors)}):")
            for msg in self.errors:
                print(f"  [FAIL] {msg}")

        print("\n" + "-" * 60)
        total = len(self.passes) + len(self.warnings) + len(self.errors)
        print(f"Total checks: {total}")
        print(f"  Passed:   {len(self.passes)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Errors:   {len(self.errors)}")

        if self.errors:
            print("\nRESULT: FAIL — Fix errors before running simulation")
            return 2
        elif self.warnings:
            print("\nRESULT: WARN — Simulation may have issues")
            return 1
        else:
            print("\nRESULT: PASS — Configuration looks good")
            return 0


def parse_xml(xml_path: str) -> Optional[ET.Element]:
    """Parse XML configuration file."""
    try:
        tree = ET.parse(xml_path)
        return tree.getroot()
    except ET.ParseError as e:
        print(f"ERROR: Failed to parse XML: {e}")
        return None
    except FileNotFoundError:
        print(f"ERROR: XML file not found: {xml_path}")
        return None


def parse_rules_csv(csv_path: str) -> List[Dict]:
    """Parse cell rules CSV file.

    Expected format (no header):
    cell_type, signal, direction, behavior, base_value, half_max, hill_power, apply_to_dead
    """
    rules = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                # Skip empty rows and comments
                if not row or row[0].strip().startswith('#') or row[0].strip().startswith('//'):
                    continue
                # Strip whitespace from all fields
                row = [field.strip() for field in row]
                if len(row) >= 7:
                    rules.append({
                        'row': row_num,
                        'cell_type': row[0],
                        'signal': row[1],
                        'direction': row[2],
                        'behavior': row[3],
                        'base_value': float(row[4]),
                        'half_max': float(row[5]),
                        'hill_power': float(row[6]),
                        'apply_to_dead': int(row[7]) if len(row) > 7 else 0,
                    })
                else:
                    rules.append({
                        'row': row_num,
                        'error': f"Row {row_num} has {len(row)} columns, expected at least 7",
                        'raw': row,
                    })
    except FileNotFoundError:
        print(f"WARNING: Rules CSV not found: {csv_path}")
    except Exception as e:
        print(f"ERROR: Failed to parse rules CSV: {e}")
    return rules


def get_cell_types(root: ET.Element) -> Dict[str, ET.Element]:
    """Extract cell type definitions from XML."""
    cell_types = {}
    for cell_def in root.iter('cell_definition'):
        name = cell_def.get('name')
        if name:
            cell_types[name] = cell_def
    return cell_types


def get_substrates(root: ET.Element) -> Dict[str, ET.Element]:
    """Extract substrate definitions from XML."""
    substrates = {}
    for variable in root.iter('variable'):
        name = variable.get('name')
        if name:
            substrates[name] = variable
    return substrates


def get_death_rate(cell_def: ET.Element, death_code: str) -> float:
    """Get death rate for a specific death model from cell definition.

    death_code: '100' for apoptosis, '101' for necrosis
    """
    for model in cell_def.iter('model'):
        if model.get('code') == death_code:
            death_rate = model.find('death_rate')
            if death_rate is not None and death_rate.text:
                try:
                    return float(death_rate.text.strip())
                except ValueError:
                    pass
    return 0.0


def get_cycle_rate(cell_def: ET.Element) -> float:
    """Get the primary cycle transition rate from cell definition."""
    cycle = cell_def.find('.//cycle')
    if cycle is None:
        return 0.0

    # Look for phase_transition_rates or transition_rates
    for rates_elem in cycle.iter('phase_transition_rates'):
        for rate in rates_elem.iter('rate'):
            try:
                return float(rate.text.strip())
            except (ValueError, AttributeError):
                pass

    # Also check transition_rates directly
    for rate in cycle.iter('rate'):
        try:
            val = float(rate.text.strip())
            if val > 0:
                return val
        except (ValueError, AttributeError):
            pass

    return 0.0


def get_motility_speed(cell_def: ET.Element) -> float:
    """Get motility speed from cell definition."""
    speed = cell_def.find('.//motility/speed')
    if speed is not None and speed.text:
        try:
            return float(speed.text.strip())
        except ValueError:
            pass
    return 0.0


def get_secretion_rate(cell_def: ET.Element, substrate_name: str) -> float:
    """Get secretion rate for a substrate from cell definition."""
    for substrate in cell_def.iter('substrate'):
        if substrate.get('name') == substrate_name:
            sec_rate = substrate.find('secretion_rate')
            if sec_rate is not None and sec_rate.text:
                try:
                    return float(sec_rate.text.strip())
                except ValueError:
                    pass
    return 0.0


def get_uptake_rate(cell_def: ET.Element, substrate_name: str) -> float:
    """Get uptake rate for a substrate from cell definition."""
    for substrate in cell_def.iter('substrate'):
        if substrate.get('name') == substrate_name:
            up_rate = substrate.find('uptake_rate')
            if up_rate is not None and up_rate.text:
                try:
                    return float(up_rate.text.strip())
                except ValueError:
                    pass
    return 0.0


def get_behavior_default(cell_def: ET.Element, behavior: str, substrates: Dict) -> float:
    """Get the XML default value for a behavior.

    Returns the value that PhysiCell will use as the saturation value for rules.
    """
    behavior_lower = behavior.lower().strip()

    if behavior_lower in ('apoptosis', 'apoptosis rate'):
        return get_death_rate(cell_def, '100')
    elif behavior_lower in ('necrosis', 'necrosis rate'):
        return get_death_rate(cell_def, '101')
    elif behavior_lower in ('cycle entry', 'cycle entry rate'):
        return get_cycle_rate(cell_def)
    elif behavior_lower in ('migration speed',):
        return get_motility_speed(cell_def)
    elif behavior_lower.endswith(' secretion'):
        substrate = behavior_lower[:-10].strip()
        return get_secretion_rate(cell_def, substrate)
    elif behavior_lower.endswith(' uptake'):
        substrate = behavior_lower[:-7].strip()
        return get_uptake_rate(cell_def, substrate)

    # For other behaviors, we can't easily look up the default
    return -1.0  # sentinel: unknown


# ── Validation checks ────────────────────────────────────────────────────


def check_xml_basics(root: ET.Element, result: ValidationResult):
    """Check basic XML structure."""
    # Check domain exists
    domain = root.find('.//domain')
    if domain is not None:
        result.passed("Domain configuration found")
    else:
        result.error("No <domain> element found in XML")

    # Check cell definitions exist
    cell_types = get_cell_types(root)
    if cell_types:
        result.passed(f"Found {len(cell_types)} cell type(s): {', '.join(cell_types.keys())}")
    else:
        result.error("No cell definitions found in XML")

    # Check substrates exist
    substrates = get_substrates(root)
    if substrates:
        result.passed(f"Found {len(substrates)} substrate(s): {', '.join(substrates.keys())}")
    else:
        result.warn("No substrates found in XML")


def check_dirichlet_boundaries(root: ET.Element, result: ValidationResult):
    """Check Dirichlet boundary settings."""
    substrates = get_substrates(root)

    for name, var_elem in substrates.items():
        # Check for Dirichlet boundary conditions
        bc = var_elem.find('.//Dirichlet_boundary_condition')
        if bc is not None:
            enabled = bc.get('enabled', 'false').lower()
            if enabled == 'true':
                result.passed(f"Substrate '{name}': Dirichlet boundaries enabled")

                # Check individual boundaries
                options = root.find('.//microenvironment_setup/options')
                if options is not None:
                    for boundary in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
                        bc_elem = options.find(f".//Dirichlet_options//{boundary}")
                        if bc_elem is not None:
                            bc_enabled = bc_elem.get('enabled', 'true').lower()
                            if bc_enabled == 'false':
                                result.warn(
                                    f"Substrate '{name}': Dirichlet enabled globally but "
                                    f"'{boundary}' boundary is disabled"
                                )
            elif name == 'oxygen':
                result.warn(
                    f"Substrate 'oxygen': Dirichlet boundaries NOT enabled. "
                    f"Oxygen will be consumed without replenishment."
                )


def check_initial_conditions(root: ET.Element, result: ValidationResult):
    """Check initial conditions settings."""
    ic = root.find('.//initial_conditions')
    if ic is not None:
        cell_positions = ic.find('.//cell_positions')
        if cell_positions is not None:
            enabled = cell_positions.get('enabled', 'false').lower()
            if enabled == 'true':
                folder = cell_positions.find('folder')
                filename = cell_positions.find('filename')
                if folder is not None and filename is not None:
                    csv_path = f"{folder.text}/{filename.text}"
                    result.passed(f"Initial conditions enabled, loading from: {csv_path}")
                else:
                    result.warn("Initial conditions enabled but folder/filename not specified")
            else:
                result.warn(
                    "Initial conditions not enabled (cell_positions enabled='false'). "
                    "Cells may be placed by number_of_cells in cell definitions."
                )
    else:
        result.warn("No <initial_conditions> element found. Check cell placement method.")


def check_cell_rules_enabled(root: ET.Element, result: ValidationResult):
    """Check that cell rules are properly enabled in XML."""
    rules = root.find('.//cell_rules')
    if rules is not None:
        rulesets = rules.findall('.//ruleset')
        for ruleset in rulesets:
            enabled = ruleset.get('enabled', 'false').lower()
            protocol = ruleset.get('protocol', '')
            if enabled == 'true':
                folder = ruleset.find('folder')
                filename = ruleset.find('filename')
                if folder is not None and filename is not None:
                    result.passed(f"Cell rules enabled: {folder.text}/{filename.text}")
                else:
                    result.warn("Cell rules enabled but folder/filename not specified")
            else:
                result.warn(f"Cell rules ruleset found but not enabled (protocol: {protocol})")
    else:
        result.warn("No <cell_rules> element found in XML. Rules CSV won't be loaded.")


def check_rules_from_0_towards_0(
    rules: List[Dict],
    cell_types: Dict[str, ET.Element],
    substrates: Dict[str, ET.Element],
    result: ValidationResult,
):
    """Check for 'from 0 towards 0' rules — THE most common PhysiCell bug."""
    for rule in rules:
        if 'error' in rule:
            result.error(rule['error'])
            continue

        cell_type_name = rule['cell_type']
        behavior = rule['behavior']
        base_value = rule['base_value']

        if cell_type_name not in cell_types:
            result.error(
                f"Row {rule['row']}: Rule references cell type '{cell_type_name}' "
                f"which is not defined in XML. Available: {list(cell_types.keys())}"
            )
            continue

        cell_def = cell_types[cell_type_name]
        xml_default = get_behavior_default(cell_def, behavior, substrates)

        if xml_default == -1.0:
            # Unknown behavior — can't look up the XML default
            # Warn about transition behaviors since they almost always default to 0
            if abs(base_value) < 1e-15 and (
                'transition' in behavior.lower() or
                'transform' in behavior.lower() or
                'exit from' in behavior.lower()
            ):
                result.warn(
                    f"Row {rule['row']}: Possible FROM 0 TOWARDS 0 — "
                    f"Rule '{cell_type_name}: {rule['signal']} {rule['direction']} {behavior}' "
                    f"has base_value=0 and targets '{behavior}' which typically defaults to 0 in XML. "
                    f"Verify that the XML default for this behavior is nonzero, otherwise the rule "
                    f"will have no effect."
                )
            continue

        # Check for "from 0 towards 0"
        if abs(base_value) < 1e-15 and abs(xml_default) < 1e-15:
            result.error(
                f"Row {rule['row']}: FROM 0 TOWARDS 0 — "
                f"Rule '{cell_type_name}: {rule['signal']} {rule['direction']} {behavior}' "
                f"has base_value=0 AND XML default for '{behavior}'=0. "
                f"This rule will have NO EFFECT. "
                f"Fix: Set a nonzero {behavior} rate via configure_cell_parameters() "
                f"before adding this rule."
            )
        elif abs(base_value) < 1e-15 and abs(xml_default) < 1e-10:
            result.warn(
                f"Row {rule['row']}: Near-zero saturation — "
                f"Rule '{cell_type_name}: {rule['signal']} {rule['direction']} {behavior}' "
                f"has base_value=0 and XML default={xml_default:.2e}. "
                f"Effect will be very small."
            )
        elif abs(base_value - xml_default) < 1e-15 and abs(base_value) > 1e-15:
            result.warn(
                f"Row {rule['row']}: base_value equals XML default — "
                f"Rule '{cell_type_name}: {rule['signal']} {rule['direction']} {behavior}' "
                f"has base_value={base_value} ≈ XML default={xml_default}. "
                f"The rule range is near-zero."
            )
        else:
            result.passed(
                f"Row {rule['row']}: Rule '{cell_type_name}: {rule['signal']} "
                f"{rule['direction']} {behavior}' — "
                f"from {base_value} towards {xml_default}"
            )


def check_rules_references(
    rules: List[Dict],
    cell_types: Dict[str, ET.Element],
    substrates: Dict[str, ET.Element],
    result: ValidationResult,
):
    """Check that rules reference existing substrates and cell types."""
    substrate_names = set(substrates.keys())
    cell_type_names = set(cell_types.keys())

    for rule in rules:
        if 'error' in rule:
            continue

        signal = rule['signal']

        # Check if signal is a substrate name
        if signal in substrate_names:
            result.passed(f"Row {rule['row']}: Signal '{signal}' matches a substrate")
        elif signal.startswith('contact with '):
            target = signal[13:]
            if target not in cell_type_names:
                result.warn(
                    f"Row {rule['row']}: Signal '{signal}' references cell type "
                    f"'{target}' which is not defined in XML"
                )
        elif signal in ('pressure', 'volume', 'damage', 'dead', 'time',
                        'apoptotic', 'necrotic', 'attacking', 'total attack time'):
            pass  # built-in signals, always valid
        else:
            # Could be a custom signal; just note it
            pass

        # Check direction
        if rule['direction'] not in ('increases', 'decreases'):
            result.error(
                f"Row {rule['row']}: Invalid direction '{rule['direction']}'. "
                f"Must be 'increases' or 'decreases'."
            )

        # Check half_max > 0
        if rule['half_max'] <= 0:
            result.error(
                f"Row {rule['row']}: half_max={rule['half_max']} must be positive"
            )

        # Check hill_power > 0
        if rule['hill_power'] <= 0:
            result.error(
                f"Row {rule['row']}: hill_power={rule['hill_power']} must be positive"
            )


def check_rules_csv_format(csv_path: str, result: ValidationResult):
    """Check for CRLF line endings in rules CSV."""
    try:
        with open(csv_path, 'rb') as f:
            content = f.read()
            if b'\r\n' in content:
                result.warn(
                    f"Rules CSV has Windows-style line endings (CRLF). "
                    f"PhysiCell may have issues parsing cell type names with trailing \\r. "
                    f"The MCP server should handle this, but verify if cell types aren't found."
                )
            elif b'\r' in content:
                result.warn("Rules CSV has old Mac-style line endings (CR only)")
            else:
                result.passed("Rules CSV has Unix-style line endings (LF)")
    except FileNotFoundError:
        pass  # Already reported elsewhere


# ── Main ──────────────────────────────────────────────────────────────────


def validate(xml_path: str, csv_path: Optional[str] = None) -> int:
    """Run all validation checks.

    Returns exit code: 0=pass, 1=warnings, 2=errors.
    """
    result = ValidationResult()

    # Parse XML
    root = parse_xml(xml_path)
    if root is None:
        result.error(f"Cannot parse XML file: {xml_path}")
        return result.print_report()

    # Basic XML checks
    check_xml_basics(root, result)

    # Get cell types and substrates for cross-referencing
    cell_types = get_cell_types(root)
    substrates = get_substrates(root)

    # Dirichlet boundaries
    check_dirichlet_boundaries(root, result)

    # Initial conditions
    check_initial_conditions(root, result)

    # Cell rules enabled in XML
    check_cell_rules_enabled(root, result)

    # Parse and check rules CSV
    if csv_path:
        rules = parse_rules_csv(csv_path)
        if rules:
            result.passed(f"Parsed {len(rules)} rule(s) from CSV")

            # Check CSV format
            check_rules_csv_format(csv_path, result)

            # Check for "from 0 towards 0"
            check_rules_from_0_towards_0(rules, cell_types, substrates, result)

            # Check rule references
            check_rules_references(rules, cell_types, substrates, result)
        else:
            result.warn("No rules found in CSV file")
    else:
        # Try to find rules CSV from XML
        rules_elem = root.find('.//cell_rules//ruleset//folder')
        rules_file = root.find('.//cell_rules//ruleset//filename')
        if rules_elem is not None and rules_file is not None:
            inferred_csv = os.path.join(
                os.path.dirname(xml_path),
                rules_elem.text or '',
                rules_file.text or ''
            )
            if os.path.exists(inferred_csv):
                result.passed(f"Found rules CSV at: {inferred_csv}")
                rules = parse_rules_csv(inferred_csv)
                if rules:
                    check_rules_csv_format(inferred_csv, result)
                    check_rules_from_0_towards_0(rules, cell_types, substrates, result)
                    check_rules_references(rules, cell_types, substrates, result)
            else:
                result.warn(f"Rules CSV referenced in XML but not found: {inferred_csv}")

    return result.print_report()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    xml_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else None

    exit_code = validate(xml_path, csv_path)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
