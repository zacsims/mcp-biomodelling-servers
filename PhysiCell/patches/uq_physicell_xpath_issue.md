# Bug: `_set_xml_element_value` corrupts element text when xpath ends in `[@name='X']` selector

**Package:** uq-physicell
**Version observed:** 1.2.4
**File:** `uq_physicell/pc_model.py`
**Functions affected:** `_set_xml_element_value`, `_get_xml_element_value`

## Summary

When a UQ parameter xpath targets an element identified by a predicate selector of the form `[@name='SomeName']`, the setter incorrectly treats the selector as an **attribute-write target** and overwrites the `name` attribute with the new numeric value, instead of writing to the element's text content.

Consequence: the selector no longer matches on subsequent reads (the element now has a different `name`), and PhysiCell's rule loader silently falls back to the behavior's default of 0 because it can't find a transformation / death_rate / etc. with the expected name. The symptom is that every ABC / UQ particle produces identical simulation output regardless of the substituted parameter.

This silently invalidates any calibration sweep over a rule parameter stored in an attribute-selected XML element — e.g., `cell_definition[@name='tumor']/phenotype/cell_transformations/transformation_rate[@name='motile tumor']`, `phenotype/death/model[@code='100']/death_rate`, and similar.

## Minimal reproducer

```python
import xml.etree.ElementTree as ET
from uq_physicell.pc_model import PhysiCell_Model  # or the private helper

xml = """
<root>
  <transformation_rate name="motile tumor" units="1/min">0.001</transformation_rate>
</root>
"""
root = ET.fromstring(xml)

xpath = "transformation_rate[@name='motile tumor']"
# PhysiCell_Model._set_xml_element_value(root, xpath, 0.002)  # (private, via model setter path)

# Expected: <transformation_rate name="motile tumor" units="1/min">0.002</transformation_rate>
# Actual:   <transformation_rate name="0.002"        units="1/min">0.001</transformation_rate>
#                                  ^^^^^^^^^^^^^^  the name attribute was overwritten,
#                                                  the text remained 0.001
```

You can observe the same behaviour via a full UQ run: `PhysiCell_Model` constructed with any `params_change` keyed on a transformation rate or apoptosis/necrosis death_rate xpath, then run a single particle. Inspect the generated XML in `UQ_PC_InputFolder/config_*.xml` — the `name` attribute will be numeric.

## Root cause

Both `_set_xml_element_value` and `_get_xml_element_value` use `_attr_name_re` to decide whether the terminal xpath segment is an attribute target. That regex matches any `[@something]`, including predicate selectors like `[@name='X']`. The fix is to distinguish a **bare attribute** `[@attr]` (correct attribute target) from a **selector** `[@attr='value']` (not an attribute target — it identifies the element).

## Patch

`fix_xpath_setter.patch` in this directory. Minimal, three-hunk change: introduce a stricter regex `_attr_target_re = re.compile(r"\[@([A-Za-z_:][\w:.\-]*)\]")` that matches only bare `[@attr]` without a value, and route attribute vs text writes through it in both `_set_xml_element_value` and `_get_xml_element_value`.

Verified correct behaviour: selector-style `[@name='X']` falls through to `elem.text`; bare-attribute xpaths like `rate[@index]` still work.

## Suggested upstream issue title

`_set_xml_element_value corrupts element selectors like [@name='X'] by writing to the name attribute`

## Suggested commit message

```
Fix XPath setter: don't treat [@attr='value'] selectors as attribute targets

Previously, _set_xml_element_value and _get_xml_element_value treated any
terminal [@...] bracket as an attribute target. For selector-style paths
like transformation_rate[@name='motile tumor'], the setter would overwrite
the name attribute with the new numeric value instead of writing the
element's text content, causing the selector to no longer match on
subsequent reads and silently zeroing the parameter in PhysiCell runs.

Introduce a stricter regex `_attr_target_re` that matches only bare
[@attr] (no `=`), and use it to route attribute vs text writes in both
the setter and getter. Selector-style predicates fall through to elem.text
as intended.
```

## Who should file

Filed by user of uq-physicell 1.2.4 via `~/mcp-biomodelling-servers/PhysiCell/.venv`. The patch has been validated against the hypoxia-recap blind recapitulation workflow (see parent project) — pre-patch, a rejection-ABC calibration of a hypoxia-induced transformation rate produced zero motile cells in all 50 particles; post-patch, the posterior concentrates correctly and produces a model scoring 0.446 at full horizon (acceptable band) vs 0.705 baseline.

## Where to file

Upstream repository: **https://github.com/heberlr/UQ_PhysiCell**. Open an issue with the content above; attach `fix_xpath_setter.patch` or open a PR with the same change.
