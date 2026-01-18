#!/usr/bin/env python3
"""
Nu1lm Microplastics Knowledge Base

Comprehensive knowledge about microplastics for training and RAG.
This serves as the core knowledge that Nu1lm learns from.
"""

MICROPLASTICS_KNOWLEDGE = {

    # ===================
    # FUNDAMENTALS
    # ===================
    "definition": """
Microplastics are plastic particles smaller than 5mm in diameter. They are classified as:
- Large microplastics: 1-5 mm
- Small microplastics: 1 µm - 1 mm
- Nanoplastics: < 1 µm (emerging concern)

Primary microplastics: Manufactured small (microbeads, nurdles, fibers)
Secondary microplastics: Degradation of larger plastics
""",

    "plastic_types": """
Common microplastic polymers and their properties:

1. Polyethylene (PE) - Most common (bags, bottles, packaging)
   - Density: 0.91-0.96 g/cm³ (floats in seawater)
   - Raman peaks: 1062, 1128, 1295, 1440 cm⁻¹
   - FTIR: 2915, 2848, 1472, 730 cm⁻¹

2. Polypropylene (PP) - Second most common (containers, caps)
   - Density: 0.90-0.91 g/cm³ (floats)
   - Raman peaks: 808, 841, 1458 cm⁻¹
   - FTIR: 2950, 2916, 1455, 1376 cm⁻¹

3. Polystyrene (PS) - Food containers, foam
   - Density: 1.04-1.06 g/cm³ (sinks slowly)
   - Raman peaks: 1001 (diagnostic), 1602, 3054 cm⁻¹
   - FTIR: 3026, 1493, 757, 699 cm⁻¹

4. Polyethylene Terephthalate (PET) - Bottles, fibers
   - Density: 1.38-1.40 g/cm³ (sinks)
   - Raman peaks: 1614, 1726, 858 cm⁻¹
   - FTIR: 1714, 1245, 1097 cm⁻¹

5. Polyvinyl Chloride (PVC) - Pipes, packaging
   - Density: 1.3-1.45 g/cm³ (sinks)
   - Raman peaks: 636, 694 cm⁻¹ (C-Cl diagnostic)
   - FTIR: 2912, 1427, 1254, 690 cm⁻¹

6. Polyamide/Nylon (PA) - Fishing nets, textiles
   - Density: 1.13-1.15 g/cm³ (sinks)
   - Raman peaks: 1635, 3300 cm⁻¹ (amide)
   - FTIR: 3300, 1640, 1545 cm⁻¹

7. Polymethyl Methacrylate (PMMA) - Acrylic
   - Density: 1.17-1.20 g/cm³ (sinks)
   - Raman peaks: 812, 1730 cm⁻¹
   - FTIR: 1730, 1450, 1145 cm⁻¹
""",

    "morphologies": """
Microplastic morphology classification:

1. Fibers/Filaments
   - Aspect ratio > 3:1
   - Sources: Textiles, fishing gear, ropes
   - Most common in wastewater
   - Often nylon, polyester, acrylic

2. Fragments
   - Irregular shape, sharp edges
   - From breakdown of larger plastics
   - Variable polymer types
   - Dominant in beach/sediment samples

3. Films/Sheets
   - Thin, flexible, low circularity
   - From bags, packaging
   - Often PE or PP
   - Easily fragmented further

4. Spheres/Beads
   - High circularity (>0.85)
   - Primary microplastics
   - Cosmetics, industrial abrasives
   - Often PE, PS

5. Foam
   - Porous, low density
   - Expanded polystyrene (EPS)
   - Food packaging, insulation
   - Fragmentsto small particles

6. Pellets/Nurdles
   - Pre-production plastic
   - Cylindrical, 2-5 mm
   - Industrial spillage
   - Various polymers
""",

    # ===================
    # DETECTION METHODS
    # ===================
    "raman_spectroscopy": """
Raman Spectroscopy for Microplastic Identification:

Principle: Inelastic scattering of light revealing molecular vibrations.

Advantages:
- Non-destructive
- Works on particles >1 µm
- Chemical identification
- Can work through water
- Minimal sample preparation

Limitations:
- Fluorescence interference (common with environmental samples)
- Time-consuming for many particles
- Requires clean samples
- Dark/black particles absorb laser

Key diagnostic peaks (cm⁻¹):
- PE: 1295 (CH₂ twist), 1440 (CH₂ bend)
- PP: 841 (CH₃ rock), 1458 (CH₂ deform)
- PS: 1001 (ring breathing) - DIAGNOSTIC
- PET: 1614 (ring), 1726 (C=O)
- PVC: 636, 694 (C-Cl stretch)
- PA: 1635 (amide I), 3300 (N-H)

Best practices:
1. Use 785nm laser to reduce fluorescence
2. Bleach samples with H₂O₂ to remove organics
3. Multiple accumulations for weak signals
4. Reference library matching (>70% match = positive ID)
""",

    "ftir_spectroscopy": """
FTIR Spectroscopy for Microplastic Analysis:

Modes:
1. ATR-FTIR: Particles >500 µm, quick analysis
2. µ-FTIR: Particles 10-500 µm, slower but detailed
3. Focal Plane Array (FPA): Imaging of filter surfaces

Advantages:
- Established, reliable technique
- Less fluorescence issues than Raman
- Good for weathered samples
- High throughput with FPA

Limitations:
- Particle must contact crystal (ATR)
- Water interference
- Minimum size ~10 µm for µ-FTIR
- Sample preparation important

Key regions:
- 3000-2800 cm⁻¹: C-H stretching
- 1800-1600 cm⁻¹: C=O, C=C
- 1500-1300 cm⁻¹: CH₂, CH₃ bending
- 1300-900 cm⁻¹: C-O, fingerprint region
- <900 cm⁻¹: characteristic vibrations

Carbonyl Index (weathering):
CI = A(1715) / A(1380)  for PE
Higher CI = more degraded
""",

    "fluorescence_microscopy": """
Fluorescence Microscopy for Microplastic Detection:

Nile Red Staining Protocol:
1. Prepare Nile Red stock: 1 mg/mL in acetone
2. Working solution: 10 µg/mL in n-hexane or water
3. Stain samples 30 min at room temperature
4. Rinse to remove excess dye
5. Image under fluorescence microscope

Excitation/Emission:
- Green/yellow: 450-490 nm ex / 515-565 nm em
- Orange/red: 510-560 nm ex / >590 nm em
- PE, PP, PS: strong yellow-orange fluorescence
- PA, PET: weaker, more red-shifted

Advantages:
- Fast screening of large areas
- Works on small particles (<10 µm)
- Semi-automated counting possible
- Good for environmental samples

Limitations:
- Not specific to polymer type
- Organic matter can fluoresce
- Requires confirmation with spectroscopy
- Staining can be inconsistent

Image Analysis:
1. Background subtraction
2. Thresholding (Otsu method)
3. Particle detection (watershed for touching)
4. Size and shape measurement
5. Morphology classification
""",

    "uv_vis_spectroscopy": """
UV-Vis Spectroscopy Applications:

Direct UV-Vis:
- Limited use for identification
- Can detect aromatic plastics (PS, PET, PC): 250-280 nm
- Monitors plastic degradation (increase in absorption)

With Nile Red:
- Quantification method
- Absorption at 552 nm (aqueous) or 580 nm (organic)
- Can create calibration curves
- Estimates particle concentration

Degradation Monitoring:
- Carbonyl formation: absorption increase 260-280 nm
- Yellowing index: absorption at 380 nm vs 500 nm
- Total UV absorption: integral 300-400 nm

Advantages:
- Simple, widely available
- Good for concentration estimates
- Monitors weathering
- High-throughput

Limitations:
- Cannot identify specific polymers
- Interference from dissolved organics
- Needs extraction/staining for sensitivity
""",

    "pyrolysis_gcms": """
Pyrolysis GC-MS for Microplastics:

Principle: Thermal decomposition followed by chromatographic separation and mass spectrometric identification.

Advantages:
- Mass-based quantification
- Polymer AND additive identification
- Works on nanoplastics
- No interference from particle size/color
- Identifies specific polymer grades

Limitations:
- Destructive method
- Complex data interpretation
- Equipment expensive
- Can't provide particle count
- Matrix effects possible

Pyrolysis products:
- PE: alkenes, alkanes (C₆-C₃₀)
- PP: 2,4-dimethyl-1-heptene
- PS: styrene monomer/dimer/trimer
- PET: benzoic acid, acetaldehyde
- PVC: HCl, benzene, toluene
- PA: caprolactam (PA6)

Thermoextraction-Desorption GC-MS (TED-GC-MS):
- Lower temperature, gentler
- Better for additives
- Combined with thermogravimetric analysis
""",

    # ===================
    # SAMPLE PREPARATION
    # ===================
    "sample_preparation": """
Sample Preparation for Microplastic Analysis:

1. SAMPLE COLLECTION
Water:
- Manta/neuston nets: surface (>300 µm)
- Plankton nets: water column
- Pumping/filtration: smaller particles
- Grab samples: discrete volumes

Sediment:
- Grab samplers, cores
- Note: top 5 cm for recent deposition
- Wet sieve to remove large debris

Biota:
- Whole organism or tissues
- GI tract for ingestion studies
- Need tissue digestion

2. DENSITY SEPARATION
Principle: Float plastics away from denser sediment

Solutions:
- NaCl (1.2 g/cm³): floats PE, PP only
- ZnCl₂ (1.5-1.8 g/cm³): floats all common plastics
- NaI (1.6-1.8 g/cm³): alternative to ZnCl₂
- ZnBr₂: high density, expensive

Protocol:
1. Add sample to density solution
2. Stir/shake vigorously
3. Allow settling (2-24 hours)
4. Collect supernatant
5. Repeat 2-3 times
6. Filter through appropriate mesh

3. ORGANIC MATTER REMOVAL
Critical for spectroscopic analysis

Methods:
- H₂O₂ (30%): 24-48h at 50-60°C (gentle)
- Fenton's reagent (H₂O₂ + Fe): faster, more aggressive
- KOH (10%): good for biota, attacks some plastics
- HNO₃: aggressive, may damage plastics
- Enzymatic: specific, gentle, expensive

Recommended:
- 30% H₂O₂ at 50°C for 24-48h
- Repeat until solution clear
- Works well with wet peroxide oxidation (WPO)

4. FILTRATION
- Membrane filters: 0.45-5 µm pore size
- Metal or glass fiber filters for Raman
- Polycarbonate for SEM
- Avoid plastic filters!

5. QUALITY CONTROL
- Procedural blanks (every 5-10 samples)
- Positive controls (spiked samples)
- Clean air handling (laminar flow)
- Cotton lab coats (not synthetic)
- Cover samples during processing
""",

    # ===================
    # ENVIRONMENTAL OCCURRENCE
    # ===================
    "environmental_occurrence": """
Global Microplastic Distribution:

MARINE ENVIRONMENT:
Surface water:
- Open ocean: 0.1-100 particles/m³
- Coastal: 10-1000 particles/m³
- Gyres: accumulation zones (1000+ particles/m³)
- Great Pacific Garbage Patch: highest concentrations

Deep sea:
- Found at 10,000m depth
- Sediment concentrations: 50-4000 particles/kg
- Dominant: fibers (textile origin)

Beaches:
- Sandy beaches: 10-10,000 particles/kg
- High tide line: maximum accumulation
- Near urban areas: highest loads

FRESHWATER:
Rivers:
- Major transport pathway to oceans
- 0.4-7000 particles/m³ reported
- Higher near wastewater outfalls
- Fibers dominate

Lakes:
- Surface: 0.5-100 particles/m³
- Sediments: accumulation over time
- Great Lakes: well-studied

TERRESTRIAL:
Soils:
- Agricultural: 300-67,500 particles/kg
- Sources: sewage sludge, mulch films, irrigation
- Can affect soil biota

Air:
- Indoor: 1-60 fibers/m³
- Outdoor: 0.3-1.5 fibers/m³
- Atmospheric transport possible

BIOTA:
- Found in >700 species
- Fish: 1-7 particles/individual common
- Bivalves: filter feeders accumulate
- Zooplankton: can ingest nanoplastics
- Humans: 0.1-5 g/week estimated intake
""",

    "sources": """
Microplastic Sources:

PRIMARY SOURCES (manufactured small):
1. Cosmetics/personal care products
   - Microbeads in scrubs, toothpaste (being phased out)
   - 4,000-95,000 beads per container

2. Industrial abrasives
   - Sandblasting media
   - Cleaning products

3. Pre-production pellets (nurdles)
   - Industrial spillage
   - Transport losses

4. Textile fibers (during manufacturing)

SECONDARY SOURCES (degradation):
1. Plastic litter breakdown
   - UV degradation
   - Mechanical fragmentation
   - Biological degradation (slow)

2. Tire wear particles
   - Major urban source
   - 0.8 kg/capita/year estimated
   - Contains styrene-butadiene rubber

3. Road markings
   - Paint particles
   - Thermoplastic markings

4. Synthetic textiles (washing)
   - 700,000+ fibers per wash
   - Polyester, acrylic, nylon
   - Passes through WWTPs

5. Fishing gear
   - Lost/discarded nets
   - Ghost fishing continues

6. Agriculture
   - Mulch films
   - Greenhouse plastics
   - Slow-release fertilizer coatings

PATHWAYS TO ENVIRONMENT:
- Wastewater treatment plants (90%+ removal, but...)
- Stormwater runoff (unfiltered)
- Direct littering
- Industrial discharge
- Atmospheric deposition
""",

    # ===================
    # HEALTH & ECOLOGY
    # ===================
    "ecological_effects": """
Ecological Effects of Microplastics:

INGESTION:
Physical effects:
- False satiation (reduced feeding)
- Gut blockage
- Internal abrasion
- Reduced energy reserves

Chemical effects:
- Plastic additive leaching (phthalates, BPA)
- Adsorbed pollutant transfer (POPs, metals)
- Oxidative stress

Observed in:
- Zooplankton: reduced feeding, reproduction
- Bivalves: inflammation, reduced filtration
- Fish: liver toxicity, behavioral changes
- Seabirds: highest plastic loads
- Marine mammals: documented ingestion

TROPHIC TRANSFER:
- Demonstrated in lab studies
- Field evidence growing
- Biomagnification of associated chemicals
- Nanoplastics cross membranes

ECOSYSTEM EFFECTS:
- Altered sediment properties
- Microbial community changes ("plastisphere")
- Potential pathogen transport
- Invasive species rafting

TERRESTRIAL:
- Soil structure alteration
- Earthworm effects (reduced burrowing)
- Plant uptake of nanoplastics shown
- Soil microbiome changes
""",

    "human_health": """
Human Health Implications:

EXPOSURE ROUTES:
1. Ingestion
   - Drinking water: 0-61 particles/L (tap), 0-6000 (bottled)
   - Seafood: 0-10 particles/g
   - Salt: 550-680 particles/kg
   - Beer, honey, air, etc.
   - Estimated: 39,000-52,000 particles/year

2. Inhalation
   - Indoor air contamination
   - Textile fibers dominant
   - Occupational exposure (plastic workers)
   - Estimated: 74,000-121,000 particles/year

3. Dermal (limited evidence)
   - Cosmetic products
   - Synthetic textiles

POTENTIAL HEALTH EFFECTS (research ongoing):
Physical:
- Particle accumulation in tissues
- Inflammatory response
- Oxidative stress

Chemical:
- Endocrine disruption (BPA, phthalates)
- Carcinogenic additives
- Adsorbed pollutants

Observed:
- Microplastics found in human blood (2022)
- Found in placenta, breast milk
- Found in lung tissue
- Stool samples: average 20 particles/10g

REGULATORY STATUS:
- WHO: no evidence of harm from drinking water (as of 2019)
- EFSA: more research needed
- No established safety limits
- Precautionary principle advocated
""",

    # ===================
    # RESEARCH METHODS
    # ===================
    "quantification": """
Microplastic Quantification Methods:

PARTICLE COUNTING:
Visual/microscopy:
- Stereomicroscope: >100 µm
- Fluorescence microscopy: >1 µm with staining
- SEM: nanoplastics visible
- Units: particles/L, particles/kg, particles/m²

Automated:
- FlowCam: imaging flow cytometry
- Coulter counter: size distribution
- Image analysis software

MASS-BASED:
Gravimetric:
- Weigh collected particles
- Not polymer-specific
- Minimum detection limits high

Py-GC/MS:
- Quantitative for each polymer
- µg/L or mg/kg units
- Requires calibration curves

Thermoextraction:
- Similar to Py-GC/MS
- Combined with TGA

REPORTING UNITS:
- Particles per volume (water): particles/L, particles/m³
- Particles per mass (sediment, biota): particles/kg, particles/g
- Particles per area: particles/m², particles/km²
- Mass concentration: µg/L, mg/kg

QUALITY ASSURANCE:
1. Blanks: minimum 3 per batch
2. Positive controls: spiked samples
3. Recovery tests
4. Interlaboratory comparisons
5. Report: size range, morphology, polymer type
""",

    "data_analysis": """
Data Analysis in Microplastic Research:

STATISTICAL APPROACHES:
- High variability typical (CV > 100% common)
- Non-normal distributions
- Use non-parametric tests
- Log-transform for normalization
- Report median and range, not just mean

SPATIAL ANALYSIS:
- GIS mapping of contamination
- Source identification
- Transport modeling
- Hot spot identification

RISK ASSESSMENT:
Components:
1. Hazard identification (what effects?)
2. Exposure assessment (how much?)
3. Dose-response (at what levels?)
4. Risk characterization (probability of harm?)

Challenges:
- No established dose-response for particles
- Chemical mixture effects
- Size-dependent toxicity
- Difficult to define "safe" levels

SPECTRAL ANALYSIS:
Raman/FTIR interpretation:
- Library matching (commercial: SLOPP, Open Specy)
- Hit Quality Index (HQI) > 70% typical threshold
- Manual verification important
- Weathering affects spectra

Machine learning:
- Random forests for polymer classification
- Neural networks for image analysis
- Reducing manual analysis time
- Training data quality critical
""",

    # ===================
    # CURRENT RESEARCH
    # ===================
    "current_research": """
Current Frontiers in Microplastic Research (2024):

NANOPLASTICS:
- Detection methods developing
- Environmental concentrations unknown
- Health effects priority
- Cross membrane barriers

ATMOSPHERIC TRANSPORT:
- Long-range transport confirmed
- Found in remote mountain snow
- Arctic/Antarctic contamination
- Deposition rates being measured

BIODEGRADATION:
- Microbial degradation slow but possible
- Enzyme engineering (PETase)
- Wax worm gut bacteria
- In-situ degradation rates unknown

STANDARDIZATION:
- ISO methods in development
- Need for reference materials
- Interlaboratory harmonization
- Quality assurance guidelines

POLICY:
- EU Single-Use Plastics Directive
- Microplastic bans (cosmetics)
- Extended producer responsibility
- Circular economy initiatives

EMERGING CONCERNS:
- Tire wear particles (major source)
- Textile fibers (laundry mitigation)
- Agricultural plastics
- COVID-19 PPE pollution

SOLUTIONS RESEARCH:
- Laundry filters (70-80% reduction)
- Improved WWTP removal
- Biodegradable alternatives
- Source reduction strategies
""",

    "key_publications": """
Key Publications in Microplastics (for RAG training):

FOUNDATIONAL:
- Thompson et al. (2004) Science - "Lost at Sea" - coined microplastics
- Browne et al. (2011) ES&T - Textiles as source
- Jambeck et al. (2015) Science - Plastic input to oceans
- Geyer et al. (2017) Science Advances - Global plastic production

DETECTION METHODS:
- Löder & Gerdts (2015) - Sample preparation
- Primpke et al. (2017) - µ-FTIR imaging
- Araujo et al. (2018) - Py-GC/MS methods
- Maes et al. (2017) - Nile Red staining

ENVIRONMENTAL:
- Eriksen et al. (2014) - Ocean surface plastics
- Van Cauwenberghe et al. (2015) - Deep sea MPs
- Allen et al. (2019) - Atmospheric transport
- Horton et al. (2017) - Freshwater review

BIOLOGICAL EFFECTS:
- Wright et al. (2013) - Review of effects
- Lusher et al. (2017) - Microplastics in fish
- Galloway et al. (2017) - Interactions with organisms
- Leslie et al. (2022) - Microplastics in human blood

REVIEWS:
- Rochman et al. (2019) - Rethinking microplastics
- Hartmann et al. (2019) - Are we speaking same language?
- Bucci et al. (2020) - What's known about effects?
- Koelmans et al. (2022) - Risk assessment
""",

}


def get_knowledge_text() -> str:
    """Get all knowledge as a single text for training."""
    parts = []
    for topic, content in MICROPLASTICS_KNOWLEDGE.items():
        parts.append(f"## {topic.replace('_', ' ').title()}\n{content.strip()}")
    return "\n\n".join(parts)


def get_training_prompts() -> list:
    """Generate Q&A pairs for fine-tuning."""
    qa_pairs = [
        # Fundamentals
        {"instruction": "What are microplastics?", "output": MICROPLASTICS_KNOWLEDGE["definition"]},
        {"instruction": "What are the main types of plastic polymers found as microplastics?", "output": MICROPLASTICS_KNOWLEDGE["plastic_types"]},
        {"instruction": "How are microplastic particles classified by shape?", "output": MICROPLASTICS_KNOWLEDGE["morphologies"]},

        # Detection
        {"instruction": "How do you identify microplastics using Raman spectroscopy?", "output": MICROPLASTICS_KNOWLEDGE["raman_spectroscopy"]},
        {"instruction": "What are the key Raman peaks for polystyrene?", "output": "Polystyrene has a diagnostic peak at 1001 cm⁻¹ (aromatic ring breathing mode). Other characteristic peaks include 620, 795, 1031, 1155, 1583, 1602, and 3054 cm⁻¹."},
        {"instruction": "How is FTIR used to analyze microplastics?", "output": MICROPLASTICS_KNOWLEDGE["ftir_spectroscopy"]},
        {"instruction": "How do you use fluorescence microscopy with Nile Red to detect microplastics?", "output": MICROPLASTICS_KNOWLEDGE["fluorescence_microscopy"]},
        {"instruction": "What is Py-GC/MS and how is it used for microplastics?", "output": MICROPLASTICS_KNOWLEDGE["pyrolysis_gcms"]},

        # Sample prep
        {"instruction": "How do you prepare environmental samples for microplastic analysis?", "output": MICROPLASTICS_KNOWLEDGE["sample_preparation"]},
        {"instruction": "What density solutions are used for microplastic separation?", "output": "Common density solutions include: NaCl (1.2 g/cm³) for PE and PP only, ZnCl₂ (1.5-1.8 g/cm³) for all common plastics, NaI (1.6-1.8 g/cm³) as an alternative, and ZnBr₂ for very high density separation."},

        # Environment
        {"instruction": "Where are microplastics found in the environment?", "output": MICROPLASTICS_KNOWLEDGE["environmental_occurrence"]},
        {"instruction": "What are the sources of microplastics?", "output": MICROPLASTICS_KNOWLEDGE["sources"]},

        # Health
        {"instruction": "What are the ecological effects of microplastics?", "output": MICROPLASTICS_KNOWLEDGE["ecological_effects"]},
        {"instruction": "What are the potential human health effects of microplastics?", "output": MICROPLASTICS_KNOWLEDGE["human_health"]},

        # Methods
        {"instruction": "How do you quantify microplastics in samples?", "output": MICROPLASTICS_KNOWLEDGE["quantification"]},
        {"instruction": "What statistical methods are used in microplastic research?", "output": MICROPLASTICS_KNOWLEDGE["data_analysis"]},

        # Current research
        {"instruction": "What are the current frontiers in microplastic research?", "output": MICROPLASTICS_KNOWLEDGE["current_research"]},

        # Specific technical questions
        {"instruction": "How do you calculate carbonyl index for plastic weathering?", "output": "The Carbonyl Index (CI) is calculated as: CI = A(1715) / A(reference). For PE, use A(1715)/A(1380). Higher CI values indicate more oxidative degradation. Values >0.5 typically indicate significant weathering."},
        {"instruction": "What is the difference between primary and secondary microplastics?", "output": "Primary microplastics are manufactured at small sizes (microbeads in cosmetics, nurdles/pellets for plastic production, synthetic textile fibers). Secondary microplastics form from the breakdown of larger plastic items through UV degradation, mechanical fragmentation, and weathering."},
        {"instruction": "How many microplastic fibers are released per laundry cycle?", "output": "Studies show 700,000 to over 1 million fibers can be released per wash of synthetic textiles. Polyester fleece releases the most. Factors include fabric type, age, washing temperature, and machine type. Mitigation includes filter bags (Guppyfriend) and external filters."},
    ]

    return qa_pairs


if __name__ == "__main__":
    # Print total knowledge size
    text = get_knowledge_text()
    print(f"Total knowledge base: {len(text)} characters")
    print(f"Training Q&A pairs: {len(get_training_prompts())}")

    # Save for training
    import json
    from pathlib import Path

    output_dir = Path(__file__).parent.parent / "training_data"
    output_dir.mkdir(exist_ok=True)

    # Save knowledge as text
    (output_dir / "microplastics_knowledge.txt").write_text(text)

    # Save Q&A pairs for fine-tuning
    with open(output_dir / "microplastics_qa.jsonl", "w") as f:
        for qa in get_training_prompts():
            f.write(json.dumps(qa) + "\n")

    print(f"Saved to {output_dir}")
