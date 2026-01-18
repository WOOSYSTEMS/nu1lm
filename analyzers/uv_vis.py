#!/usr/bin/env python3
"""
Nu1lm UV-Vis Spectroscopy Analyzer

Analyzes UV-Vis spectra for microplastic characterization.
UV-Vis is useful for:
- Detecting additives and dyes in plastics
- Monitoring plastic degradation (carbonyl index)
- Quantifying microplastic concentration with dyes (Nile Red)
- Identifying colored microplastics

Common UV-Vis features:
- Nile Red fluorescence for MP detection: ~580 nm emission
- Aromatic plastics (PS, PET): absorption 250-280 nm
- Degradation products: broad absorption increase
- Additives/stabilizers: various UV absorbers
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class UVVisFeature:
    """A UV-Vis spectral feature."""
    wavelength: float  # nm
    absorbance: float
    feature_type: str  # "peak", "shoulder", "edge"


@dataclass
class UVVisAnalysis:
    """UV-Vis analysis result."""
    features: List[UVVisFeature]
    possible_components: List[str]
    degradation_indicators: Dict
    concentration_estimate: Optional[float] = None


# Common UV-Vis signatures
UV_VIS_SIGNATURES = {
    "aromatic_plastics": {
        "range": (250, 290),
        "description": "π→π* transitions in aromatic rings (PS, PET, PC)",
        "plastics": ["PS", "PET", "PC"]
    },
    "nile_red_bound": {
        "range": (550, 600),
        "description": "Nile Red bound to microplastics",
        "plastics": ["all hydrophobic plastics"]
    },
    "carbonyl_degradation": {
        "range": (260, 290),
        "description": "Carbonyl groups from oxidative degradation",
        "indicator": "weathering/UV degradation"
    },
    "uv_stabilizers": {
        "range": (300, 400),
        "description": "UV stabilizer additives (benzotriazoles, etc.)",
        "indicator": "plastic additives present"
    },
    "titanium_dioxide": {
        "range": (350, 400),
        "description": "TiO₂ pigment absorption edge",
        "indicator": "white pigment/opacity agent"
    }
}


class UVVisAnalyzer:
    """Analyze UV-Vis spectra for microplastic characterization."""

    def __init__(self):
        self.signatures = UV_VIS_SIGNATURES

    def load_spectrum(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load UV-Vis spectrum from file.

        Supports:
        - .txt, .csv (wavelength, absorbance)
        - .dx (JCAMP-DX format)
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix in ['.txt', '.csv', '.dat']:
            data = np.loadtxt(file_path, delimiter=None)
            return data[:, 0], data[:, 1]

        elif suffix == '.dx':
            # Simple JCAMP-DX parser
            return self._parse_jcamp(file_path)

        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _parse_jcamp(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse JCAMP-DX file."""
        wavelengths = []
        absorbances = []
        in_data = False

        with open(file_path) as f:
            for line in f:
                if "##XYDATA" in line or "##XYPOINTS" in line:
                    in_data = True
                    continue
                if line.startswith("##END"):
                    break
                if in_data and not line.startswith("##"):
                    parts = line.split()
                    for i in range(0, len(parts), 2):
                        if i + 1 < len(parts):
                            wavelengths.append(float(parts[i]))
                            absorbances.append(float(parts[i + 1]))

        return np.array(wavelengths), np.array(absorbances)

    def find_features(
        self,
        wavelengths: np.ndarray,
        absorbances: np.ndarray
    ) -> List[UVVisFeature]:
        """Find spectral features."""
        try:
            from scipy.signal import find_peaks
            from scipy.ndimage import gaussian_filter1d
        except ImportError:
            raise ImportError("Install scipy: pip install scipy")

        # Smooth and find peaks
        smoothed = gaussian_filter1d(absorbances, sigma=2)
        peak_idx, _ = find_peaks(smoothed, prominence=0.01)

        features = []
        for idx in peak_idx:
            features.append(UVVisFeature(
                wavelength=wavelengths[idx],
                absorbance=absorbances[idx],
                feature_type="peak"
            ))

        return sorted(features, key=lambda f: f.absorbance, reverse=True)

    def calculate_degradation_index(
        self,
        wavelengths: np.ndarray,
        absorbances: np.ndarray
    ) -> Dict:
        """
        Calculate degradation indicators.

        Carbonyl Index: A(1715)/A(reference)
        For UV-Vis, we use A(280)/A(250) as proxy for aromatic degradation
        """
        # Find absorbance at key wavelengths
        def get_abs_at(wl: float) -> float:
            idx = np.argmin(np.abs(wavelengths - wl))
            return absorbances[idx]

        results = {}

        # UV absorption ratio (higher = more degradation)
        if wavelengths.min() < 260 and wavelengths.max() > 280:
            a280 = get_abs_at(280)
            a250 = get_abs_at(250)
            if a250 > 0:
                results["uv_ratio_280_250"] = a280 / a250
                results["degradation_level"] = (
                    "high" if a280/a250 > 1.5 else
                    "moderate" if a280/a250 > 1.0 else
                    "low"
                )

        # Total UV absorption (yellowing indicator)
        uv_mask = (wavelengths >= 300) & (wavelengths <= 400)
        if uv_mask.any():
            results["uv_absorption_300_400"] = np.trapz(
                absorbances[uv_mask],
                wavelengths[uv_mask]
            )

        return results

    def identify_components(
        self,
        wavelengths: np.ndarray,
        absorbances: np.ndarray,
        features: List[UVVisFeature]
    ) -> List[str]:
        """Identify possible components based on spectral features."""
        components = []

        for name, sig in self.signatures.items():
            wl_range = sig["range"]
            # Check if there's significant absorption in this range
            mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
            if mask.any():
                region_abs = absorbances[mask]
                if region_abs.max() > 0.05:  # Significant absorption
                    components.append(f"{name}: {sig['description']}")

        return components

    def estimate_concentration(
        self,
        wavelengths: np.ndarray,
        absorbances: np.ndarray,
        method: str = "nile_red"
    ) -> Optional[float]:
        """
        Estimate microplastic concentration using standard curves.

        Methods:
        - nile_red: Uses Nile Red fluorescence/absorption
        - turbidity: Uses scattering at 600nm
        """
        if method == "nile_red":
            # Find Nile Red peak around 580nm
            mask = (wavelengths >= 550) & (wavelengths <= 600)
            if mask.any():
                peak_abs = absorbances[mask].max()
                # Rough calibration: 0.1 AU ≈ 10 particles/mL
                # This needs proper calibration for real use
                return peak_abs * 100  # particles/mL estimate

        elif method == "turbidity":
            # Turbidity at 600nm
            idx = np.argmin(np.abs(wavelengths - 600))
            abs_600 = absorbances[idx]
            return abs_600 * 50  # rough estimate

        return None

    def analyze(self, file_path: Path) -> Dict:
        """Full UV-Vis analysis."""
        wavelengths, absorbances = self.load_spectrum(file_path)
        features = self.find_features(wavelengths, absorbances)
        degradation = self.calculate_degradation_index(wavelengths, absorbances)
        components = self.identify_components(wavelengths, absorbances, features)
        concentration = self.estimate_concentration(wavelengths, absorbances)

        return {
            "file": str(file_path),
            "spectral_range": f"{wavelengths.min():.1f} - {wavelengths.max():.1f} nm",
            "features": [
                {"wavelength": f.wavelength, "absorbance": f.absorbance}
                for f in features[:10]
            ],
            "possible_components": components,
            "degradation_indicators": degradation,
            "concentration_estimate": concentration,
            "interpretation": self._generate_interpretation(components, degradation, concentration)
        }

    def _generate_interpretation(
        self,
        components: List[str],
        degradation: Dict,
        concentration: Optional[float]
    ) -> str:
        """Generate interpretation text."""
        parts = []

        if components:
            parts.append(f"Detected features suggest: {'; '.join(components[:3])}")

        if degradation.get("degradation_level"):
            parts.append(f"Degradation level: {degradation['degradation_level']}")

        if concentration:
            parts.append(f"Estimated concentration: ~{concentration:.0f} particles/mL")

        return " | ".join(parts) if parts else "No significant features detected."

    def format_for_llm(self, analysis: Dict) -> str:
        """Format for LLM context."""
        text = f"""UV-Vis Spectroscopy Analysis:

File: {analysis['file']}
Spectral Range: {analysis['spectral_range']}

Key Features:
"""
        for f in analysis['features'][:5]:
            text += f"  - {f['wavelength']:.1f} nm (A = {f['absorbance']:.3f})\n"

        text += "\nPossible Components:\n"
        for c in analysis['possible_components']:
            text += f"  - {c}\n"

        text += f"\nDegradation Analysis: {analysis['degradation_indicators']}\n"
        text += f"\nInterpretation: {analysis['interpretation']}"

        return text


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Nu1lm UV-Vis Analyzer")
    parser.add_argument("file", type=Path, help="UV-Vis spectrum file")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    analyzer = UVVisAnalyzer()
    result = analyzer.analyze(args.file)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(analyzer.format_for_llm(result))
