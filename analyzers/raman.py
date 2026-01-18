#!/usr/bin/env python3
"""
Nu1lm Raman Spectroscopy Analyzer

Analyzes Raman spectra to identify microplastic types.
Common microplastic Raman signatures:
- PE (Polyethylene): 1062, 1128, 1295, 1440 cm⁻¹
- PP (Polypropylene): 808, 841, 972, 1152, 1168, 1458 cm⁻¹
- PS (Polystyrene): 620, 1001, 1031, 1602, 3054 cm⁻¹
- PET (Polyethylene terephthalate): 858, 1096, 1614, 1726 cm⁻¹
- PVC (Polyvinyl chloride): 636, 694, 1427 cm⁻¹
- PMMA (Polymethyl methacrylate): 812, 1450, 1730 cm⁻¹
- PA/Nylon: 1130, 1635, 2900, 3300 cm⁻¹
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class RamanPeak:
    """A Raman spectral peak."""
    position: float  # cm⁻¹
    intensity: float
    width: float = 0.0


@dataclass
class PlasticMatch:
    """A potential plastic type match."""
    plastic_type: str
    confidence: float
    matched_peaks: List[Tuple[float, float]]  # (reference, observed)
    notes: str = ""


# Reference Raman signatures for common microplastics
PLASTIC_SIGNATURES = {
    "PE": {
        "name": "Polyethylene",
        "peaks": [1062, 1128, 1295, 1440, 2848, 2881],
        "strong_peaks": [1295, 1440],
        "description": "Most common microplastic. Strong CH₂ vibrations."
    },
    "PP": {
        "name": "Polypropylene",
        "peaks": [808, 841, 972, 1152, 1168, 1458, 2840, 2870, 2950],
        "strong_peaks": [841, 1458],
        "description": "Second most common. CH₃ and CH₂ deformations."
    },
    "PS": {
        "name": "Polystyrene",
        "peaks": [620, 795, 1001, 1031, 1155, 1583, 1602, 3054],
        "strong_peaks": [1001, 1602],
        "description": "Aromatic ring breathing at 1001 cm⁻¹ is diagnostic."
    },
    "PET": {
        "name": "Polyethylene terephthalate",
        "peaks": [632, 858, 1096, 1118, 1286, 1614, 1726],
        "strong_peaks": [1614, 1726],
        "description": "C=O stretch at 1726 and ring mode at 1614."
    },
    "PVC": {
        "name": "Polyvinyl chloride",
        "peaks": [636, 694, 1099, 1178, 1427, 2912, 2970],
        "strong_peaks": [636, 694],
        "description": "C-Cl stretch around 636-694 cm⁻¹."
    },
    "PMMA": {
        "name": "Polymethyl methacrylate",
        "peaks": [601, 812, 966, 1450, 1730, 2951],
        "strong_peaks": [812, 1730],
        "description": "Acrylic. C=O at 1730, C-O-C at 812."
    },
    "PA6": {
        "name": "Polyamide 6 (Nylon 6)",
        "peaks": [934, 1130, 1280, 1440, 1635, 2900, 3300],
        "strong_peaks": [1635, 3300],
        "description": "Amide I at 1635, N-H stretch at 3300."
    },
    "PC": {
        "name": "Polycarbonate",
        "peaks": [637, 706, 826, 888, 1111, 1235, 1602, 1775],
        "strong_peaks": [888, 1235],
        "description": "Carbonate C=O and aromatic modes."
    },
}


class RamanAnalyzer:
    """Analyze Raman spectra for microplastic identification."""

    def __init__(self, tolerance: float = 10.0):
        """
        Initialize analyzer.

        Args:
            tolerance: Peak matching tolerance in cm⁻¹ (default 10)
        """
        self.tolerance = tolerance
        self.signatures = PLASTIC_SIGNATURES

    def load_spectrum(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Raman spectrum from file.

        Supports:
        - .txt, .csv (two columns: wavenumber, intensity)
        - .spc (Thermo/Galactic format)
        - .wdf (Renishaw format) - requires renishawWiRE
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        if suffix in ['.txt', '.csv', '.dat']:
            # Simple two-column format
            data = np.loadtxt(file_path, delimiter=None)
            if data.shape[1] >= 2:
                return data[:, 0], data[:, 1]
            raise ValueError("File must have at least 2 columns (wavenumber, intensity)")

        elif suffix == '.spc':
            try:
                import spc
                f = spc.File(str(file_path))
                return f.x, f.sub[0].y
            except ImportError:
                raise ImportError("Install spc: pip install spc")

        elif suffix == '.wdf':
            try:
                from renishawWiRE import WDFReader
                reader = WDFReader(str(file_path))
                return reader.xdata, reader.spectra[0]
            except ImportError:
                raise ImportError("Install renishawWiRE: pip install renishawWiRE")

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def find_peaks(
        self,
        wavenumbers: np.ndarray,
        intensities: np.ndarray,
        threshold: float = 0.1,
        min_distance: int = 5
    ) -> List[RamanPeak]:
        """Find peaks in spectrum."""
        try:
            from scipy.signal import find_peaks as scipy_find_peaks
            from scipy.signal import peak_widths
        except ImportError:
            raise ImportError("Install scipy: pip install scipy")

        # Normalize
        intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min())

        # Find peaks
        peak_indices, properties = scipy_find_peaks(
            intensities_norm,
            height=threshold,
            distance=min_distance,
            prominence=0.02
        )

        # Get widths
        widths = peak_widths(intensities_norm, peak_indices, rel_height=0.5)[0]

        peaks = []
        for i, idx in enumerate(peak_indices):
            peaks.append(RamanPeak(
                position=wavenumbers[idx],
                intensity=intensities_norm[idx],
                width=widths[i] * (wavenumbers[1] - wavenumbers[0]) if len(wavenumbers) > 1 else 0
            ))

        return sorted(peaks, key=lambda p: p.intensity, reverse=True)

    def match_plastic(
        self,
        peaks: List[RamanPeak],
        top_n: int = 3
    ) -> List[PlasticMatch]:
        """Match peaks against known plastic signatures."""
        matches = []

        observed_positions = [p.position for p in peaks]

        for code, sig in self.signatures.items():
            ref_peaks = sig["peaks"]
            strong_peaks = sig["strong_peaks"]

            matched = []
            strong_matched = 0

            for ref_pos in ref_peaks:
                for obs_pos in observed_positions:
                    if abs(ref_pos - obs_pos) <= self.tolerance:
                        matched.append((ref_pos, obs_pos))
                        if ref_pos in strong_peaks:
                            strong_matched += 1
                        break

            if matched:
                # Calculate confidence
                # Weight strong peaks more heavily
                base_score = len(matched) / len(ref_peaks)
                strong_score = strong_matched / len(strong_peaks) if strong_peaks else 0
                confidence = 0.4 * base_score + 0.6 * strong_score

                matches.append(PlasticMatch(
                    plastic_type=f"{code} ({sig['name']})",
                    confidence=confidence,
                    matched_peaks=matched,
                    notes=sig["description"]
                ))

        # Sort by confidence
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches[:top_n]

    def analyze(self, file_path: Path) -> Dict:
        """
        Full analysis of a Raman spectrum.

        Returns dict with peaks found and plastic matches.
        """
        wavenumbers, intensities = self.load_spectrum(file_path)
        peaks = self.find_peaks(wavenumbers, intensities)
        matches = self.match_plastic(peaks)

        return {
            "file": str(file_path),
            "spectral_range": f"{wavenumbers.min():.1f} - {wavenumbers.max():.1f} cm⁻¹",
            "peaks_found": len(peaks),
            "top_peaks": [
                {"position": p.position, "intensity": p.intensity}
                for p in peaks[:10]
            ],
            "plastic_matches": [
                {
                    "type": m.plastic_type,
                    "confidence": f"{m.confidence:.1%}",
                    "matched_peaks": m.matched_peaks,
                    "notes": m.notes
                }
                for m in matches
            ],
            "interpretation": self._generate_interpretation(matches, peaks)
        }

    def _generate_interpretation(self, matches: List[PlasticMatch], peaks: List[RamanPeak]) -> str:
        """Generate human-readable interpretation."""
        if not matches:
            return "No clear plastic signature detected. Sample may be non-plastic or heavily degraded."

        top = matches[0]
        if top.confidence >= 0.7:
            return f"High confidence match: {top.plastic_type}. {top.notes}"
        elif top.confidence >= 0.4:
            return f"Moderate confidence match: {top.plastic_type}. Consider confirming with additional techniques."
        else:
            candidates = ", ".join([m.plastic_type.split()[0] for m in matches[:3]])
            return f"Low confidence. Possible candidates: {candidates}. Sample may be a blend or degraded."

    def format_for_llm(self, analysis: Dict) -> str:
        """Format analysis results for LLM context."""
        text = f"""Raman Spectroscopy Analysis Results:

File: {analysis['file']}
Spectral Range: {analysis['spectral_range']}
Peaks Detected: {analysis['peaks_found']}

Top 5 Peaks (cm⁻¹):
"""
        for p in analysis['top_peaks'][:5]:
            text += f"  - {p['position']:.1f} cm⁻¹ (intensity: {p['intensity']:.2f})\n"

        text += "\nPlastic Identification:\n"
        for m in analysis['plastic_matches']:
            text += f"  - {m['type']}: {m['confidence']} confidence\n"
            text += f"    Matched peaks: {m['matched_peaks']}\n"

        text += f"\nInterpretation: {analysis['interpretation']}"

        return text


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nu1lm Raman Analyzer")
    parser.add_argument("file", type=Path, help="Raman spectrum file")
    parser.add_argument("--tolerance", type=float, default=10, help="Peak matching tolerance (cm⁻¹)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    analyzer = RamanAnalyzer(tolerance=args.tolerance)
    result = analyzer.analyze(args.file)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(analyzer.format_for_llm(result))
