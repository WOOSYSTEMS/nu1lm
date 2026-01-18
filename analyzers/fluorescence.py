#!/usr/bin/env python3
"""
Nu1lm Fluorescence Image Analyzer

Analyzes fluorescence microscopy images for microplastic detection.
Uses Nile Red staining - the gold standard for microplastic visualization.

Key features:
- Detects fluorescent particles in images
- Measures particle size, shape, and count
- Classifies particles by morphology (fiber, fragment, sphere, film)
- Generates statistics for microplastic quantification
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class Particle:
    """A detected microplastic particle."""
    id: int
    centroid: Tuple[float, float]
    area: float  # pixels²
    perimeter: float
    major_axis: float
    minor_axis: float
    circularity: float
    aspect_ratio: float
    mean_intensity: float
    morphology: str  # fiber, fragment, sphere, film


@dataclass
class ImageAnalysis:
    """Complete image analysis result."""
    particle_count: int
    particles: List[Particle]
    size_distribution: Dict
    morphology_distribution: Dict
    total_area: float
    image_stats: Dict


class FluorescenceAnalyzer:
    """Analyze fluorescence microscopy images for microplastic detection."""

    def __init__(
        self,
        min_size: int = 10,      # minimum particle area in pixels
        max_size: int = 100000,  # maximum particle area in pixels
        threshold_method: str = "otsu",
        scale: float = 1.0       # µm per pixel
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.threshold_method = threshold_method
        self.scale = scale

    def load_image(self, file_path: Path) -> np.ndarray:
        """Load fluorescence image."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Install Pillow: pip install Pillow")

        img = Image.open(file_path)

        # Convert to grayscale if needed (use red channel for Nile Red)
        if img.mode == 'RGB':
            # For Nile Red, red channel often has best signal
            r, g, b = img.split()
            img = r
        elif img.mode == 'RGBA':
            r, g, b, a = img.split()
            img = r

        return np.array(img)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for particle detection."""
        try:
            from scipy.ndimage import gaussian_filter, median_filter
        except ImportError:
            raise ImportError("Install scipy: pip install scipy")

        # Reduce noise
        filtered = median_filter(image, size=3)

        # Background subtraction (rolling ball approximation)
        background = gaussian_filter(filtered.astype(float), sigma=50)
        corrected = filtered.astype(float) - background
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)

        return corrected

    def threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply thresholding to create binary mask."""
        try:
            from skimage.filters import threshold_otsu, threshold_triangle
        except ImportError:
            raise ImportError("Install scikit-image: pip install scikit-image")

        if self.threshold_method == "otsu":
            thresh_val = threshold_otsu(image)
        elif self.threshold_method == "triangle":
            thresh_val = threshold_triangle(image)
        else:
            # Manual threshold at 2 standard deviations
            thresh_val = image.mean() + 2 * image.std()

        return image > thresh_val

    def detect_particles(
        self,
        binary: np.ndarray,
        intensity_image: np.ndarray
    ) -> List[Particle]:
        """Detect and measure particles in binary image."""
        try:
            from skimage.measure import label, regionprops
        except ImportError:
            raise ImportError("Install scikit-image: pip install scikit-image")

        # Label connected components
        labeled = label(binary)
        regions = regionprops(labeled, intensity_image=intensity_image)

        particles = []
        for i, region in enumerate(regions):
            # Filter by size
            if region.area < self.min_size or region.area > self.max_size:
                continue

            # Calculate morphological features
            if region.perimeter > 0:
                circularity = 4 * np.pi * region.area / (region.perimeter ** 2)
            else:
                circularity = 0

            if region.minor_axis_length > 0:
                aspect_ratio = region.major_axis_length / region.minor_axis_length
            else:
                aspect_ratio = 1

            # Classify morphology
            morphology = self._classify_morphology(
                circularity,
                aspect_ratio,
                region.area
            )

            particles.append(Particle(
                id=i + 1,
                centroid=region.centroid,
                area=region.area * (self.scale ** 2),  # convert to µm²
                perimeter=region.perimeter * self.scale,
                major_axis=region.major_axis_length * self.scale,
                minor_axis=region.minor_axis_length * self.scale,
                circularity=circularity,
                aspect_ratio=aspect_ratio,
                mean_intensity=region.mean_intensity,
                morphology=morphology
            ))

        return particles

    def _classify_morphology(
        self,
        circularity: float,
        aspect_ratio: float,
        area: float
    ) -> str:
        """Classify particle morphology."""
        if aspect_ratio > 5:
            return "fiber"
        elif circularity > 0.85:
            return "sphere"
        elif circularity < 0.3 and aspect_ratio < 3:
            return "film"
        else:
            return "fragment"

    def calculate_statistics(self, particles: List[Particle]) -> Dict:
        """Calculate particle statistics."""
        if not particles:
            return {
                "count": 0,
                "size_distribution": {},
                "morphology_distribution": {},
                "mean_size": 0,
                "total_area": 0
            }

        areas = [p.area for p in particles]
        morphologies = [p.morphology for p in particles]

        # Size bins (µm²)
        size_bins = [0, 100, 500, 1000, 5000, float('inf')]
        size_labels = ["<100", "100-500", "500-1000", "1000-5000", ">5000"]
        size_dist = {}
        for label in size_labels:
            size_dist[label] = 0
        for area in areas:
            for i, (low, high) in enumerate(zip(size_bins[:-1], size_bins[1:])):
                if low <= area < high:
                    size_dist[size_labels[i]] += 1
                    break

        # Morphology distribution
        morph_dist = {}
        for m in set(morphologies):
            morph_dist[m] = morphologies.count(m)

        return {
            "count": len(particles),
            "size_distribution": size_dist,
            "morphology_distribution": morph_dist,
            "mean_size": np.mean(areas),
            "median_size": np.median(areas),
            "std_size": np.std(areas),
            "total_area": sum(areas),
            "fibers_percent": morph_dist.get("fiber", 0) / len(particles) * 100,
            "fragments_percent": morph_dist.get("fragment", 0) / len(particles) * 100,
        }

    def analyze(self, file_path: Path) -> Dict:
        """Full image analysis."""
        # Load and process
        image = self.load_image(file_path)
        processed = self.preprocess(image)
        binary = self.threshold(processed)
        particles = self.detect_particles(binary, processed)
        stats = self.calculate_statistics(particles)

        return {
            "file": str(file_path),
            "image_size": f"{image.shape[1]} x {image.shape[0]} pixels",
            "scale": f"{self.scale} µm/pixel",
            "particle_count": stats["count"],
            "size_distribution": stats["size_distribution"],
            "morphology_distribution": stats["morphology_distribution"],
            "mean_particle_size": f"{stats['mean_size']:.1f} µm²",
            "total_particle_area": f"{stats['total_area']:.1f} µm²",
            "fiber_percentage": f"{stats.get('fibers_percent', 0):.1f}%",
            "fragment_percentage": f"{stats.get('fragments_percent', 0):.1f}%",
            "particles": [
                {
                    "id": p.id,
                    "area_um2": p.area,
                    "morphology": p.morphology,
                    "aspect_ratio": p.aspect_ratio
                }
                for p in particles[:50]  # Limit to first 50
            ],
            "interpretation": self._generate_interpretation(stats, particles)
        }

    def _generate_interpretation(self, stats: Dict, particles: List[Particle]) -> str:
        """Generate interpretation."""
        if stats["count"] == 0:
            return "No microplastic particles detected in this image."

        parts = [f"Detected {stats['count']} microplastic particles."]

        # Dominant morphology
        morph_dist = stats["morphology_distribution"]
        if morph_dist:
            dominant = max(morph_dist, key=morph_dist.get)
            parts.append(f"Dominant morphology: {dominant} ({morph_dist[dominant]}/{stats['count']}).")

        # Size assessment
        mean_size = stats["mean_size"]
        if mean_size < 100:
            parts.append("Particles are predominantly small (<100 µm²).")
        elif mean_size < 1000:
            parts.append("Particles are medium-sized (100-1000 µm²).")
        else:
            parts.append("Particles are relatively large (>1000 µm²).")

        # Fiber warning
        if stats.get("fibers_percent", 0) > 50:
            parts.append("High fiber content - consider textile sources.")

        return " ".join(parts)

    def format_for_llm(self, analysis: Dict) -> str:
        """Format for LLM context."""
        text = f"""Fluorescence Microscopy Analysis:

File: {analysis['file']}
Image Size: {analysis['image_size']}
Scale: {analysis['scale']}

Particle Detection Results:
- Total Particles: {analysis['particle_count']}
- Mean Particle Size: {analysis['mean_particle_size']}
- Total Particle Area: {analysis['total_particle_area']}

Size Distribution:
"""
        for size_range, count in analysis['size_distribution'].items():
            text += f"  - {size_range} µm²: {count} particles\n"

        text += "\nMorphology Distribution:\n"
        for morph, count in analysis['morphology_distribution'].items():
            text += f"  - {morph}: {count}\n"

        text += f"\nInterpretation: {analysis['interpretation']}"

        return text

    def generate_annotated_image(
        self,
        file_path: Path,
        output_path: Path,
        particles: List[Particle]
    ):
        """Generate annotated image with detected particles."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise ImportError("Install Pillow: pip install Pillow")

        img = Image.open(file_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        colors = {
            "fiber": (255, 0, 0),      # Red
            "fragment": (0, 255, 0),   # Green
            "sphere": (0, 0, 255),     # Blue
            "film": (255, 255, 0)      # Yellow
        }

        for p in particles:
            color = colors.get(p.morphology, (255, 255, 255))
            y, x = p.centroid
            # Draw circle around particle
            r = max(5, int(np.sqrt(p.area / np.pi)))
            draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=2)

        img.save(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nu1lm Fluorescence Analyzer")
    parser.add_argument("file", type=Path, help="Fluorescence image file")
    parser.add_argument("--scale", type=float, default=1.0, help="µm per pixel")
    parser.add_argument("--min-size", type=int, default=10, help="Min particle size (pixels)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--annotate", type=Path, help="Save annotated image to path")

    args = parser.parse_args()

    analyzer = FluorescenceAnalyzer(
        min_size=args.min_size,
        scale=args.scale
    )

    result = analyzer.analyze(args.file)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(analyzer.format_for_llm(result))

    if args.annotate:
        particles = analyzer.detect_particles(
            analyzer.threshold(analyzer.preprocess(analyzer.load_image(args.file))),
            analyzer.preprocess(analyzer.load_image(args.file))
        )
        analyzer.generate_annotated_image(args.file, args.annotate, particles)
        print(f"\nAnnotated image saved to {args.annotate}")
