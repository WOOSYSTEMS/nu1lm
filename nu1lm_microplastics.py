#!/usr/bin/env python3
"""
Nu1lm Microplastics Expert

An AI assistant specialized in microplastics research that can:
- Answer questions about microplastics (sources, detection, effects)
- Analyze Raman spectra to identify plastic types
- Analyze UV-Vis spectra for degradation and quantification
- Analyze fluorescence microscopy images for particle detection
- Interpret spectral data and provide research recommendations
"""

import argparse
from pathlib import Path
from typing import Optional, Dict, List
import json

# Import analyzers
from analyzers.raman import RamanAnalyzer
from analyzers.uv_vis import UVVisAnalyzer
from analyzers.fluorescence import FluorescenceAnalyzer
from knowledge.microplastics_kb import MICROPLASTICS_KNOWLEDGE, get_knowledge_text

# LLM imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     ███╗   ██╗██╗   ██╗ ██╗██╗     ███╗   ███╗           ║
║     ████╗  ██║██║   ██║███║██║     ████╗ ████║           ║
║     ██╔██╗ ██║██║   ██║╚██║██║     ██╔████╔██║           ║
║     ██║╚██╗██║██║   ██║ ██║██║     ██║╚██╔╝██║           ║
║     ██║ ╚████║╚██████╔╝ ██║███████╗██║ ╚═╝ ██║           ║
║     ╚═╝  ╚═══╝ ╚═════╝  ╚═╝╚══════╝╚═╝     ╚═╝           ║
║                                                           ║
║            M I C R O P L A S T I C S   E X P E R T       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝

Commands:
  analyze raman <file>       - Analyze Raman spectrum
  analyze uv-vis <file>      - Analyze UV-Vis spectrum
  analyze image <file>       - Analyze fluorescence image
  help                       - Show help
  quit                       - Exit

Or just ask any question about microplastics!
"""

SYSTEM_PROMPT = """You are Nu1lm, a specialized AI expert on microplastics. You have deep knowledge about:

1. DETECTION METHODS:
- Raman spectroscopy (peak identification, polymer matching)
- FTIR spectroscopy (ATR, µ-FTIR, FPA imaging)
- Fluorescence microscopy with Nile Red staining
- UV-Vis spectroscopy for degradation monitoring
- Py-GC/MS for mass-based quantification

2. PLASTIC TYPES:
- PE, PP, PS, PET, PVC, PA, PMMA, PC and their spectral signatures
- Morphologies: fibers, fragments, films, spheres, foams

3. SAMPLE PREPARATION:
- Density separation, organic removal, filtration
- Quality control and blank procedures

4. ENVIRONMENTAL OCCURRENCE:
- Marine, freshwater, terrestrial, atmospheric
- Sources and transport pathways

5. HEALTH & ECOLOGY:
- Ecological effects on biota
- Human exposure and potential health effects

6. DATA ANALYSIS:
- Spectral interpretation
- Statistical methods
- Risk assessment approaches

When analyzing data, provide:
- Clear identification of plastic types if possible
- Confidence levels for your conclusions
- Recommendations for additional analysis if needed
- Relevant scientific context

Be precise, cite specific peak positions and spectral features, and acknowledge uncertainty when appropriate.
"""


class Nu1lmMicroplastics:
    """Nu1lm Microplastics Expert System."""

    def __init__(self, model_path: Optional[Path] = None):
        """Initialize Nu1lm."""
        self.raman_analyzer = RamanAnalyzer()
        self.uv_vis_analyzer = UVVisAnalyzer()
        self.fluorescence_analyzer = FluorescenceAnalyzer()

        self.model = None
        self.tokenizer = None
        self.model_loaded = False

        if model_path and model_path.exists():
            self._load_model(model_path)

    def _load_model(self, model_path: Path):
        """Load the LLM model."""
        try:
            print("Loading Nu1lm model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Running in knowledge-base only mode.")
            self.model_loaded = False

    def analyze_raman(self, file_path: Path) -> str:
        """Analyze a Raman spectrum file."""
        try:
            result = self.raman_analyzer.analyze(file_path)
            analysis_text = self.raman_analyzer.format_for_llm(result)

            # If model is loaded, get interpretation
            if self.model_loaded:
                prompt = f"""Based on this Raman analysis, provide a detailed interpretation:

{analysis_text}

Discuss:
1. Most likely plastic type and confidence
2. Evidence for this identification (specific peaks)
3. Any concerns about the identification
4. Recommended next steps if uncertain
"""
                interpretation = self._generate(prompt)
                return f"{analysis_text}\n\n--- Nu1lm Interpretation ---\n{interpretation}"
            else:
                return analysis_text

        except Exception as e:
            return f"Error analyzing Raman spectrum: {e}"

    def analyze_uv_vis(self, file_path: Path) -> str:
        """Analyze a UV-Vis spectrum file."""
        try:
            result = self.uv_vis_analyzer.analyze(file_path)
            analysis_text = self.uv_vis_analyzer.format_for_llm(result)

            if self.model_loaded:
                prompt = f"""Interpret this UV-Vis analysis for microplastics:

{analysis_text}

Discuss:
1. Evidence of plastic presence
2. Degradation state if determinable
3. Concentration estimate reliability
4. Recommended follow-up analysis
"""
                interpretation = self._generate(prompt)
                return f"{analysis_text}\n\n--- Nu1lm Interpretation ---\n{interpretation}"
            else:
                return analysis_text

        except Exception as e:
            return f"Error analyzing UV-Vis spectrum: {e}"

    def analyze_image(self, file_path: Path, scale: float = 1.0) -> str:
        """Analyze a fluorescence microscopy image."""
        try:
            self.fluorescence_analyzer.scale = scale
            result = self.fluorescence_analyzer.analyze(file_path)
            analysis_text = self.fluorescence_analyzer.format_for_llm(result)

            if self.model_loaded:
                prompt = f"""Interpret this fluorescence microscopy analysis for microplastics:

{analysis_text}

Discuss:
1. Microplastic contamination level
2. Dominant particle types and their likely sources
3. Size distribution significance
4. Recommendations for confirmation (spectroscopy)
"""
                interpretation = self._generate(prompt)
                return f"{analysis_text}\n\n--- Nu1lm Interpretation ---\n{interpretation}"
            else:
                return analysis_text

        except Exception as e:
            return f"Error analyzing image: {e}"

    def _generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response using the LLM."""
        if not self.model_loaded:
            return self._knowledge_lookup(prompt)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        if hasattr(self.tokenizer, 'apply_chat_template'):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(self.model.device)
        else:
            text = f"System: {SYSTEM_PROMPT}\n\nUser: {prompt}\n\nNu1lm:"
            inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        response = self.tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        return response.strip()

    def _knowledge_lookup(self, query: str) -> str:
        """Simple keyword-based knowledge lookup when model not loaded."""
        query_lower = query.lower()

        # Simple keyword matching
        best_match = None
        best_score = 0

        for topic, content in MICROPLASTICS_KNOWLEDGE.items():
            topic_words = topic.replace("_", " ").split()
            score = sum(1 for word in topic_words if word in query_lower)
            if score > best_score:
                best_score = score
                best_match = content

        if best_match:
            return best_match.strip()
        else:
            return "I don't have specific information about that. Please load the Nu1lm model for more detailed responses, or try a different question about microplastics."

    def answer(self, question: str) -> str:
        """Answer a question about microplastics."""
        if self.model_loaded:
            return self._generate(question)
        else:
            return self._knowledge_lookup(question)

    def run_interactive(self):
        """Run interactive chat interface."""
        print(BANNER)

        while True:
            try:
                user_input = input("\033[94mYou:\033[0m ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'help':
                print(BANNER)
                continue

            # Check for analysis commands
            if user_input.lower().startswith('analyze'):
                parts = user_input.split(maxsplit=2)
                if len(parts) < 3:
                    print("Usage: analyze <type> <file_path>")
                    print("Types: raman, uv-vis, image")
                    continue

                analysis_type = parts[1].lower()
                file_path = Path(parts[2])

                if not file_path.exists():
                    print(f"File not found: {file_path}")
                    continue

                if analysis_type == 'raman':
                    result = self.analyze_raman(file_path)
                elif analysis_type in ['uv-vis', 'uvvis', 'uv']:
                    result = self.analyze_uv_vis(file_path)
                elif analysis_type in ['image', 'img', 'fluorescence']:
                    result = self.analyze_image(file_path)
                else:
                    print(f"Unknown analysis type: {analysis_type}")
                    continue

                print(f"\033[92mNu1lm:\033[0m\n{result}\n")
            else:
                # Regular question
                response = self.answer(user_input)
                print(f"\033[92mNu1lm:\033[0m {response}\n")


def main():
    parser = argparse.ArgumentParser(description="Nu1lm Microplastics Expert")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "models" / "smollm-360m",
        help="Path to Nu1lm model"
    )
    parser.add_argument(
        "--analyze",
        type=str,
        choices=["raman", "uv-vis", "image"],
        help="Analysis type for batch mode"
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="File to analyze (batch mode)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Ask a single question (batch mode)"
    )

    args = parser.parse_args()

    nu1lm = Nu1lmMicroplastics(args.model)

    # Batch mode
    if args.analyze and args.file:
        if args.analyze == "raman":
            print(nu1lm.analyze_raman(args.file))
        elif args.analyze == "uv-vis":
            print(nu1lm.analyze_uv_vis(args.file))
        elif args.analyze == "image":
            print(nu1lm.analyze_image(args.file))
        return

    if args.question:
        print(nu1lm.answer(args.question))
        return

    # Interactive mode
    nu1lm.run_interactive()


if __name__ == "__main__":
    main()
