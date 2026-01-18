#!/usr/bin/env python3
"""
Nu1lm Web UI

A simple web interface for the Nu1lm Microplastics Expert.
"""

import gradio as gr
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from scripts.rag import SimpleVectorStore
from analyzers.raman import RamanAnalyzer
from analyzers.uv_vis import UVVisAnalyzer
from knowledge.microplastics_kb import MICROPLASTICS_KNOWLEDGE

# Initialize components
vector_store = None
raman_analyzer = RamanAnalyzer()
uv_vis_analyzer = UVVisAnalyzer()

def load_knowledge_base():
    global vector_store
    index_path = Path(__file__).parent / "data" / "microplastics_index.npz"
    if index_path.exists():
        vector_store = SimpleVectorStore()
        vector_store.load(index_path)
        return True
    return False

def answer_question(question):
    """Answer a question using the knowledge base."""
    if not question.strip():
        return "Please enter a question."

    if vector_store is None:
        # Fallback to direct knowledge lookup
        question_lower = question.lower()
        for topic, content in MICROPLASTICS_KNOWLEDGE.items():
            if any(word in question_lower for word in topic.split("_")):
                return content.strip()
        return "Knowledge base not loaded. Please run: python scripts/rag.py index --docs training_data --output data/microplastics_index.npz"

    results = vector_store.search(question, top_k=2)

    if not results or results[0][1] < 0.3:
        return "No relevant information found for that question."

    response = ""
    for doc, score, meta in results:
        if score > 0.3:
            response += doc.strip() + "\n\n"

    return response.strip()

def analyze_raman_file(file):
    """Analyze uploaded Raman spectrum."""
    if file is None:
        return "Please upload a Raman spectrum file (.txt, .csv)"

    try:
        result = raman_analyzer.analyze(Path(file.name))
        return raman_analyzer.format_for_llm(result)
    except Exception as e:
        return f"Error analyzing file: {str(e)}"

def analyze_uv_vis_file(file):
    """Analyze uploaded UV-Vis spectrum."""
    if file is None:
        return "Please upload a UV-Vis spectrum file (.txt, .csv)"

    try:
        result = uv_vis_analyzer.analyze(Path(file.name))
        return uv_vis_analyzer.format_for_llm(result)
    except Exception as e:
        return f"Error analyzing file: {str(e)}"

def get_plastic_info(plastic_type):
    """Get information about a specific plastic type."""
    plastics = {
        "PE (Polyethylene)": """**Polyethylene (PE)**
- Most common microplastic
- Density: 0.91-0.96 g/cm³ (floats in seawater)
- Raman peaks: 1062, 1128, 1295, 1440 cm⁻¹
- FTIR: 2915, 2848, 1472, 730 cm⁻¹
- Sources: Bags, bottles, packaging""",

        "PP (Polypropylene)": """**Polypropylene (PP)**
- Second most common
- Density: 0.90-0.91 g/cm³ (floats)
- Raman peaks: 808, 841, 1458 cm⁻¹
- FTIR: 2950, 2916, 1455, 1376 cm⁻¹
- Sources: Containers, caps, packaging""",

        "PS (Polystyrene)": """**Polystyrene (PS)**
- Density: 1.04-1.06 g/cm³ (sinks slowly)
- Raman peaks: 1001 (diagnostic), 1602, 3054 cm⁻¹
- FTIR: 3026, 1493, 757, 699 cm⁻¹
- Sources: Food containers, foam (Styrofoam)""",

        "PET (Polyethylene terephthalate)": """**PET (Polyethylene terephthalate)**
- Density: 1.38-1.40 g/cm³ (sinks)
- Raman peaks: 1614, 1726, 858 cm⁻¹
- FTIR: 1714, 1245, 1097 cm⁻¹
- Sources: Bottles, textile fibers""",

        "PVC (Polyvinyl chloride)": """**PVC (Polyvinyl chloride)**
- Density: 1.3-1.45 g/cm³ (sinks)
- Raman peaks: 636, 694 cm⁻¹ (C-Cl diagnostic)
- FTIR: 2912, 1427, 1254, 690 cm⁻¹
- Sources: Pipes, packaging, flooring""",

        "PA/Nylon (Polyamide)": """**Polyamide/Nylon (PA)**
- Density: 1.13-1.15 g/cm³ (sinks)
- Raman peaks: 1635, 3300 cm⁻¹ (amide)
- FTIR: 3300, 1640, 1545 cm⁻¹
- Sources: Fishing nets, textiles""",
    }
    return plastics.get(plastic_type, "Select a plastic type")

# Load knowledge base
load_knowledge_base()

# Create Gradio interface
with gr.Blocks(title="Nu1lm Microplastics Expert", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Nu1lm Microplastics Expert
    ### AI-powered microplastics analysis and knowledge system
    """)

    with gr.Tabs():
        # Q&A Tab
        with gr.Tab("Ask Questions"):
            gr.Markdown("Ask any question about microplastics - detection methods, environmental occurrence, health effects, etc.")
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What are the Raman peaks for polystyrene?",
                        lines=2
                    )
                    ask_btn = gr.Button("Ask Nu1lm", variant="primary")
                with gr.Column():
                    answer_output = gr.Textbox(label="Answer", lines=10)

            ask_btn.click(answer_question, inputs=question_input, outputs=answer_output)
            question_input.submit(answer_question, inputs=question_input, outputs=answer_output)

            gr.Examples(
                examples=[
                    "What are the Raman peaks for polyethylene?",
                    "How do you prepare samples for microplastic analysis?",
                    "What is Nile Red staining?",
                    "What are the sources of microplastics?",
                    "How do microplastics affect human health?",
                ],
                inputs=question_input
            )

        # Raman Analysis Tab
        with gr.Tab("Raman Analysis"):
            gr.Markdown("Upload a Raman spectrum file to identify the plastic type.")
            with gr.Row():
                raman_file = gr.File(label="Upload Raman Spectrum (.txt, .csv)")
                raman_btn = gr.Button("Analyze", variant="primary")
            raman_output = gr.Textbox(label="Analysis Results", lines=15)
            raman_btn.click(analyze_raman_file, inputs=raman_file, outputs=raman_output)

        # UV-Vis Analysis Tab
        with gr.Tab("UV-Vis Analysis"):
            gr.Markdown("Upload a UV-Vis spectrum file to analyze degradation and concentration.")
            with gr.Row():
                uv_file = gr.File(label="Upload UV-Vis Spectrum (.txt, .csv)")
                uv_btn = gr.Button("Analyze", variant="primary")
            uv_output = gr.Textbox(label="Analysis Results", lines=15)
            uv_btn.click(analyze_uv_vis_file, inputs=uv_file, outputs=uv_output)

        # Plastic Reference Tab
        with gr.Tab("Plastic Reference"):
            gr.Markdown("Quick reference for common microplastic types and their spectral signatures.")
            plastic_dropdown = gr.Dropdown(
                choices=[
                    "PE (Polyethylene)",
                    "PP (Polypropylene)",
                    "PS (Polystyrene)",
                    "PET (Polyethylene terephthalate)",
                    "PVC (Polyvinyl chloride)",
                    "PA/Nylon (Polyamide)",
                ],
                label="Select Plastic Type"
            )
            plastic_info = gr.Markdown()
            plastic_dropdown.change(get_plastic_info, inputs=plastic_dropdown, outputs=plastic_info)

    gr.Markdown("""
    ---
    **Nu1lm** - Microplastics Expert AI | [GitHub](https://github.com/WOOSYSTEMS/nu1lm)
    """)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Starting Nu1lm Web UI...")
    print("="*50 + "\n")
    demo.launch(share=False)
