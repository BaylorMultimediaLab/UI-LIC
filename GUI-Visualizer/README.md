# LIC Model Visualizer

This tool provides a graphical interface for running and comparing Learned Image Compression (LIC) models.

## Features

- **Model Selection**: Select one or multiple models to compare.
- **Dynamic Configuration**: Configure model-specific parameters and checkpoints.
- **Side-by-Side Comparison**: Interactive slider to compare Ground Truth images with compressed results.
- **Batch Processing**: Process an entire directory of images and view results.

## Setup

Ensure you have the required dependencies installed:

```bash
pip install streamlit pillow
```

## Running the Visualizer

From the project root directory, run:

```bash
streamlit run GUI-Visualizer/app.py
```

## Supported Models

The tool dynamically loads interfaces from `Interfaces/Testing-Interfaces`. Currently supported models include:

- **DCVC-RT**
- **ELIC**
- **LIC-HPCM**
- **StableCodec**
- **LIC-TCM** (Note: Metric evaluation only, image saving requires modification)

## Usage Tips

1. **Select Models**: Use the sidebar to pick the models you want to evaluate.
2. **Set Weights**: For each model, provide the path to its pre-trained weight file (checkpoint).
3. **Select Data**: Provide the path to a directory containing ground truth images (.png, .jpg).
4. **Run**: Click "Run Evaluation" to start the compression process.
5. **Compare**: Once finished, select an image from the dropdown to compare the GT and compressed versions side-by-side.
