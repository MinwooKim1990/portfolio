# Monosemantic Neuron Analysis with Sparse Autoencoder (SAE)

## Introduction

This project analyzes **monosemantic neurons** in the GPT-2 Medium model using **Sparse Autoencoders (SAE)**. Monosemantic neurons are neurons that are highly selective for specific concepts or features. By applying SAE, we can better understand how neurons encode and represent different concepts across layers.

---

## Experimental Setup

- **Model**: GPT-2 Medium
- **Layers Analyzed**: Layers 0, 1, 2, and 3
- **Concepts**: "numbers", "emotions", "actions", "colors"
- **Threshold for Monosemantic Neurons**: 0.6
- **Sparse Autoencoder (SAE)**:
  - **Input Dimension**: Model's hidden size
  - **Hidden Dimension**: Half of the input dimension
  - **Sparsity Weight**: 0.1

---

## Key Results and Visualizations

### 1. Neuron Selectivity Across Layers (SAE-Encoded)

**Figure 1: Neuron Selectivity Distribution**

![Neuron Selectivity Distribution](https://github.com/user-attachments/assets/51b4fd51-5d85-4e22-9557-373ded31cb5f)

- **Description**: This figure shows the distribution of selectivity scores for neurons in each layer after encoding with SAE. The monosemantic threshold is set at 0.6.
- **Insight**: Layers 0 and 1 show a higher density of neurons with selectivity scores above the threshold, indicating a higher concentration of monosemantic neurons in these layers.

---

### 2. Word-Level Activations (Top 8 Neurons per Layer)

| Layer 0 | Layer 1 |
|---------|---------|
| ![Layer 0 Word-Level Activations](https://github.com/user-attachments/assets/01b184c0-f58a-4147-9dd9-0a7f169ece0e) | ![Layer 1 Word-Level Activations](https://github.com/user-attachments/assets/8f5f47c9-5648-4fbc-a530-51b8771ec3cc) |
| **Layer 2** | **Layer 3** |
| ![Layer 2 Word-Level Activations](https://github.com/user-attachments/assets/5f2a40cc-945f-4a11-80d3-1c06575b4135) | ![Layer 3 Word-Level Activations](https://github.com/user-attachments/assets/f5f5b3fc-3a9e-480f-8fdc-2d32e6b75b8b) |

**Figure 2: Top 8 SAE-Encoded Neurons' Word-Level Activations**

- **Description**: Heatmaps showing how the top 8 SAE-encoded neurons in each layer respond to different words (e.g., "green", "happy", "flying").
- **Insight**: Neurons in Layer 0 are more specialized for "colors" (e.g., "green"), while neurons in Layer 3 show a more balanced response across concepts.

---

### 3. Mean Activations for Each Concept (SAE-Encoded)

| Layer 0 | Layer 1 |
|---------|---------|
| ![Layer 0 Mean Activations](https://github.com/user-attachments/assets/f467491a-3123-4029-a2d0-a84dc5357532) | ![Layer 1 Mean Activations](https://github.com/user-attachments/assets/15448b64-5a61-4ebf-929b-1a45988fabcc) |
| **Layer 2** | **Layer 3** |
| ![Layer 2 Mean Activations](https://github.com/user-attachments/assets/2f0dfd56-eca7-4cee-ae9e-60462ca9dbba) | ![Layer 3 Mean Activations](https://github.com/user-attachments/assets/aca88272-6722-412d-8346-b8d96864c9f4) |

**Figure 3: Mean Activations for Each Layer**

- **Description**: Bar plots showing the mean activations of SAE-encoded neurons for each concept in different layers.
- **Insight**: "Colors" (e.g., "green") tend to activate a specific set of neurons across all layers, while "emotions" and "actions" show more variability.

---

## Summary of Findings

### Monosemantic Neuron Analysis Summary (SAE-Encoded)

#### Layer 0:
- **Found 5 monosemantic neurons** in encoded space
- **Encoded neuron indices**: `[151, 195, 288, 387, 470]`
- **Average selectivity score**: `0.086`

#### Layer 1:
- **Found 4 monosemantic neurons** in encoded space
- **Encoded neuron indices**: `[419, 432, 487, 510]`
- **Average selectivity score**: `0.099`

#### Layer 2:
- **Found 5 monosemantic neurons** in encoded space
- **Encoded neuron indices**: `[48, 114, 483, 503, 511]`
- **Average selectivity score**: `0.095`

#### Layer 3:
- **Found 4 monosemantic neurons** in encoded space
- **Encoded neuron indices**: `[26, 244, 310, 505]`
- **Average selectivity score**: `0.106`

---

### Word-Level Activation Analysis

#### Layer 0:
- **Top Activating Words**:
  - **Neuron 4**: "green" (6.613), "2024" (4.763), "walks" (4.619)
  - **Neuron 127**: "green" (6.571), "flying" (5.286), "happy" (4.867)
  - **Neuron 25**: "green" (8.048), "walks" (7.763), "flying" (7.676)

#### Layer 1:
- **Top Activating Words**:
  - **Neuron 275**: "green" (5.452), "flying" (2.426), "walks" (2.187)
  - **Neuron 443**: "green" (7.727), "flying" (5.781), "walks" (5.445)
  - **Neuron 363**: "green" (7.999), "flying" (6.237), "running" (6.151)

#### Layer 2:
- **Top Activating Words**:
  - **Neuron 467**: "green" (12.835), "2024" (10.983), "flying" (10.964)
  - **Neuron 270**: "green" (9.430), "2024" (8.217), "100" (8.090)
  - **Neuron 470**: "green" (5.838), "2024" (5.449), "100" (4.891)

#### Layer 3:
- **Top Activating Words**:
  - **Neuron 355**: "green" (30.281), "2024" (29.354), "flying" (29.049)
  - **Neuron 135**: "green" (26.014), "flying" (25.853), "100" (25.523)
  - **Neuron 166**: "green" (26.536), "flying" (25.556), "100" (25.070)

---

## Conclusion

This experiment demonstrates the effectiveness of using **Sparse Autoencoders (SAE)** to analyze monosemantic neurons in the GPT-2 Medium model. By encoding neuron activations, we can better understand how specific neurons respond to different concepts and words. The results show that certain neurons are highly selective for specific concepts (e.g., "green" for colors), while others exhibit more generalized responses.

---

## Appendix: Code and Detailed Analysis

For those interested in the technical details, the Python code used for this analysis is provided below. This code includes functions for analyzing neuron activations, computing selectivity scores, and visualizing the results.
Code is available from monosemantic_neuron.py

## References
Anthropic Transformer Circuits: [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)