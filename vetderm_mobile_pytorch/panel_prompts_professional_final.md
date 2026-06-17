# VetDerm-Mobile Architecture and Deployment Panel Prompts - Professional Final Version

Use these prompts to generate clean publication panels for the VetDerm-Mobile veterinary dermatology paper.

Purpose: create simple, human-designed scientific diagrams that explain the VetDerm-Mobile PyTorch model, the mobile deployment pathway, and the LiteRT/TFLite exported versions. These prompts intentionally avoid equations, citations, dense tensor notation, and overclaiming. They preserve the actual code-side architecture.

Important framing:
- VetDerm-Mobile is not an SGCA variant.
- VetDerm-Mobile is a compact deployment model.
- It uses EfficientNetV2-B0, channel/spatial attention, global/local fusion, a species head, and a species-assisted disease head.
- Species-conditioned decoding means constraining disease predictions using the known or user-selected host species. It is a decoding rule, not a separate trained model.
- Do not show SGCA cross-attention, MC dropout, conformal prediction, or an autonomous diagnostic agent in these panels unless a separate comparison panel explicitly asks for them.

## Global Style For All Panels

Use this style instruction at the start of every prompt:

```text
Create a clean 2D scientific methods figure on a white background. The style should look like a professionally designed journal figure, not an AI-generated infographic. Use flat rounded rectangles, thin black or dark gray connector arrows, generous spacing, and a restrained color palette. Use Arial or Helvetica-style sans-serif text. Keep labels short and readable. No 3D objects, no shadows, no gradients, no decorative icons, no equations, no citations, no long paragraphs, no tiny text, and no photorealistic skin images.

Color code:
- Input and preprocessing: blue
- EfficientNetV2-B0 backbone: dark blue
- Attention and feature fusion: teal
- Species branch: green
- Disease branch: red
- Decoding rules: gold
- Mobile and TFLite deployment: purple
- Safety, OOD, and app checks: gray

All arrows must have clear direction. Boxes with the same role must have the same size and style. Leave enough whitespace around every label. Use simple panel labels in the top-left corner: A, B, C, etc.
```

## Panel A - VetDerm-Mobile Overview

```text
Create Panel A: "VetDerm-Mobile overview".

Layout: left-to-right flow with one main model path and two output heads.

Show these elements only:

1. Left input box:
   "Dermatology image"
   "384 x 384 RGB"

2. Blue preprocessing box:
   "Image preprocessing"
   "resize + normalize"

3. Dark blue backbone box:
   "EfficientNetV2-B0"
   "final CNN feature map"

4. Teal attention box:
   "Channel + spatial attention"
   "reweight visual features"

5. Teal fusion box:
   "Global/local fusion"
   "pooled global features"
   "local convolutional features"

6. Split into two branches:

   Green top branch:
   "Species head"
   "Cat / Cattle / Dog"

   Red bottom branch:
   "Species-assisted disease head"
   "21 disease logits"

7. Gold decoding box after the disease head:
   "Species-conditioned decoding"
   "restrict to diseases valid for selected species"

8. Final output box:
   "Ranked disease prediction"
   "top-1 and top-3 outputs"

Add a small footer note:
"Compact deployment model; not an SGCA cross-attention model."

Do not include equations, citations, parameter counts, SGCA blocks, MC dropout, conformal prediction, or Grad-CAM in this overview panel.
```

## Panel B - VetDerm-Mobile Model Architecture Detail

```text
Create Panel B: "VetDerm-Mobile architecture".

Layout: vertical model diagram with a clear split into global, local, species, and disease components.

Show these blocks from top to bottom:

1. Blue:
   "Input image"
   "384 x 384 RGB"

2. Dark blue:
   "EfficientNetV2-B0 backbone"
   "tf_efficientnetv2_b0.in1k"
   "classifier removed"

3. Teal:
   "Channel-spatial attention"
   "channel gate"
   "spatial gate"

4. Split into two teal branches:

   Left branch:
   "Local branch"
   "3x3 conv + BN + SiLU"
   "3x3 conv + BN + SiLU"
   "global average pooling"

   Right branch:
   "Global branch"
   "global average pooling"
   "linear projection"
   "ReLU"

5. Merge both branches:
   "Feature fusion"
   "concatenate global + local"
   "BatchNorm + dropout"

6. Green species branch:
   "Species hidden layer"
   "Linear + ReLU + BN + dropout"
   "species logits"

7. Red disease branch:
   "Disease head"
   "fused features + species hidden features"
   "Linear + ReLU + BN + dropout"
   "21 disease logits"

8. Output boxes:
   Green: "Species prediction"
   Red: "Disease prediction"

Add a small footer note:
"The disease head is species-assisted because it receives the learned species hidden representation."

Do not show the disease head as cross-attention. Do not draw disease tokens, FiLM, SGCA, conformal sets, or MC dropout.
```

## Panel C - Attention and Fusion Block Detail

```text
Create Panel C: "Attention and feature fusion".

Layout: focused block diagram showing only the attention and fusion portion.

Use one input on the left:
"EfficientNetV2-B0 feature map"

In the center, show two teal attention steps:

1. "Channel attention"
   "average pooled features"
   "small MLP"
   "sigmoid channel gate"

2. "Spatial attention"
   "channel average + channel max"
   "7x7 convolution"
   "sigmoid spatial gate"

After attention, split into two feature branches:

1. "Local branch"
   "two 3x3 convolution blocks"
   "pooled local descriptor"

2. "Global branch"
   "global average pooling"
   "projected global descriptor"

Merge into:
"Fused feature vector"
"global + local information"

Add a small gray note:
"Attention reweights the CNN feature map before global/local fusion."

Do not include equations, tensor symbols, or classifier heads in this panel. This panel should explain the feature-extraction mechanism only.
```

## Panel D - Species-Assisted Decoding Modes

```text
Create Panel D: "Species-assisted decoding modes".

Layout: one disease-logit input feeding three parallel decoding paths.

Start with a red input box:
"Disease logits"
"21 disease classes"

Show three parallel paths:

1. Gray path:
   "Disease-only decoding"
   "argmax across all 21 classes"
   "no species constraint"

2. Green/gold path:
   "Predicted-species decoding"
   "predict species from image"
   "restrict diseases to predicted species"

3. Gold path:
   "Known-species decoding"
   "use selected/recorded species"
   "restrict diseases to valid species labels"

End all three paths at:
"Top-1 and top-3 disease outputs"

Add a small footer note:
"Known-species decoding is the main reported species-conditioned setting."

Do not imply that the model was trained as three separate species-specific classifiers. Do not call this SGCA.
```

## Panel E - PyTorch To Mobile Export Pipeline

```text
Create Panel E: "Mobile export pipeline".

Layout: left-to-right deployment pipeline.

Show these blocks:

1. Purple:
   "Final PyTorch checkpoint"
   "VetDerm-Mobile"
   "384 px input"

2. Purple:
   "Export / conversion"
   "LiteRT-Torch"
   "optional ONNX export for validation"

3. Split into three purple output artifacts:

   "FP32 TFLite"
   "reference export"

   "FP16 TFLite"
   "high-fidelity export"

   "INT8 weight-only TFLite"
   "smallest selected deployment artifact"

4. Evaluation box:
   "Deployment checks"
   "prediction agreement"
   "accuracy and macro-F1"
   "latency and model size"

5. Final box:
   "Mobile-ready inference"
   "species + disease logits"

Add a small footer note:
"INT8 weight-only was selected for deployment because it preserved performance while reducing size and CPU latency."

Do not draw TensorFlow training. Do not show direct PyTorch-to-mobile app use without the TFLite export step.
```

## Panel F - Mobile Application Inference Workflow

```text
Create Panel F: "Mobile inference workflow".

Layout: left-to-right app pipeline with a safety branch.

Show these blocks:

1. Blue input box:
   "User image"
   "required species selection"

2. Gray check box:
   "Input checks"
   "image format"
   "field suitability"

3. Purple model box:
   "TFLite VetDerm-Mobile"
   "on-device inference"

4. Output split:

   Green:
   "Species logits"
   "optional species consistency check"

   Red:
   "Disease logits"
   "21 disease classes"

5. Gold decoding box:
   "Known-species constrained decoding"
   "valid disease labels only"

6. Purple output box:
   "Mobile result"
   "top-1 disease"
   "top-3 candidates"
   "confidence"

7. Gray safety note box:
   "Educational decision support"
   "not autonomous diagnosis"

Add a rejected-input branch from the input checks:
"unsupported image"
"no disease result returned"

Do not draw an LLM, chatbot, treatment recommendation, medication output, or autonomous diagnostic decision.
```

## Panel G - Optional SGCA vs VetDerm-Mobile Comparison

Use this only if the paper explicitly compares VetDerm-Mobile with the heavier SGCA research model.

```text
Create Panel G: "Research model versus mobile model".

Layout: two columns with SGCA on the left and VetDerm-Mobile on the right.

Left column:
"Unified SGCA"
"EfficientNetV2-S"
"species-gated cross-attention"
"uncertainty-aware decision support"
"higher-capacity research model"

Right column:
"VetDerm-Mobile"
"EfficientNetV2-B0"
"channel-spatial attention"
"global/local fusion"
"TFLite deployment"
"compact mobile model"

Bottom comparison row:
"Same 21-class taxonomy"
"same three host species"
"different deployment purpose"

Add a footer note:
"VetDerm-Mobile is positioned as a deployable companion model, not a replacement for the SGCA research architecture."

Keep this panel simple. Do not include performance numbers unless the journal figure specifically requires them.
```

## Panel H - Optional Combined Main-Figure Layout

Use this if you want one multi-panel architecture figure rather than separate standalone panels.

```text
Create a six-panel methods figure for a journal article.

Use panels A-F in a clean 2 x 3 grid:

Top row:
A. VetDerm-Mobile overview
B. VetDerm-Mobile architecture
C. Attention and feature fusion

Bottom row:
D. Species-assisted decoding modes
E. Mobile export pipeline
F. Mobile inference workflow

Use the same color palette, typography, box style, and arrow style across all panels. Each panel should have a short bold title and a panel letter in the top-left corner. Avoid equations, citations, tiny layer details, decorative icons, or visual clutter. Make the figure readable at journal column width.
```

## Quality Checklist Before Accepting A Generated Panel

- The generated figure clearly says VetDerm-Mobile, not SGCA.
- EfficientNetV2-B0 is shown, not EfficientNetV2-S.
- Input size is shown as 384 x 384 RGB.
- Channel attention and spatial attention are both shown.
- Global and local feature branches are both shown.
- Species head predicts Cat, Cattle, and Dog.
- Disease head outputs 21 disease logits.
- Disease head is shown as species-assisted by concatenating species hidden features with fused visual features.
- Species-conditioned decoding is shown as a decoding constraint using known or user-selected species, not as a separate trained model.
- TFLite export includes FP32, FP16, and INT8 weight-only variants.
- INT8 weight-only is shown as the selected compact deployment artifact.
- The mobile workflow shows educational decision support only.
- The figure does not show SGCA cross-attention, FiLM, disease tokens, MC dropout, conformal prediction, medication advice, or autonomous diagnosis.
- All labels are readable at manuscript size.
- Colors, arrows, box sizes, and spacing are consistent across panels.
