# TensorFlow Lite / LiteRT Artifacts

This folder contains deployment artifacts for DSAN / VetDerm-Mobile.

Included:

- `vetderm_mobile_final_single_384_int8_weightonly.tflite`
- `vetderm_mobile_final_single_384_fp16.tflite`
- `tflite_deployment_comparison.csv`
- `tflite_export_summary.json`

The INT8 weight-only artifact is the compact deployment candidate discussed in the manuscript. Reported latency was measured in a WSL CPU/XNNPACK environment and should not be interpreted as a validated Android-device benchmark.
