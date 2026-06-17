# Data Notes

Raw veterinary dermatology images are not included in this GitHub-ready release.

The expected private data layout for full reproduction is:

```text
data/training_data_deduped_splits/seed42/
  train/
  val/
  test/

data/development_external_2026_05_02/
  Cat/
  Cattles/
  Dog/
```

The development-external cohort is an internet-derived robustness cohort and is not a locked prospective clinical validation set.
