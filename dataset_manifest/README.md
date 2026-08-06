# DSAN dataset manifest

This directory contains an **index of the images used to evaluate DSAN. It contains no
image data.** It is released in place of the images so that the study can be verified
without redistributing material whose licensing status has not been established.

## Files

| File | Contents |
|---|---|
| `DSAN_dataset_manifest.csv` | One row per evaluation image (7,864 rows) |
| `source_summary.csv` | Image counts grouped by upstream collection and redistribution status |

## Coverage

| Split | Images |
|---|---:|
| Validation | 3,254 |
| Internal test | 3,259 |
| Development external | 1,351 |
| **Total** | **7,864** |

These are the three splits on which all reported results are computed. The 16,618
training images are **not** covered by this manifest; see *Known limitations*.

## Columns

| Column | Meaning |
|---|---|
| `image_id` | First 16 hex characters of the SHA-256 digest. Stable, content-derived, reveals no filename |
| `split` | `validation`, `internal_test`, or `development_external` |
| `species` | Host species directory (`Cat`, `Cattles`, `Dog`) |
| `class_label` | One of the 21 diagnostic categories |
| `source_collection` | Upstream collection the image was drawn from |
| `source_reference` | Canonical reference for that collection, where identifiable |
| `redistribution_status` | Whether redistribution rights have been established (see below) |
| `licence` | **Intentionally blank.** To be completed by the authors after verification |
| `width`, `height`, `file_size_bytes` | Image geometry and size |
| `sha256` | Full SHA-256 of the file bytes — exact-match verification |
| `phash`, `dhash` | Perceptual and difference hashes — near-duplicate verification after re-encoding or rescaling |
| `exif_tag_count`, `has_exif` | Whether EXIF metadata is still present in the file |
| `embedded_info_keys` | Non-EXIF metadata keys carried in the container |

## Verifying an image against this manifest

1. Obtain the image from its original source.
2. Compute `sha256`. If it matches a row, the file is byte-identical to the one used here.
3. If it does not match — for example because the source re-encoded it — compute `phash`
   and `dhash` and compare. A Hamming distance of 0–4 indicates the same underlying image.

## Redistribution status

| Status | Images | % | Meaning |
|---|---:|---:|---|
| `unknown_provenance_verify_required` | 6,434 | 81.8% | Original licence not established. **Must not be redistributed** until verified |
| `public_dataset_verify_licence` | 1,430 | 18.2% | Traced to a named public dataset; that dataset's own licence governs reuse and must be checked before redistribution |

`public_dataset_verify_licence` records that the **upstream collection is identified**, not
that redistribution is permitted. Each upstream licence must be read before any release.

## Identified upstream collections

| Collection | Images | Reference |
|---|---:|---|
| Stanford Dogs | 960 | Khosla et al., 2011; derived from ImageNet |
| dog skin split dataset | 210 | Roboflow export |
| Oxford-IIIT Pet | 200 | Parkhi et al., 2012 |
| Mendeley pet dog skin | 60 | Mendeley Data deposit `5dbht54kw7-1` |

The development-external cohort is further resolved to named curated archives
(`CAT_SKIN_DISEASE__Ringworm`, `Dogs__Fungal_infections`, `lumpycows`, and others);
1,148 of 1,351 carry an archive attribution and 203 remain unattributed.

## Known limitations

1. **Training images are not covered.** The archived training split retained directory
   links rather than image copies, so the 16,618 training images could not be inventoried
   at file level. Hash-level records for an earlier, differently deduplicated
   source family (11,859 training images) exist separately.
2. **No source URLs.** Original retrieval URLs were not recorded at collection time and
   cannot be reconstructed. Provenance is reported at collection level, not per-image URL.
3. **Licences are not asserted.** The `licence` column is deliberately empty. Determining
   the licence of each upstream collection, and of the 6,434 images with unestablished
   provenance, is outstanding work that must be completed before any public image release.
4. **EXIF is present in 337 images (4.3%).** These files retain camera or software metadata
   that has not been stripped. Any future release should remove it.

## Integrity check

No byte-identical duplicates were found across the three evaluation splits
(0 of 7,864), consistent with the deduplication protocol described in the manuscript.
