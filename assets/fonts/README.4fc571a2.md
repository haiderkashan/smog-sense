# Fonts

Generated at build time by `smogsense site build` (git-ignored). Source fonts come from the Ubuntu package
`fonts-noto-core` already installed in the image (SIL Open Font License 1.1):

* `/usr/share/fonts/truetype/noto/NotoNastaliqUrdu-Regular.ttf` (571 KB) → subset WOFF2 ≈ 107 KB
* `/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf` and `-Bold.ttf` → Latin subset

Subsetting uses `fontTools.subset` with `layout_features=*` (Nastaliq depends on GSUB/GPOS contextual rules; dropping
layout features breaks joining) and the Unicode ranges in `configs/bulletin.yaml → site.font_subset_unicodes`.
Equivalent CLI (for manual checks):

```bash
pyftsubset NotoNastaliqUrdu-Regular.ttf --unicodes="U+0020-007E,U+00A0,U+060C,U+061B,U+061F,U+0600-06FF,U+200C-200F,U+2013,U+2014,U+2026,U+00B5,U+00B3" \
  --layout-features='*' --flavor=woff2 --no-hinting --desubroutinize --output-file=NotoNastaliqUrdu-subset.woff2
```
