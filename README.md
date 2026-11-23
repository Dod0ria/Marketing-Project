# LG TV project: code

Group project for Marketing Analytics (30420). The code compares LG and Samsung TVs at one retailer, using the course's behaviour log for 1 October to 30 November 2019.

## Folder layout

The code reads the course files through the relative path `../data/01_LG_Electronics_Appliances/`, so the data folder must sit next to the project folder and the code must run with `Marketing-Project` as the working directory.

```text
any-folder/
├── Marketing-Project/
│   ├── README.md
│   ├── 00_scope_and_methodology_rationale.ipynb
│   ├── 01_build_tv_dataset.py
│   ├── 02_preliminary_analysis.ipynb
│   ├── 03_models.ipynb
│   ├── tv_considerations_dictionary.md
│   └── results/                      
└── data/
    └── 01_LG_Electronics_Appliances/
        ├── electronics_market_behavior.csv
        └── appliance_brand_reviews.csv
```

Place the two course files in `data/01_LG_Electronics_Appliances/` with their original names. `01_build_tv_dataset.py` writes its two working files to the same folder.

## Install

```bash
python -m pip install pandas numpy scipy statsmodels matplotlib ipykernel
```

The code was run with Python 3.13, pandas 3.0, NumPy 2.5, SciPy 1.18, statsmodels 0.15 and matplotlib 3.11.

## Run order

Run the notebooks from a fresh kernel with Run All, and the script with `python 01_build_tv_dataset.py`.

| Step | File | Reads | Writes | Time |
| --- | --- | --- | --- | --- |
| 1 | `00_scope_and_methodology_rationale.ipynb` | The two course files | `results/scope/` (Tables 0.1 to 0.5, Figures 0.1 to 0.4) | About 10 seconds |
| 2 | `01_build_tv_dataset.py` | `electronics_market_behavior.csv` | `events_clean.csv` and `tv_considerations.csv` in the data folder | About 30 seconds |
| 3 | `02_preliminary_analysis.ipynb` | The two working files | `results/preliminary/` (Tables 2.1 to 2.15, Figures 2.1 to 2.6) | About 30 seconds |
| 4 | `03_models.ipynb` | The two working files, and the exhibits saved by steps 1 and 3 for the combined file | `results/models/` (Tables 3.1 to 3.9, Figure 3.1) and `results/figures_and_tables.html` | About 45 seconds |

`01_build_tv_dataset.py` ends with an independent recomputation of every derived column on a sample of rows and with logical checks on all rows. It stops with an error if any check fails and saves the two files only after every check has passed.

Each results folder holds an `exhibit_index.csv` that lists its tables and figures with the files each one is built from. The last cell of `03_models.ipynb` gathers every table and figure into `results/figures_and_tables.html`, and `figures_and_tables.pdf` is that page printed to PDF.

`tv_considerations_dictionary.md` defines the columns of the two working files and the rules used to build them.

## Data version

The results were produced with these versions of the course files (SHA-256).

| File | SHA-256 |
| --- | --- |
| `electronics_market_behavior.csv` | `3303a56e62cff3e38491d58e41f40706317c500039c26f135bc89a811bc59022` |
| `appliance_brand_reviews.csv` | `64b57e87b44ec047403bf1b29bdc60ecd9ead2add64b8327ac6011524bb31a60` |
