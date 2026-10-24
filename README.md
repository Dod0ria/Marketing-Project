# Marketing-Project

Shared workspace for the Marketing Analytics course project: notebooks, Python scripts, notes, and analysis results.

[Interim presentation slides](https://canva.link/gmd20bli081apc9)

**Do not share this link outside the team: anyone with the link can edit the presentation.**

## Project structure

Create a personal local folder containing both the `Marketing-Project/` repository and a separate `data/` folder. Keep notebooks and scripts in the repository, and all datasets in `data/`, alongside the repository:

```text
Your-personal-folder/       # Local folder; choose any name and location
├── Marketing-Project/     # Git repository
│   ├── README.md
│   ├── .gitignore
│   ├── analysis.ipynb     # Example notebook
│   └── analysis.py        # Example Python script
└── data/                  # Local datasets, outside the repository
    ├── dataset_01.csv
    └── dataset_02.csv
```

File names in this example are placeholders. Keep the original names of your downloaded datasets and use those names in your code.

## Local setup

1. Create a personal local folder and clone `Marketing-Project` inside it.
2. Inside that same personal folder, create `data/` next to `Marketing-Project/`, as shown above. This folder is outside the repository and must be created separately on each computer.
3. Download the datasets you need from the course Dropbox on Blackboard and place them in `data/`.
4. Open `Marketing-Project` in VS Code and select your Python environment. For notebooks, select the same environment as the notebook kernel.

Install the required packages in that environment if needed:

```bash
python -m pip install pandas ipykernel
```

Each team member maintains their own `data/` folder outside the repository, so Git does not track its contents. To reproduce a teammate's analysis, use the same folder layout, dataset file names, and dataset versions.

## Load a dataset

Run notebooks and scripts with **`Marketing-Project` as the working directory**. This lets everyone use the same relative paths:

```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path("..") / "data"
df = pd.read_csv(DATA_DIR / "dataset_01.csv")
print(df.head())
```

`..` goes up from `Marketing-Project/` to your personal folder; `/ "data"` then selects the dataset folder beside the repository. Replace `dataset_01.csv` with the actual file name. Use this pattern in both notebooks and scripts; avoid paths tied to your own computer, such as `/Users/your-name/...`.

For example, to run a script, open a terminal in `Marketing-Project` and run:

```bash
python analysis.py
```

If a file cannot be found, run `print(Path.cwd())` to check that the working directory is `Marketing-Project`, then check that `../data/` contains the requested file.

## Share your work

- Commit notebooks, scripts, notes, and selected analysis results.
- Write observations in notebook Markdown cells or separate Markdown files.
- Keep datasets in the external `data/` folder. The dataset rules in `.gitignore` are only a safeguard against accidentally copying data into the repository.
- Before committing a notebook, clear outputs that display raw dataset rows. Keep the explanations, charts, and summary results needed to understand the analysis.
