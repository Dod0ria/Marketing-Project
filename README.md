# Marketing-Project
This Repository is used to coordinate the work of the Marketing Analytics course project

Here is the link for the Interim Slides: https://canva.link/gmd20bli081apc9
!DO NOT share it as allows everyone with the link to edit the presentation!


## Quick start: notebooks and local datasets

### 1. Put the datasets outside the repository

Each team member downloads the datasets from the course Dropbox on Blackboard and keeps this layout on their own computer:

```text
Your private project folder/
├── Marketing-Project/                 # Git repository
│   ├── README.md
│   ├── .gitignore
│   └── your notebooks and notes
├── 01_LG_Electronics_Appliances/       # Local data, outside Git
│   ├── appliance_brand_reviews.csv
│   └── electronics_market_behavior.csv
└── 05_Hospitality_Travel/
    └── hospitality_bookings.csv
```

Your private folder can be anywhere. Keep the same dataset folder names, file names, and course file versions. Each person downloads their own copy; Git does not download or synchronize datasets. No setup scripts or configuration files are needed.

### 2. Create a notebook and read a file

Open `Marketing-Project` in VS Code. Save your notebooks directly in the repository's main folder and select a Python kernel with pandas installed. If needed, install it in that environment with `python -m pip install pandas ipykernel`.

Run this cell with `Marketing-Project` as the kernel's working directory:

```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path("..").resolve()
reviews = pd.read_csv(
    DATA_DIR / "01_LG_Electronics_Appliances" / "appliance_brand_reviews.csv"
)
reviews.head()
```

`..` means one folder above the current working directory. Everyone uses the same code because the relative folder layout matches. For another dataset, change only the dataset subfolder and file name.

### 3. Save and share your work

- Keep all notebooks and observations in the repository. Write observations in Markdown cells or separate Markdown files.
- Commit notebooks, notes, and intended analysis results. Keep downloaded datasets outside the repository.
- `.gitignore` also excludes the known raw datasets if accidentally copied inside; it does not remove files already tracked by Git.
- Before committing, clear notebook outputs displaying raw dataset rows. Keep your written observations and intended results.

### Quick troubleshooting

- **File not found:** run `print(Path.cwd())`. It should show `Marketing-Project`. Check the kernel's working directory, folder layout, and exact file names.
- **Datasets not visible in Explorer:** opening the repository shows only its contents. The notebook can still read the external files. To display them, use **File → Add Folder to Workspace…** and select the dataset folders; this does not move them into Git.
