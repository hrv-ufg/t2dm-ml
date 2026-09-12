# preprocessing

Preprocessing of RRi data for HRV calculation.

To run the code, first install the required dependencies:

```bash
pip install -r requirements.txt
```

Then, ensure that the data are organized according to the following structure, or modify the paths in the `src/config.py` file:

```text
vfc-diabeticos/
├── data/
│   ├── control/
│   └── diabetic/
├── src/
│   ├── main.py
│   └── ...
├── README.md
└── requirements.txt
```

## Methodology

The adopted methodology can be summarized in four steps:

1. **Discard the first 10 RRi entries** from each file (patient).

2. **Assess the stability of the RRi signals** and discard files that do not meet the established threshold (90%). Signal stability is assessed based on:

   * **Outlier detection:** RRi < 300 or RRi > 2000;
   * **Ectopic beat detection:** RRi+1/RRi must not vary by more than 20%, either upward or downward.

3. **Transform the RRi signals into NNi** (with 3 decimal places):

   * Replace outliers using linear interpolation;
   * Replace ectopic beats using linear interpolation.

4. **Truncate the NNi files**, retaining the initial portion based on a specified time value rather than the number of NNi. If the desired minimum duration is not specified, the file with the shortest duration is used as the reference.

## Output

The results are saved in the `data/output/` directory, with two subdirectories:

* `denoised/`: contains the complete NNi sequences;
* `truncated/`: contains the truncated NNi sequences.

Additionally, reports containing basic statistical information about the initial data and the results are saved based on the `truncated/` directory, as shown in the example below:

```text
---------- GROUP: X ----------
Directory: ../data/output/truncated/X
Number of Files: X
Longest Duration (min): X
File with Longest Duration: filename_trunc_X_min.txt
Shortest Duration (min): X
File with Shortest Duration: filename_trunc_X_min.txt
Average Duration (min): X
Quality Threshold (%): 90.0
Average Quality (%): X
Files Below Threshold: 0
Files Above Threshold: 167
```
