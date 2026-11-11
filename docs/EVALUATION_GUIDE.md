# Manual accuracy evaluation protocol

The first ten CVs (`cv_01.txt` through `cv_10.txt`) have reference values in `ground_truth/cvs_01_10.json`, transcribed from the **actual provided synthetic files**. Read the documents and check each labeled field; record your verification in `results/ground_truth_review.csv` before presenting this as a manually verified result.

The eight scored CV fields are:

1. `full_name`
2. `email`
3. `phone`
4. `location`
5. `years_experience` (use the value from SUMMARY, not a calculated date interval)
6. `skills` (all listed skills, case-insensitive set comparison)
7. `work_experience` (every job's title, company and year range, preserving order)
8. `education` (degree plus graduation year)

**Scoring:** each field is one point. 10 CVs × 8 fields = **80 possible points**. Failed extractions contribute 0/8, rather than being omitted. Phone formatting is ignored by comparing digits; other string fields are compared case-insensitively. Skills are compared as sets, and work experiences are ordered.

- `results/accuracy_table_template.csv` has 80 labeled rows with blank model-output columns. Do not enter estimated accuracy numbers.
- `python -m app.main evaluate --predictions ...` calculates field accuracy and failure/speed statistics after actual model runs.
- `python -m app.main compare --local ... --hosted ...` writes the 80-row comparison CSV plus a JSON summary with measured accuracy, processing times and failures.
- Inspect the comparison CSV **by hand** for reasonable normalization and ambiguous model output; only then present the final numbers.
- If a provider doesn't return token usage, its usage fields remain null; do not substitute character counts and call them tokens.

## Vision sample reference

`ground_truth/invoices.json` contains independently transcribed invoice fields from the supplied PNG and the scanned PDF page. **Neither sample explicitly prints a currency**, so `currency=null`. These references allow manual comparison after vision inference.

## Reproducibility

Keep the same 10 documents, schema, prompt version, temperature, and field-scoring policy for both models. Record model names, system details, and run timestamps when reporting real experiments. The server's cold model load can affect timings; run a warm-up or report cold start separately. A failed request still belongs in the denominator.
