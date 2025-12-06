import pandas as pd
import os
# path to your existing results CSV
RESULT_CSV = "model_search2_results_nospm.csv"

# new evaluation result
new_result = {
    "run": "mild_reg_2ep",  # model name
    "accuracy": 0.7818     # the value you got
}

# load existing CSV or create new
if os.path.exists(RESULT_CSV):
    df = pd.read_csv(RESULT_CSV)
else:
    df = pd.DataFrame(columns=["run", "accuracy"])

# append new row
df = pd.concat([df, pd.DataFrame([new_result])], ignore_index=True)

# save back to CSV
df.to_csv(RESULT_CSV, index=False)
print(f"Appended result to {RESULT_CSV}")

