import pandas as pd

# Load CSVs
texts_csv = pd.read_csv("/home/kan/ML/text_classification/label5_texts.csv")  # only text column
data_csv = pd.read_csv("/home/kan/ML/data/augmented.csv")  # full dataset with text + label

# Get indices of rows where label == 8
label8_indices = data_csv[data_csv['label'] == 8].index

# 1) Replace the existing label-8 texts --------------------------
num_to_replace = min(len(label8_indices), len(texts_csv))

data_csv.loc[label8_indices[:num_to_replace], 'text'] = texts_csv['text'].values[:num_to_replace]

print(f"Replaced {num_to_replace} existing label-8 texts.")

# 2) Add 1000 new rows with label = 8 ---------------------------
num_extra = 6000

# Take first 1000 texts (or fewer if not enough)
extra_texts = texts_csv['text'].head(num_extra)

# Build a new dataframe for the extra rows
extra_rows = pd.DataFrame({
    'text': extra_texts,
    'label': [8] * len(extra_texts)
})

# Append to dataset
data_csv = pd.concat([data_csv, extra_rows], ignore_index=True)

print(f"Added {len(extra_rows)} new label-8 rows.")

# Save the updated CSV
data_csv.to_csv("updated_data2.csv", index=False)
print("Saved as updated_data2.csv")

