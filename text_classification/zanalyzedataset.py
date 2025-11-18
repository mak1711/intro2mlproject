import pandas as pd
from collections import Counter

# --------- CONFIGURATION ---------
csv_file = "/home/kan/ML/data/robot_commands_no_mixed_language.csv"  # <-- Replace with your CSV
text_col = "text"                      # Column name containing the text
label_col = "label"                    # Column name containing labels
# ---------------------------------

# Load CSV
df = pd.read_csv(csv_file)

print(f"Total samples: {len(df)}")

# Count duplicate texts
duplicate_texts = df[df.duplicated(subset=[text_col], keep=False)]
num_duplicates = len(duplicate_texts)
unique_texts = df[text_col].nunique()

print(f"Unique texts: {unique_texts}")
print(f"Duplicate entries: {num_duplicates}")
print(f"Percentage of duplicates: {100 * num_duplicates / len(df):.2f}%")

# Optional: show the most common repeated texts
text_counts = df[text_col].value_counts()
repeated_texts = text_counts[text_counts > 1]
print("\nMost common repeated texts:")
print(repeated_texts.head(10))

# Optional: check if labels differ for the same text
conflict_texts = []
for text, group in df.groupby(text_col):
    if group[label_col].nunique() > 1:
        conflict_texts.append((text, group[label_col].tolist()))

if conflict_texts:
    print("\nTexts with conflicting labels:")
    for text, labels in conflict_texts:
        print(f"Text: {text} | Labels: {labels}")
else:
    print("\nNo conflicting labels detected for repeated texts.")

