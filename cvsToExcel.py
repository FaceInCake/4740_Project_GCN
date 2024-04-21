import pandas as pd

# Load your CSV file
df = pd.read_csv('./epoch_statistics.csv')

# Create a summary table
summary_df = df.groupby('Configuration ID').mean()  # or any other aggregation you need
summary_df = summary_df.round(decimals={'Train Loss': 4, 'Validation Accuracy': 2, 'Epoch Time': 2})

# Save to Excel
summary_df.to_excel('epoch_statistics.xlsx')
