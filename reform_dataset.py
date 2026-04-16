import pandas as pd

df = pd.read_csv('original_dataset.csv')
output = df[['#', 'text', 'label']]
output.to_csv('reformed_dataset.csv', index=False)