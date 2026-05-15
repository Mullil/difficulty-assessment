"""Keep only the columns the rest of the pipeline expects (#, text, label) and
drop everything else from the raw training CSV."""

import pandas as pd

df = pd.read_csv('raw_trainset.csv')
output = df[['#', 'text', 'label']]
output.to_csv('preprocessed_trainset.csv', index=False)