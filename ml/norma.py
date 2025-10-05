import pandas as pd

k2n = pd.read_csv('datasets/k2.csv')

k2n['category'] = k2n['category'].apply(
    lambda x: 'FALSE POSITIVE' if 'REFUTED' == x else x
)

tessn = pd.read_csv('datasets/tess.csv')

tessn['category'] = tessn['category'].apply(
    lambda x: 'CONFIRMED' if 'CP' == x else x
)

tessn['category'] = tessn['category'].apply(
    lambda x: 'CONFIRMED' if 'KP' == x else x
)

tessn['category'] = tessn['category'].apply(
    lambda x: 'CANDIDATE' if 'PC' == x else x
)

tessn['category'] = tessn['category'].apply(
    lambda x: 'CANDIDATE' if 'APC' == x else x
)

tessn['category'] = tessn['category'].apply(
    lambda x: 'FALSE POSITIVE' if 'FP' == x else x
)

tessn['category'] = tessn['category'].apply(
    lambda x: 'FALSE POSITIVE' if 'FA' == x else x
)

#k2n.to_csv('datasets/k2n.csv', index=False)

#tessn.to_csv('datasets/tessn.csv', index=False)

kepler = pd.read_csv('datasets/kepler.csv')

print(kepler['category'].unique())
print(k2n['category'].unique())
print(tessn['category'].unique())

