import json

index = None
with open(r'cnn_results\index.json', 'r') as f:
    index = list(json.load(f).values())

with open(r'cnn_results\index.csv', 'w') as f:
    cols = index[0].keys()

    f.write(';'.join(cols))
    f.write('\n')

    for i in index:
        f.write(';'.join(map(str, list(i.values()))))
        f.write('\n')
