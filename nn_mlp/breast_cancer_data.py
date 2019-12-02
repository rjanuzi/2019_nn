import csv

# Features:
#   Age
#   BMI
#   Glucose
#   Insulin
#   HOMA
#   Leptin
#   Adiponectin
#   Resistin
#   MCP.1

# Value:
#   Classification

def get_data():
    features = []
    targets = []
    with open(r'dataset\Breast_Cancer_Coimbra_Data_Set.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            features.append([
                                float(row.get('Age')),
                                float(row.get('BMI')),
                                float(row.get('Glucose')),
                                float(row.get('Insulin')),
                                float(row.get('HOMA')),
                                float(row.get('Leptin')),
                                float(row.get('Adiponectin')),
                                float(row.get('Resistin')),
                                float(row.get('MCP.1'))
            ])

            targets.append(int(row.get('Classification')))

    return features, targets
