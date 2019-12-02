import csv

def get_data():
    features = []
    targets = []
    ethnicities = []
    contries = []
    with open(r'dataset\Autistic_Spectrum_Disorde_ Screening_Data_for_Children_Data_Set.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get('age') == 'UNKNOW':
                continue # skip

            temp_ethnicity = row.get('ethnicity')
            if temp_ethnicity not in ethnicities:
                ethnicities.append(temp_ethnicity)

            temp_contry = row.get('contry_of_res')
            if temp_contry not in contries:
                contries.append(temp_contry)

            features.append([
                                int(row.get('a1_score')),
                                int(row.get('a2_score')),
                                int(row.get('a3_score')),
                                int(row.get('a4_score')),
                                int(row.get('a5_score')),
                                int(row.get('a6_score')),
                                int(row.get('a7_score')),
                                int(row.get('a8_score')),
                                int(row.get('a9_score')),
                                int(row.get('a10_score')),
                                int(row.get('age')),
                                0 if row.get('gender') == 'f' else 1,
                                ethnicities.index(temp_ethnicity),
                                0 if row.get('jundice') == 'no' else 1,
                                0 if row.get('austim') == 'no' else 1,
                                contries.index(temp_contry),
                                0 if row.get('used_app_before') == 'no' else 1,
                                int(row.get('result'))
                                # row.get('age_desc') - Only one value
                                # row.get('relation') - Not relevant
            ])

            targets.append(int(0 if row.get('Class/ASD') == 'NO' else 1))

    return features, targets
