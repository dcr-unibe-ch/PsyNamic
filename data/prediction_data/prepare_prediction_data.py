
import pandas as pd

# Get the data that has been manually annotated as relevant (in ASReview) but was not manually annotated for classification/Ner

def determine_not_annotated():
    data = 'data/prepared_data/psychedelic_study_relevant.csv'
    ids = 'data/prepared_data/round2_ids.txt'

    df = pd.read_csv(data)

    # read ids
    with open(ids, 'r', encoding='utf-8') as f:
        ids = f.readlines()
    ids = [int(x.strip()) for x in ids]
    print(f'Number of annotated studies: {len(ids)}')

    to_be_annotated = []

    for index, row in df.iterrows():
        if row['id'] in ids:
            continue
        else:
            to_be_annotated.append(row)

    df = pd.DataFrame(to_be_annotated)
    print(f'Number of studies to be annotated: {len(df)}')
        
    df.to_csv('data/prediction_data/studies_relevant_unannotated_20240101_00-00-00.csv', index=False)

def get_all_relevant_studies_with_info():
    data = 'data/prepared_data/psychedelic_study_relevant.csv'
    studies_info = '/home/veral/PsyNamic/data/raw_data/dataset_relevant_cleaned.csv'

    df = pd.read_csv(data)
    studies_info_df = pd.read_csv(studies_info)

    # merge with studies info to get the title and abstract, to_be_annotated contains 'id', studies_info_df contains 'record_id'
    # add all columns from studies_info_df that are not in df, and merge on 'id' and 'record_id'
    # do not keep the id column of the studies info
    studies_info_df = studies_info_df.drop(columns=['id','text'])
    df = pd.merge(df, studies_info_df, left_on='id', right_on='record_id', how='left')
    # drop the record_id column
    df = df.drop(columns=['record_id'])
    print('Total number of relevant studies: ', len(df))

    df.to_csv('data/prediction_data/studies_relevant_with_info_20240101_00-00-00.csv', index=False)

def main():
    determine_not_annotated()
    get_all_relevant_studies_with_info()

if __name__ == '__main__':
    main()