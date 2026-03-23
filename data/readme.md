# Dataset Overview
raw_data/
    data from screening, used for annotation
    * asreview_dataset_all_Psychedelic Study.csv_
        * 9645 records
        * 3336 includes (=about psychedlic substances)
        * 2370 excludes (=not about psychedlic substances)
        * 3939 also excluded because after stopping criteria in ASReview
        * search completed via `asreview`
        * search performed in December 2023
    
    * dataset_relevan_cleaned.csv_
        * 3336 records
        * only included studies
        * cleaned relevant data from `asreview_dataset_all_Psychedelic Study.csv`
            * add field text = title + .^\n + abstract
            * add pubmed url from doi (if doi is available)
        
    * asreview_dataset_all_Psychedelic Study.xlsx
        * mostly the same as `asreview_dataset_all_Psychedelic Study.csv`
        * additional column with data source automaticall infered from columns
        * some stats

prepared_data/
    data for model training, after annotation
    * psychedelic_study_all.csv
        * 5642 records
        * 3313 includes (=about psychedlic substances)
            -> -23 previously included but classified as excluded during annotation
        * 2330 excludes (=not about psychedlic substances) 
            -> -64 removed due to missing abstracts
            -> +23 previously included but classified as excluded during annotation


    * psychedelic_study_relevant.csv
        * 3313 records
        * only included studies

