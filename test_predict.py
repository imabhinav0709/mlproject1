from src.pipeline.predict_pipeline import CustomData, PredictPipeline

if __name__ == '__main__':
    data = CustomData(
        gender='female',
        race_ethnicity='group B',
        parental_level_of_education='bachelor',
        lunch='standard',
        test_preparation_course='none',
        reading_score='72',
        writing_score='74'
    )

    df = data.get_data_as_data_frame()
    print('Input dataframe:')
    print(df)

    pp = PredictPipeline()
    preds = pp.predict(df)
    print('Predictions:')
    print(preds)
