import os
import logging
from churn_library import *


os.environ['QT_QPA_PLATFORM']='offscreen'


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


# @pytest.fixture(scope="module")
# def df_path():
#     return './data/bank_data.csv'

ls

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(df_path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
        
    pytest.df = df
    return df


# def test_eda(perform_eda):
#     '''
#     test perform eda function
#     '''
#     df = pytest.df
#     try:
#         perform_eda(df)
#         logging.info("SUCCESS: Eda performed successfully")
#     except Exception as err:
#         logging.error("ERROR: Error occurred when running eda")
#         raise err
        
#     try:
#         assert os.path.isfile("./images/eda/df_description.png")
#         assert os.path.isfile("./images/eda/Churn.png")
#         assert os.path.isfile("./images/eda/Customer_Age.png")
#         assert os.path.isfile("./images/eda/Marital_Status.png")
#         logging.info("SUCCESS: All eda files saved successfully")
#     except AssertionError as err:
#         logging.error("ERROR: Not all eda files are saved")
#         raise err


# def test_encoder_helper(encoder_helper):
#     '''
#     test encoder helper
#     '''
#     try:
#         encoded_df = encoder_helper(df, constants.CATEGORY_LST, constants.RESPONSE)
#         logging.info("SUCCESS: Encoding successful")
#     except Exception as err:
#         logging.error("ERROR: Error occurred when encoding")
#         raise err
        
#     try:
#         assert len(encoded_df.columns) == (len(constants.CATEGORY_LST) + len(df.columns))
#         logging.info("SUCCESS: Columns encoded s expected")
#     except AssertionError as err:
#         logging.error("ERROR: Not all eda files are saved")
#         raise err


# def test_perform_feature_engineering(perform_feature_engineering):
#     '''
#     test perform_feature_engineering
#     '''
#     try:
#         X_train, X_test, y_train, y_test = perform_feature_engineering(df, constants.CATEGORY_LST, constants.RESPONSE)
#         request.config.cache.set('cache_X_train', X_train)
#         request.config.cache.set('cache_X_test', X_test)
#         request.config.cache.set('cache_y_train', y_train)
#         request.config.cache.set('cache_y_test', y_test)
#         logging.info("SUCCESS: Feature engineering perform successfully")
#     except Exception as err:
#         logging.error("ERROR: Error occurred when encoding")
#         raise err
        
#     try:
#         assert df.shape[0] == (X_train.shape[0] + X_test.shape[0])
#         assert df.shape[0] == (y_train.shape[0] + y_test.shape[0])
#         assert df.shape[1] == X_train.shape[1] == X_test.shape[1]
#         logging.info("SUCCESS: X and y split as expected")
#     except AssertionError as err:
#         logging.error("ERROR: Split shape is incorrect")
    
    
# def test_train_models(train_models):
#     '''
#     test train_models
#     '''
#     X_train = request.config.cache.get('cache_X_train', None)
#     X_test = request.config.cache.get('cache_X_test', None)
#     y_train = request.config.cache.get('cache_y_train', None)
#     y_test = request.config.cache.get('cache_y_test', None)
#     train_models(X_train, X_test, y_train, y_test, constants.MODELS_OUT_PATH)


if __name__ == "__main__":
    pass