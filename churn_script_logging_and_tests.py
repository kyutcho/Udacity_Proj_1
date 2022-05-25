import os
import logging
from churn_library import *


os.environ['QT_QPA_PLATFORM']='offscreen'


logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(request):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(constants.DF_PATH)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    except Exception as err2:
        logging.error("ERROR: Error occurred when importing data")
        raise err2

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err
        
    request.config.cache.set('cache_df', df)
    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
        logging.info("SUCCESS: Eda performed successfully")
    except Exception as err:
        logging.error("ERROR: Error occurred when running eda")
        raise err
        
    try:
        assert os.path.isfile(constants.EDA_OUT_PATH+"df_description.png")
        assert os.path.isfile(constants.EDA_OUT_PATH+"Churn.png")
        assert os.path.isfile(constants.EDA_OUT_PATH+"Customer_Age.png")
        assert os.path.isfile(constants.EDA_OUT_PATH+"Marital_Status.png")
        logging.info("SUCCESS: All eda files saved successfully")
    except AssertionError as err:
        logging.error("ERROR: Not all eda files are saved")
        raise err


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''
    try:
        encoded_df = encoder_helper(df, constants.CATEGORY_LST, constants.RESPONSE)
        logging.info("SUCCESS: Encoding successful")
    except Exception as err:
        logging.error("ERROR: Error occurred when encoding")
        raise err
        
    try:
        assert len(encoded_df.columns) == (len(constants.CATEGORY_LST) + len(df.columns))
        logging.info("SUCCESS: Columns encoded s expected")
    except AssertionError as err:
        logging.error("ERROR: Not all eda files are saved")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, constants.CATEGORY_LST, constants.RESPONSE)
        logging.info("SUCCESS: Feature engineering perform successfully")
    except Exception as err:
        logging.error("ERROR: Error occurred when encoding")
        raise err
        
    try:
        assert df.shape[0] == (X_train.shape[0] + X_test.shape[0])
        assert df.shape[0] == (y_train.shape[0] + y_test.shape[0])
        assert df.shape[1] == X_train.shape[1] == X_test.shape[1]
        logging.info("SUCCESS: X and y split as expected")
    except AssertionError as err:
        logging.error("ERROR: Split shape is incorrect")
    
    
def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        train_models(X_train, X_test, y_train, y_test, constants.MODELS_OUT_PATH)
        logging.info("SUCCESS: Traing and testing completed successfully")
    except Exception as err:
        logging.error("ERROR: Error occurred when training and testing")
        raise err

    try:
        assert os.path.isfile(constants.RESULTS_OUT_PATH+"roc_auc.png")
        assert os.path.isfile(constants.RESULTS_OUT_PATH+"feature_importance.png")
    except AssertionError as err:
        logging.error("ERROR: Expected results images not saved")


if __name__ == "__main__":
    DATA = test_import(import_data)
    test_eda(perform_eda, DATA)
    DATA = test_encoder_helper(encoder_helper, DATA)
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        perform_feature_engineering, DATA)
    test_train_models(train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)