import pytest
import pandas as pd
import constants

def df_plugin():
    return None

# Creating a Dataframe object 'pytest.df' in Namespace
def pytest_configure():
    pytest.df = df_plugin()
    
@pytest.fixture
def df_path():
    return constants.DF_PATH