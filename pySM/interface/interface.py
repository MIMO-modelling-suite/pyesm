from pySM.src.util import prettify
from pySM.src.model import Model


# create a model instance
# model settings automatically loaded
# eventual cleanup of previus databases in the case study path
# eventual generation of blank sets.xlsx file in the defined case study folder

test = Model(
    file_settings_name='model_settings.json',
    clean_database=False,
    generate_sets_file=True
)

# after filling blank sets.xlsx, importing it in the model
test.load_sets_data()


# ancillary checks
# test.sets