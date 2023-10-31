from .file_utils import load_dataframe_file, load_json_file, read_text_file


READ_FILE_FN = {
    ".csv": load_dataframe_file,
    ".xlsx": load_dataframe_file,
    ".json": load_json_file,
    ".txt": read_text_file,
}
