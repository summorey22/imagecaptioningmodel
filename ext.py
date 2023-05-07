# importing the zipfile module
from zipfile import ZipFile
import argparse
import os

parser = argparse.ArgumentParser(
                    prog='PyExtractor by ADINA',
                    description='Extracts files from zip',
)
parser.add_argument('-f', type=str, required=True, help='Path to the zip')
parser.add_argument('-d', type=str, help='Path to unzip')
args = parser.parse_args()
ZIP_PATH = args.f
FILE_PATH = args.d or os.path.dirname(ZIP_PATH)
  
# loading the temp.zip and creating a zip object
with ZipFile(ZIP_PATH, 'r') as zObject:
  
    # Extracting all the members of the zip 
    # into a specific location.
    zObject.extractall(
        path=FILE_PATH)
