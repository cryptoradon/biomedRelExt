# Biomedical Relation Extraction

Preprocessing phase of the Biomedical Relation Extraction project.

## Prerequisites

Ensure you have Python 3.7.9 installed on your machine.

## Setup

Follow these steps to set up the project environment:

### Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

## Install Dependencies

Install all dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Running the preprocessing

There are four scripts that are to be run in order for the preprocessing for the CDR data:

```bash
python preprocessCDR1.py
python preprocessCDR2.py
python preprocessCDR3.py
python preprocessCDR4.py
```

The output will be ``` tokenizedInput.pkl ``` in ``` Preprocessed/(CDRTest|CDRTraining)/ ``` based on the input file and destination mentioned in the codes. 

Please note that you need to run the pipeline in order to get the preprocessed data as it is not readily availible in the repository. ``` preprocessCDR1.py ``` will automatically create the directory. Alternatively, you can check the Google Drive folder used by colab for the final form of the preprocessed data. 

Also note that preprocessCDR1 and preprocessCDR4 takes a very long time to run on a CPU, so the colab notebook has an implementation of this method and if you upload the ``` input.pkl ``` file to the corresponding folder, the notebook can determine the ``` tokenizedInput.pkl ``` in a very short amount of time.

There are three scripts that are to be run in order for the preprocessing for the CDR data:

```bash
python preprocessCHR1.py
python preprocessCHR2.py
python preprocessCHR3.py
```

The output will be ``` tokenizedInput.pkl ``` in ``` Preprocessed/(CHRTest|CHRTraining)/ ``` based on the input file and destination mentioned in the codes. 

Please note that you need to run the pipeline in order to get the preprocessed data as it is not readily availible in the repository. ``` preprocessCHR1.py ``` will automatically create the directory. Alternatively, you can check the Google Drive folder used by colab for the final form of the preprocessed data. 

Also note that preprocessCHR1 and preprocessCHR4 takes a very long time to run on a CPU, so the colab notebook has an implementation of this method and if you upload the ``` input.pkl ``` file to the corresponding folder, the notebook can determine the ``` tokenizedInput.pkl ``` in a very short amount of time.

## Running the Knowledge Representation Extraction

You need to run one script in order to extract the knowledge representation data:

```bash
python tripleExtraction.py
```

The outputs will be contained in  ``` Knowledge_Representation/ ```. You can also find the trained transE model in that folder.

## Deactivation

Deactivate the environmnent with this command

```bash
deactivate
```