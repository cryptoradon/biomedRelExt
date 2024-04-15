# Project Title

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

# Install Dependencies

Install all dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

# Running the project

There are three scripts that are to be run in order:

```bash
python preprocessCDR1.py
python preprocessCDR2.py
python preprocessCDR3.py
```

The output will be ``` input.pkl ```

# Deactivation

Deactivate the environmnent with this command

```bash
deactivate
```