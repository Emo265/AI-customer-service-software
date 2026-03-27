# Installation

## Step 1: Clone the Repository
```bash
git clone https://github.com/Emo265/AI-customer-service-software.git
cd AI-customer-service-software
```

## Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

## Verification
To verify the installation was successful, run:
```bash
python -c "import flask, tensorflow, sklearn; print('All dependencies installed successfully!')"
```