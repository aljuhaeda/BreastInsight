# BreastInsight: AI Image Classifier

BreastInsight is an AI Image Classifier for Breast Ultrasound Images. It uses a Convolutional Neural Network (CNN) trained on the provided dataset to predict the state of ultrasound images as normal, benign, or malignant.

## Project Files

```
BreastInsight/
│
├── BreastInsight.h5
├── BreastInsight.ipynb
├── Dataset.txt
├── Dataset_BUSI_with_GT/
│   ├── [Dataset files and folders]
│
└── model.fix
```

- **`BreastInsight.h5`**: Trained CNN model for breast ultrasound image classification.

- **`BreastInsight.ipynb`**: Jupyter Notebook containing the code for training the model and performing predictions.

- **`Dataset.txt`**: Information or description about the dataset.

- **`Dataset_BUSI_with_GT/`**: Directory containing the dataset files. (Please upload the dataset files to this directory)

- **`model.fix`**: Model file (you may provide more details on what this file contains).

## Dataset

The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset). Please make sure to follow the dataset's license and terms of use.

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/BreastInsight.git
   ```

2. **Navigate to the Project Directory:**

   ```bash
   cd BreastInsight
   ```

3. **Run the Jupyter Notebook:**

   Open `BreastInsight.ipynb` in a Jupyter environment and run the cells to explore the code, train the model, and make predictions.

## Web Application Usage

To run the web application locally for image classification:

### Prerequisites

- Python 3.x
- It's recommended to use a virtual environment.

### Setup

1.  **Clone the Repository (if you haven't already):**
    ```bash
    git clone https://github.com/your-username/BreastInsight.git
    cd BreastInsight
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    ```
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure `BreastInsight.h5` model file is present in the root directory before running the application, or update `MODEL_PATH` in `app.py` if it's located elsewhere.)

### Running the Application

1.  **Start the Flask Server:**
    ```bash
    python app.py
    ```

2.  Open your web browser and navigate to `http://127.0.0.1:5000/`.

You should see a page where you can upload an image for classification. The result will be displayed on a new page.

## Deployment to PythonAnywhere (Free Tier)

This guide explains how to deploy the BreastInsight Flask application to PythonAnywhere's free tier.

### 1. Sign up for PythonAnywhere

*   Go to [https://www.pythonanywhere.com/](https://www.pythonanywhere.com/) and sign up for a free "Beginner" account.

### 2. Upload Your Files

*   Once logged in, go to the **"Files"** tab.
*   You can upload your files one by one using the **"Upload a file"** button. Make sure to upload:
    *   `app.py`
    *   `BreastInsight.h5` (the model file)
    *   `requirements.txt`
    *   `wsgi.py`
*   Create a new directory named `templates` (using the "New directory" input at the top of the files page, e.g., `/home/YOUR_USERNAME/templates/`) and upload `index.html` and `result.html` into it.
*   Create another new directory named `uploads` (e.g., `/home/YOUR_USERNAME/uploads/`). This folder needs to be writable by your web app. Note that on the free tier, file uploads might be ephemeral or have limitations. The current app saves files temporarily here for prediction.
*   (Alternatively, if you are familiar with Git, you can clone your repository into PythonAnywhere using a Bash console - see their documentation for details.)

### 3. Set Up the Web App

*   Go to the **"Web"** tab from the dashboard.
*   Click on **"Add a new web app"**.
*   Follow the prompts. When asked, select **"Flask"** as your Python web framework.
*   Choose the Python version that matches your project (e.g., Python 3.9 or 3.10 - check what PythonAnywhere offers on the free tier).
*   PythonAnywhere will create a default Flask app. We'll modify its configuration.

### 4. Configure the WSGI File

*   On the **"Web"** tab for your new app, find the **"Code"** section.
*   Click on the link next to **"WSGI configuration file"**. This will likely be something like `/var/www/YOUR_USERNAME_pythonanywhere_com_wsgi.py`.
*   Edit this file. **Replace its entire content** with the path to *your* `wsgi.py` file. For example, if your username is "myuser" and you uploaded your files to the default directory (`/home/myuser/`), the path in this system WSGI file should point to your `wsgi.py` like this:
    ```python
    # This is an example, replace YOUR_USERNAME
    path = '/home/YOUR_USERNAME/wsgi.py' 
    # or if your project is in a subdirectory, e.g., 'myproject':
    # path = '/home/YOUR_USERNAME/myproject/wsgi.py' 
    
    # The rest of the file should be simple, PythonAnywhere often provides a template.
    # A minimal version that should work if your wsgi.py is correctly set up:
    import sys
    
    # Add your project directory to sys.path
    project_home = '/home/YOUR_USERNAME/' # Adjust if your app is in a subdirectory
    if project_home not in sys.path:
        sys.path = [project_home] + sys.path
    
    # Import the application object from your wsgi.py
    from wsgi import application # This assumes your wsgi.py defines 'application'
    ```
*   **Important Note:** The key is that PythonAnywhere's system WSGI file must correctly locate and import the `application` object from *your* `wsgi.py` file that you uploaded. Your `wsgi.py` already points to `app.py`.
*   Save the changes to this system WSGI file.

### 5. Set the Working Directory (Optional but good practice)

*   Still on the **"Web"** tab, in the **"Code"** section, you might see an option for **"Working directory"**. Set this to your project's root directory (e.g., `/home/YOUR_USERNAME/` or `/home/YOUR_USERNAME/myproject/` if you used a subdirectory).

### 6. Install Dependencies

*   Go to the **"Consoles"** tab and start a **"Bash"** console.
*   It's recommended to create a virtual environment for your web app on PythonAnywhere. For example (replace `myenv` with your preferred name and check Python version):
    ```bash
    mkvirtualenv --python=/usr/bin/python3.9 myenv 
    ```
*   Once the virtual environment is created and activated (it usually activates automatically), install your packages:
    ```bash
    pip install -r requirements.txt
    ```
*   After installation, you can type `deactivate` to exit the virtualenv for now in the console.
*   Go back to the **"Web"** tab. In the **"Virtualenv"** section, enter the path to your virtual environment, e.g., `/home/YOUR_USERNAME/.virtualenvs/myenv`.

### 7. Reload and Access Your Web App

*   On the **"Web"** tab, click the green **"Reload YOUR_USERNAME.pythonanywhere.com"** button.
*   Your application should now be live at `https://YOUR_USERNAME.pythonanywhere.com`.

### 8. Free Tier Notes

*   Free PythonAnywhere apps may "sleep" if they don't receive traffic for a while and might need a manual reload or have a short startup delay on the first visit after being idle.
*   Custom domains are not typically available on the free tier.
*   Check PythonAnywhere's current free tier limitations for CPU, memory, and disk space, especially considering the size of your model file.

## Model Usage

- Load the `BreastInsight.h5` model file into your application or script.

- Use the model to predict the state of breast ultrasound images.

## Contributing

Feel free to contribute to the project. If you have suggestions, improvements, or found a bug, please open an issue or create a pull request.

## License

BreastInsight is licensed under the [MIT License](LICENSE).
