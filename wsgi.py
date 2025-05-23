import sys
import os

# Add your project directory to the sys.path
project_home = os.path.abspath(os.path.dirname(__file__))
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Import the Flask app instance
# Assuming your Flask app instance is named 'app' in 'app.py'
from app import app as application  # noqa: E402

# If you have any other setup needed for your app environment,
# like setting environment variables, you can do it here.
# For example:
# os.environ['MY_APP_SETTING'] = 'some_value'
