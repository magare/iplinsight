# Deploying to AWS Elastic Beanstalk

This document provides instructions for manually deploying this Streamlit app to AWS Elastic Beanstalk.

## Preparation

1. Ensure all required files are in this directory:
   - application.py (entry point)
   - Procfile
   - requirements.txt
   - .ebextensions folder with configuration files
   - All application files (app.py, components, utils, etc.)

2. Create a ZIP file of this entire directory

## Deployment Steps

1. Log in to AWS Management Console
2. Navigate to Elastic Beanstalk
3. Click "Create Application"
4. Fill in the details:
   - Application name: IPL-Streamlit-App (or your preferred name)
   - Platform: Python
   - Platform branch: Python 3.9 running on 64bit Amazon Linux 2
   - Platform version: The latest version is recommended
   - Application code: Upload your code > Local file > Upload the ZIP file you created

5. Configure more options:
   - Instance type: At least t2.small is recommended
   - Environment variables: Add any if needed

6. Create the application

## Post-Deployment

Once deployment is complete, you can access your application at the URL provided by Elastic Beanstalk.

## Troubleshooting

If you encounter issues:
1. Check the application logs in the Elastic Beanstalk console
2. Verify that all required dependencies are in requirements.txt
3. Ensure your Procfile is configured correctly
