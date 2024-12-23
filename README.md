# 🚦Traffic Sign Classification through Images

A simple app template for you to classify the traffic signs and also our Computer Vision Introduction - CS231.P12 course project!

[![Open in streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://traffic-sign-classification-through-images.streamlit.app/)

<img src="https://github.com/user-attachments/assets/70f8dd7f-f775-42d5-9a60-5748ce2edf4f" width="300" height="auto" />



### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```
2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
### Members
| Student Name     | Student ID |
|------------------|------------|
| Nguyen Duy Hoang | 22520467   |
| Ha Huy Hoang     | 22520460   |

### Project Structure

1. **data**: Store traffic sign images and labels in train, test, and demo directories

2. **extras**: Contains two folders and one file:
   - One folder for code containing scripts for tasks such as checking the percentage of traffic signs in the image frame, renaming multiple image files, etc.
   - Another folder containing Vietnam's legal documents regarding traffic signs.
   - `requirements.txt` file storing Mr. Mai Tien Dung's requirements for slides and report.

3. **grid_search_results**: Store parameter tuning attempts and different data processing methods

4. **joblib**: Store the best model files, images in np.array format and extracted features

5. **model**: Contains the main notebook file (main.ipynb) for Vietnam traffic sign classification program

6. **report**: Contains files related to the report, including .tex, .pdf, and other resources.

7. **slide**: Contains the slide presentation for the project.
  
8. **deploy_web.py**: Python file containing code to deploy website using Streamlit framework

9. **requirements.txt**: Store necessary dependencies and requirements
