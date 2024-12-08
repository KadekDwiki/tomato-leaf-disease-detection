# import the necessary packages
from tkinter import *
from PIL import Image
from PIL import ImageTk
from tkinter.filedialog import *
import matplotlib.pyplot as plt
import cv2, os
import numpy as np
from classify import _FeaturesExtraction
from joblib import dump, load
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

classes_dict = {'0': 'Bacterial spot',
                    '1': 'Early blight', 
                    '2': 'healthy', 
                    '3': 'Late blight', 
                    '4': 'Leaf Mold', 
                    '5': 'Septoria leaf spot', 
                    '6': 'Spider mites : Two-spotted_spider_mite',
                    '7': 'Target Spot', 
                    '8': 'mosaic virus', 
                    '9': 'Yellow Leaf_Curl_Virus'}



def _diseasePrediction(img_path):
    print("Extracting Features...")
    # img_path = r'C:\Users\USER\Desktop\korrawiz_ws\Image_processing\project\Tomato leaf disease detection\tomato_disease_detector\val\Tomato___Tomato_Yellow_Leaf_Curl_Virus\1af07f2b-027b-4792-80c5-2c20a4ed538c___YLCV_NREC 0179.JPG'
    features = _FeaturesExtraction(image_path=img_path)
    # print(np.array(features).shape)

    print("predicting...")
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'tomato_disease_model.joblib')
    clf2 = load(filename) 
    y_pred = clf2.predict([features])
    class_pred = classes_dict[str(y_pred[0])]
    
    return class_pred
  
  
def select_image():
    # grab a reference to the image panels
    global image_A, labelA, path
    path = askopenfilename()

    # check if there is file path
    if len(path) > 0:
        image = cv2.imread(path)

        # convert channel bgr to rgb channel
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Mat to PIL format
        image = Image.fromarray(image)
        # PIL to imageTK format
        image = ImageTk.PhotoImage(image)

		# if the panels are None, initialize them
    if image_A is None:
      # the first panel will store our original image
      image_A = Label(image=image)
      image_A.image = image
      image_A.pack()
    # otherwise, update the image panels
    else:
      # update the pannels
      image_A.configure(image=image)
      image_A.image = image
      image_A.pack()
      
      
def showPrediction():
    # grab a reference to the image panels
    global image_A, labelA, path,disease
    # check if there is file path
    if len(path) > 0:
        disease = _diseasePrediction(path)
    else:
      pass
    disease = f'Your tomato leaf might be "{disease}"'
		# if the panels are None, initialize them
    if labelA is None:
      labelA = Label(text=disease)
      labelA.pack()
    # otherwise, update the image panels
    else:
      # update the pannels
      labelA.configure(text=disease)
      labelA.text = disease
      labelA.pack()
  
  # initialize the window toolkit along with the two image panels
root = Tk()
root.title('Tomato Leaf Disease Detection')
image_A = None
labelA = None

# Set the window size and position it at the center of the screen
window_width = 400
window_height = 200

# Get the screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate the position to center the window
position_x = (screen_width // 2) - (window_width // 2)
position_y = (screen_height // 2) - (window_height // 2)

# Set the geometry of the window
root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")

# button the GUI
btn = Button(root, text="Select an image", command=select_image)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")

btn2 = Button(root, text="show disease", command=showPrediction)
btn2.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()
  
  
  