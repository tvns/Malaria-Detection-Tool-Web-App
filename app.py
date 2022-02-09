import streamlit as st 
import numpy as np
import tensorflow as tf 
from PIL import Image
import os
import time
import cv2


st.set_page_config(page_title="Malaria Detection Tool", page_icon="image.jpeg", layout='centered', initial_sidebar_state='expanded')

st.sidebar.markdown('#### Diagnostic Lab:')
st.sidebar.image("image.jpeg",width=300)
st.sidebar.markdown('###### Source: [Malaria](https://www.parashospitals.com/wp-content/uploads/2017/05/malaria.jpg)')

st.sidebar.markdown('### Lets start the Diagnosis:' )

image_file = st.sidebar.file_uploader("Upload an Blood sample image (jpg, png or jpeg)", type=['jpg','png','jpeg'])



if image_file is not None:
		uploaded_image = Image.open(image_file)



html_templ = """

	<h1 style="color:purple"> Malaria Detection Tool </h1>
	</div>

	"""

st.markdown(html_templ,unsafe_allow_html=True)

st.write('A simple proposal for an entirely automated Convolutional Neural Network (CNN) based model for the diagnosis of malaria from the microscopic blood smear images')

st.image('https://miro.medium.com/max/1146/1*EJWcvx7ufm3wYpIeoqMa1Q.jpeg')

st.markdown('###### Source: [🦟Malaria](https://miro.medium.com/max/1146/1*EJWcvx7ufm3wYpIeoqMa1Q.jpeg)')

st.markdown('#### Motivation behind developing this Tool:')

st.markdown("""

	###### Malaria is a life-threatening disease that is spread by the Plasmodium parasites

	###### The standard way of diagnosing malaria is by examining microscopic blood smears images by trained microscopists 

	###### The need for this type of trained personnel can be greatly reduced with the development of an automatic accurate and efficient model deep learning model




	"""

	)


st.markdown('#### Disclaimer:')

st.markdown("""

	###### The Author is not a Doctor
	###### This tool is just a showcase of practical experience in Convolutional Neural Networks using Python  and Streamlit
	###### You should always consult a Doctor to get your Health Check
	###### Please don't take the outcome of this tool seriously and NEVER consider it Valid. There is no clinical value in its Diagnosis

	"""
	)


st.markdown('### Visit our Diagnostic Lab on the sidebar and submit your Blood sample')

col1, col2 = st.columns([1.5,3])

with col1:

	st.image('https://previews.123rf.com/images/nopember30/nopember301610/nopember30161000009/64151757-cartoon-businessman-pointing-for-direction-isolated.jpg')
	st.markdown('###### Source: [Direction to the Sidebar](https://previews.123rf.com/images/nopember30/nopember301610/nopember30161000009/64151757-cartoon-businessman-pointing-for-direction-isolated.jpg)')

with col2:

	st.markdown("""

		##### Instructions:

		###### Before you visit the Lab, take an Blood sample Image of jpg, png, jpeg format

		###### Submit the Blood sample Image file for diagnosis

		###### Wait for a few seconds to get the results

		"""
		)

	st.warning("Please don't take the outcome of this tool seriously and NEVER consider it Valid. There is no clinical value in its Diagnosis")

if st.sidebar.button('Submit'):


	new_img = np.array(uploaded_image)
	image_size = (130,130)
	new_img = cv2.resize(new_img, image_size)
	new_img = new_img/255

	final_img = new_img.reshape(1,130,130,3)



	model = tf.keras.models.load_model("Malaria_detector.h5")


	diagnosis = model.predict_classes(final_img)
	

	my_bar = st.sidebar.progress(0)


	for percent_complete in range(100):

		time.sleep(0.025)
					
		my_bar.progress(percent_complete + 1)


	if diagnosis == 1:

		st.sidebar.success("RESULT: Malaria Negative")

	else:

		st.sidebar.error("RESULT: Malaria Positive") 



st.markdown("")
st.markdown("### 😊Stay Healthy and Safe")


st.markdown("")
st.markdown('#### Model Summary:')
st.markdown("""

	###### The data is taken from the official NIH website: https://lhncbc.nlm.nih.gov/publication/pub9932

	###### It contains 27,558 images divided into 2 folders: Parasitized and Uninfected

	###### The model expects a fixed image size of (130, 130, 3). Cropping or Padding to the image will be done if the image is of a different size

	###### This is a 10 layer CNN Sequential model (3 Convolution, 3 Pooling, 2 Dense, 1 Dropout, and 1 Flattening layers)

	###### The model achieved a very good F1-score of 0.95 and we can tune the model better to achieve an even better F1-score

	###### Unfortunately, upon testing the model we have encountered 46 False-Positive cases and 86 False-Negative cases

	###### Hence, further Fine Tuning is required to achieve an accurate model

	"""
	)


st.markdown("")
with st.expander('Click here to know about the Author ', expanded=True):

	photo, info = st.columns([1, 1])

	with photo:

		st.image('IMG_6959.jpg', width=300)

	with info:

		st.markdown('### Karthik Thallam')
		st.markdown('##### Machine Learning Engineer')
		st.caption('Wish to connect?')
		st.markdown('Linkedin : [Karthik Thallam](https://www.linkedin.com/in/karthikthallam/)')
		st.write('📧: karthik.thallam1@gmail.com')


st.markdown("")
with st.expander('Click here to see the References', expanded=False):

	st.markdown('Reference 01: https://www.mdpi.com/2075-4418/10/5/329/htm')
	st.markdown('Reference 02: https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html')
	
