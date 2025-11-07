
import requests
import streamlit as st
import json
from streamlit_lottie import st_lottie
from demo import test_genomic_model
st.set_page_config(page_title="Genomic Interpreter", page_icon="🧬", layout="wide")

def load_lottieurl(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
#-----  LOAD ASSETS  -----
lottie_dna = load_lottieurl("D:/PS/GI code/1d-swin-main/images/dna2.json")
  


#st.header("GENOMIC INTERPRETER")
st.title("GENOMIC INTERPRETER")

#-----  HEADER SECTION -----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2) 
    with left_column:
        dna_sequence = st.text_area("Enter DNA Sequence","     ")

        if st.button("Analyze"):
            if not dna_sequence:
                st.warning("Please enter a DNA sequence.")
            else:
            # Perform genomic interpretation
                result = analyze_dna_sequence(dna_sequence)

            # Display the results
            #st.success(f"Genomic Analysis Result:\n\nDNA Sequence Length: {result} base pairs")
        #st.header("What I do")
        #st.write("##")
    with right_column:
        st_lottie(lottie_dna,speed=2)


'''
# frontend.py
import streamlit as st
import json
from streamlit_lottie import st_lottie
from demo import test_genomic_model, visualize_dna_sequence, extract_white_part

st.set_page_config(page_title="Genomic Interpreter", page_icon="dotted_line_face", layout="wide")

def load_lottieurl(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

#-----  LOAD ASSETS  -----
lottie_dna = load_lottieurl("D:/PS/GI code/1d-swin-main/images/dna2.json")

#st.header("GENOMIC INTERPRETER")
st.title("GENOMIC INTERPRETER")

#-----  HEADER SECTION -----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2) 
    with left_column:
        dna_sequence = st.text_area("Enter DNA Sequence", "     ")

        if st.button("Analyze"):
            if not dna_sequence:
                st.warning("Please enter a DNA sequence.")
            else:
                # Perform genomic interpretation
                output = test_genomic_model(dna_sequence)

                # Visualize the genomic model output and display the white part
                visualize_dna_sequence(dna_sequence)
                white_part = extract_white_part(output)
                st.image([output, white_part], caption=['Genomic Model Output', 'White Part'], use_container_width=True)

    with right_column:
        st_lottie(lottie_dna, speed=2)
  
'''




'''
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Import backend functions from swin1d backend code
from swin1d.module import swin_1d_block
from demo import generate_random_dna, onehot_encoder, extract_white_part, visualize_dna_sequence, repeat_sequence

# Function to analyze DNA sequence using Swin1D model
def analyze_dna_sequence(dna_sequence):
    # Encode the input DNA sequence
    encoded_sequence = onehot_encoder(dna_sequence)
    
    # Generate Swin1D model
    model = swin1d_block(4)
    
    # Get the output from the model
    output = model(encoded_sequence)
    
    # Convert the output tensor to a NumPy array
    output_array = output.detach().numpy().squeeze()
    
    # Extract the white part
    white_part = extract_white_part(output_array)
    
    # Visualize the DNA sequence
    visualize_dna_sequence(dna_sequence)
    
    # Display the original DNA sequence
    st.write("Original DNA Sequence:")
    st.write(dna_sequence)
    
    # Display the output image
    st.write("Output Image:")
    st.image(output_array, caption='Genomic Model Output', use_column_width=True)
    
    # Display the extracted white part
    st.write("Extracted White Part:")
    st.image(white_part, caption='White Part', use_column_width=True)

# Streamlit app
def main():
    st.set_page_config(page_title="Genomic Interpreter", page_icon="🧬", layout="wide")
    
    st.title("Genomic Interpreter")
    
    # Text area for entering DNA sequence
    dna_sequence = st.text_area("Enter DNA Sequence", "")
    
    # Analyze button
    if st.button("Analyze"):
        if not dna_sequence.strip():
            st.warning("Please enter a DNA sequence.")
        else:
            # Call the function to analyze DNA sequence
            analyze_dna_sequence(dna_sequence.strip())

if __name__ == "__main__":
    main()

'''

'''
import requests
import streamlit as st
import json
from streamlit_lottie import st_lottie
from demo import test_genomic_model
st.set_page_config(page_title="Genomic Interpreter", page_icon="🧬", layout="wide")

def load_lottieurl(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
#-----  LOAD ASSETS  -----
lottie_dna = load_lottieurl("D:/PS/GI code/1d-swin-main/images/dna2.json")



# Import backend functions from swin1d backend code
from swin1d.module import swin_1d_block
from demo import generate_random_dna, onehot_encoder, extract_white_part, visualize_dna_sequence, repeat_sequence

# Function to analyze DNA sequence using Swin1D model
def analyze_dna_sequence(dna_sequence):
    # Encode the input DNA sequence
    encoded_sequence = onehot_encoder(dna_sequence)
    
    # Generate Swin1D model
    model = swin1d_block(4)
    
    # Get the output from the model
    output = model(encoded_sequence)
    
    # Convert the output tensor to a NumPy array
    output_array = output.detach().numpy().squeeze()
    
    # Extract the white part
    white_part = extract_white_part(output_array)
    
    # Visualize the DNA sequence
    visualize_dna_sequence(dna_sequence)
    
    # Display the original DNA sequence
    st.write("Original DNA Sequence:")
    st.write(dna_sequence)
    
    # Display the output image
    st.write("Output Image:")
    st.image(output_array, caption='Genomic Model Output', use_column_width=True)
    
    # Display the extracted white part
    st.write("Extracted White Part:")
    st.image(white_part, caption='White Part', use_column_width=True)



#st.header("GENOMIC INTERPRETER")
st.title("GENOMIC INTERPRETER")

#-----  HEADER SECTION -----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2) 
    with left_column:
        dna_sequence = st.text_area("Enter DNA Sequence","     ")

        if st.button("Analyze"):
            if not dna_sequence:
                st.warning("Please enter a DNA sequence.")
            else:
            # Perform genomic interpretation
                result = analyze_dna_sequence(dna_sequence)

            # Display the results
            #st.success(f"Genomic Analysis Result:\n\nDNA Sequence Length: {result} base pairs")
        #st.header("What I do")
        #st.write("##")
    with right_column:
        st_lottie(lottie_dna,speed=2)

        ''' 

'''import requests
import streamlit as st
import json
from streamlit_lottie import st_lottie
from demo import test_genomic_model
st.set_page_config(page_title="Genomic Interpreter", page_icon="🧬", layout="wide")

def load_lottieurl(filepath:str):
    with open(filepath,"r") as f:
        return json.load(f)
#-----  LOAD ASSETS  -----
lottie_dna = load_lottieurl("D:/PS/GI code/1d-swin-main/images/dna2.json")
  


#st.header("GENOMIC INTERPRETER")
st.title("GENOMIC INTERPRETER")

#-----  HEADER SECTION -----
with st.container():
    st.write("---")
    left_column, right_column = st.columns(2) 
    with left_column:
        dna_sequence = st.text_area("Enter DNA Sequence","     ")

        if st.button("Analyze"):
            if len(sequence_input) != 512:
                st.error('Please enter exactly 512 characters.')
            else:
                # Call function to get image output from model
                image_output = test_genomic_model(sequence_input)

                # Convert image data to PIL Image
                img = Image.open(BytesIO(image_output))

                # Display image output
                st.image(img, caption='Model Output', use_column_width=True)


            # Display the results
            #st.success(f"Genomic Analysis Result:\n\nDNA Sequence Length: {result} base pairs")
        #st.header("What I do")
        #st.write("##")
    with right_column:
        st_lottie(lottie_dna,speed=2)
'''