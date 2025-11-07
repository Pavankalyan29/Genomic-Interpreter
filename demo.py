from swin1d.module import swin_1d_block
from swin1d.examples import (
    random_text_generator,
    generate_random_dna,
    onehot_encoder,
)
"""torch.manual_seed(42)
random.seed(42)
np.random.seed(42)"""

def test_genomic_model(seq_length=512):
    input = generate_random_dna(seq_length)
    encode_input = onehot_encoder(input)                    
    model = swin1d_block(4)
    output = model(encode_input)
    print(output.shape)
    return output


def test_language_model(seq_length=512):
    input = random_text_generator(2, seq_length, tokenized=True)
    model = swin1d_block(1)
    output = model(input)
    print(output.shape)
    return output


def swin1d_block(dim):
    # stage = (number layers in each swin,
    #          whether to merge the ouput of each swin,
    #          window size)

    window_size = 32
    stages = [
        (
            4,
            True,
            window_size,
        ),
        (
            2,
            False,
            window_size,
        ),
        (
            2,
            False,
            window_size,
        ),
        (
            2,
            False,
            window_size,
        ),
    ]
    model = swin_1d_block(stages, dim)
    return model


if __name__ == "__main__":
    test_genomic_model()



# #Output-Genomic Model output image,original image,white extraction image
from swin1d.module import swin_1d_block
from swin1d.examples import (
    random_text_generator,
    generate_random_dna,
    onehot_encoder,
)

import torch
import matplotlib.pyplot as plt
import numpy as np

'''takes a DNA sequence as input and visualizes it using a color mapping. The DNA bases ('A', 'T', 'C', 'G') are mapped
to numeric values, and the sequence is displayed as an image using matplotlib.pyplot.'''

def visualize_dna_sequence(sequence):
    if isinstance(sequence, list):
        # Concatenate the list of strings into a single string
        sequence = ''.join(sequence)
    mapping = {'A': 0.8, 'T': 0.5, 'C': 1, 'G': 1.5}
    colors = np.array([mapping[base] for base in sequence])

    plt.imshow(colors.reshape(1, -1), cmap='coolwarm', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.title("DNA Sequence")
    plt.show()
    
    
''' takes an image array and a threshold as input. It performs thresholding 
on the input image array to extract the white part, returning a binary array where white pixels are represented by 1.0.'''
def extract_white_part(image_array, threshold=0.5):
    # Thresholding to extract white part
    white_part = (image_array > threshold).astype(float)
    return white_part




def repeat_sequence(original_sequence, desired_length):
    # Repeat the sequence to achieve the desired length
    repeated_sequence = original_sequence * (desired_length // len(original_sequence)) + original_sequence[:desired_length % len(original_sequence)]
    return repeated_sequence

 # Example usage:
original_sequence = "TGCATTTTTTTCACATCGGAATGGTGTTGTCGACCTTGCCTTCATAGTTACTTGATGTTAATGATTGGACATGTTCATCAAACCGGGTCACAGAGGTTACGGCTGTT"
desired_length = 512
result_sequence = repeat_sequence(original_sequence, desired_length)
print(result_sequence)






'''generates a DNA sequence, prints it, encodes it using a one-hot encoder, and passes it through a Swin Transformer 
model (swin1d_block). The output shape of the model is printed.'''


def test_genomic_model(seq_length=512):
    # input_sequence = generate_random_dna(seq_length)
    input_sequence=['CAAATGCCTAAACCCGATCTGAATCGTGTTTTACTGTTATCACGCGTGAAAACTGTCTAGCGCAGTGGGATCTTATGCAAGTTATAGGTCCCATTCTGGCGCGCCTCTTGCTGTGCAACTTGCGTGAGGAGGGGTCTTTTAACCTCTTAACACTTACTAGAGACAAAAACTGAACGTACTCAGGGTTCTTCCCGAGGTTTATCTTCTGCGTTAGCAAACCTGAGTCTGCGTTGACCCTCGATTTTTAAGCCGTATAGAAGACGGTGTAGTGGGTGGTTCGTCTTTGCTGAAACGAGACCGCGTAGTACAGGGGCTGTATGACTGGGGACCTCTGAAAATCCAATACTGAGTAGAAACAAGCACTCCTGCTCCCACTACGTTCAACCACCTAATCGTGATCGAGATAAAAGATTATGGGCCACCGATAAGTCGATTTTCTGACATTGTGTATACGTGCAGTAGGATTATATTCTGGCATGGAAAAACCTGTCTTTAGGGTGCAAGGTATAACA']
    # input_sequence = repeat_sequence('TGCATTTTTTTCACATCGGAATGGTGTTGTCGACCTTGCCTTCATAGTTACTTGATGTTAATGATTGGACATGTTCATCAAACCGGGTCACAGAGGTTACGGCTGTT', 512)
    print("Generated DNA Sequence:", input_sequence)  # Print the generated sequence
    encode_input = onehot_encoder(input_sequence)
    model = swin1d_block(4)
    output = model(encode_input)
    print(output.shape)

    '''Converts the model output tensor to a NumPy array, visualizes the genomic model output, and displays the 
    original image and the extracted white part using matplotlib.pyplot.'''
    # Convert torch tensor to NumPy array
    output_array = output.detach().numpy().squeeze()

    # Display the genomic model output
    plt.subplot(1, 3, 1)
    plt.imshow(output_array, cmap='gray')
    plt.title("Genomic Model Output")

    # Extract the white part
    white_part = extract_white_part(output_array)

    # Display the original image
    plt.subplot(1, 3, 2)
    plt.imshow(output_array, cmap='gray')
    plt.title("Original Image")

    # Display the extracted white part
    plt.subplot(1, 3, 3)
    plt.imshow(white_part, cmap='gray')
    plt.title("White Part")

    plt.show()

    # Visualize the DNA sequence
    visualize_dna_sequence(input_sequence)

'''creates a Swin Transformer model with specified parameters, such as the dimension and window size.'''
def swin1d_block(dim):
    window_size = 32
    stages = [
        (4, True, window_size),
        (2, False, window_size),
        (2, False, window_size),
        (2, False, window_size),
    ]
    model = swin_1d_block(stages, dim)
    return model

if __name__ == "__main__":
    # Generate the genomic model output and visualize
    test_genomic_model()

    
