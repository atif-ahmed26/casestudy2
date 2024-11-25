import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained generator model
@st.cache(allow_output_mutation=True)
def load_generator_model():
    return load_model('generator_model.h5')

# Function to generate a single random image
def generate_single_image(generator, latent_dim=100):
    # noise = np.random.normal(0, 1, (1, latent_dim))
    # generated_image = generator.predict(noise)


    label_indices = [1]
    labels = np.array(label_indices)
    noise = np.random.normal(0,1,(len(label_indices),latent_dim))
    generated_images = generator.predict([noise,labels])

    return generated_images

# Streamlit App
def main():
    st.title("DCGAN Model")
    st.write("Click the button to generate a single random image.")

    # Load generator model
    generator = load_generator_model()
    latent_dim = 100  # Adjust if your model uses a different latent dimension

    if st.button("Generate Image"):
        st.write("Generating a random image...")
        generated_image = generate_single_image(generator, latent_dim)
        st.image((generated_image + 1) / 2.0, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()
