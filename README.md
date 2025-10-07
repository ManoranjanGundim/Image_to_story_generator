# Image to Story Generator

## Project Overview

The **Image to Story Generator** is a web application that transforms an image into a compelling short story. This project uses a sophisticated AI pipeline to first generate a descriptive caption for an uploaded image and then expands that caption into a creative narrative[cite: 13, 14].

This tool bridges the gap between computer vision and natural language generation, offering an intuitive way to turn visual data into engaging stories[cite: 24]. It is deployed as a user-friendly web application using Flask and Vercel, making AI-powered storytelling accessible to everyone[cite: 17, 34].

***

##  Features

* **Automated Image Captioning**: Generates a detailed and contextually relevant caption for any uploaded image[cite: 15].
* **Creative Story Generation**: Uses the generated caption as a prompt to craft a unique and coherent short story with a narrative hook[cite: 16, 257].
* **Intuitive Web Interface**: A clean, simple frontend allows users to easily upload an image and receive the generated content in real-time[cite: 38].
* **Efficient AI Pipeline**: Employs a two-stage deep learning architecture for robust performance, combining CNN-RNN models with a Transformer-based language model[cite: 14, 30].
* **Serverless Deployment**: Hosted on Vercel for scalable, on-demand processing that is both efficient and cost-effective.

***

## How It Works: The Technical Pipeline

The application's core is a two-stage process that seamlessly converts pixels to prose.

### Stage 1: Image Captioning Model

This model generates a concise, descriptive sentence about the uploaded image[cite: 52].

* **Feature Extraction (InceptionV3)**: A pre-trained **InceptionV3** model, a powerful Convolutional Neural Network (CNN), processes the input image to extract high-level visual features. All images are first resized to **299x299** pixels and normalized to a range of **-1 to 1** to be compatible with the model. The output from the last convolutional layer is an `8x8x2048` feature map[cite: 88].
* **CNN Encoder**: The extracted features are passed through a fully connected layer (Dense layer) which condenses the information into a more compact 256-dimensional vector.
* **RNN Decoder**: An RNN, specifically a **GRU** (Gated Recurrent Unit), takes the encoded feature vector and generates the caption word by word. It uses a **Bahdanau Attention mechanism**, which allows the decoder to dynamically focus on the most relevant parts of the image while generating each word, significantly improving contextual accuracy.



### Stage 2: Story Generation Model

The generated caption becomes the seed for a creative narrative, which is produced by a large language model[cite: 232].

* **Text Preprocessing**: The caption is first cleaned, converted to the past tense, and combined with a random **narrative hook** (e.g., "A shrill cry echoed in the mist...") to create a more compelling prompt.
* **Language Model (GPT-2)**: The enhanced prompt is fed into a **GPT-2 model**, a state-of-the-art transformer-based model developed by OpenAI. GPT-2 expands the prompt into a full story, ensuring narrative coherence and creative flair by predicting the most probable sequence of words[cite: 265].
* **Text Post-processing**: The final story is cleaned to remove any incomplete sentences, ensuring the output is polished and readable.



***

## Project Structure

The project is organized into a clean directory structure, optimized for Vercel deployment.


image-to-story-app/
├── api/
│   └── index.py            # Flask backend for captioning and story generation
├── models/
│   ├── encoder_weights.h5  # Trained weights for the CNN Encoder
│   ├── decoder_weights.h5  # Trained weights for the RNN Decoder
│   └── tokenizer.pickle    # Vocabulary tokenizer for image captioning
├── static/
│   ├── style.css           # Styling for the web interface
│   └── script.js           # Frontend logic for image upload and API calls
├── templates/
│   └── index.html          # Main HTML page for the web interface
├── requirements.txt        # Python dependencies for the backend
└── vercel.json             # Vercel deployment configuration


