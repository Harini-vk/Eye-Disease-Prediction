# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# # Load the trained model
# model = tf.keras.models.load_model('best_model.h5')

# # Define your class names (make sure these match your training data folders)
# # Replace these with your actual class names from the dataset folder
# class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
# # OR get them automatically if you still have the dataset loaded:
# # class_names = train_ds.class_names

# def predict_eye_disease(image_path):
#     """
#     Predicts eye disease from an image
    
#     Args:
#         image_path: Path to the image file
    
#     Returns:
#         prediction: The predicted class name
#         confidence: Confidence percentage
#     """
#     # Load and preprocess the image
#     img = Image.open(image_path)
#     img = img.resize((224, 224))  # Resize to model input size
#     img_array = np.array(img)
    
#     # Add batch dimension
#     img_array = np.expand_dims(img_array, axis=0)
    
#     # Make prediction
#     predictions = model.predict(img_array, verbose=0)
#     predicted_class_idx = np.argmax(predictions[0])
#     confidence = predictions[0][predicted_class_idx] * 100
#     predicted_class = class_names[predicted_class_idx]
    
#     return predicted_class, confidence, predictions[0]

# def predict_and_display(image_path):
#     """
#     Predict and display the result with the image
#     """
#     # Get prediction
#     disease, confidence, all_predictions = predict_eye_disease(image_path)
    
#     # Display results
#     print(f"\n{'='*50}")
#     print(f"Image: {image_path}")
#     print(f"{'='*50}")
#     print(f"Prediction: {disease.upper()}")
#     print(f"Confidence: {confidence:.2f}%")
#     print(f"\nAll class probabilities:")
#     for i, class_name in enumerate(class_names):
#         print(f"  {class_name}: {all_predictions[i]*100:.2f}%")
#     print(f"{'='*50}\n")
    
#     # Display image with prediction
#     img = Image.open(image_path)
#     plt.figure(figsize=(8, 6))
#     plt.imshow(img)
#     plt.axis('off')
    
#     # Color based on prediction
#     color = 'green' if disease.lower() == 'normal' else 'red'
#     plt.title(f"Prediction: {disease.upper()}\nConfidence: {confidence:.2f}%", 
#               fontsize=16, color=color, weight='bold')
#     plt.tight_layout()
#     plt.show()
    
#     return disease, confidence

# # Example usage:
# if __name__ == "__main__":
#     # Test with a single image
#     test_image = "test_image.png"  # Change this to your image path
#     predict_and_display(test_image)
    
#     # # OR test with multiple images
#     # test_images = [
#     #     "test_image.jpg",  # Change these to your image paths
#     # ]
    
#     for img_path in test_images:
#         predict_and_display(img_path)

import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('best_model.h5')

# Define class names
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

def predict_eye_disease(image_path):
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    predicted_class = class_names[predicted_class_idx]

    return predicted_class, confidence, predictions[0]

def predict_and_display(image_path):
    disease, confidence, all_predictions = predict_eye_disease(image_path)

    print(f"\n{'='*50}")
    print(f"Image: {image_path}")
    print(f"{'='*50}")
    print(f"Prediction: {disease.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nAll class probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {all_predictions[i]*100:.2f}%")
    print(f"{'='*50}\n")

    # Display image
    img = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')

    color = 'green' if disease.lower() == 'normal' else 'red'
    plt.title(f"Prediction: {disease.upper()}\nConfidence: {confidence:.2f}%",
              fontsize=16, color=color, weight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 🔹 Ask user for image path
    image_path = input("Enter the path of the eye image: ")

    # Run prediction
    predict_and_display(image_path)