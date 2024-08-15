from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load the YOLO model
model = YOLO('./best.pt')

# Make a prediction on the image
results = model.predict(source='/home/quoccu/Documents/MLOps/Yolov8_PalletDection/test/images/frame_0068_jpg.rf.77c54c4986230f89581505c51f355c2c.jpg')

# Get the results and visualize
result_image = results[0].plot()  # Plot the first (and only) result

# Show the result image using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Turn off axis
plt.show()
