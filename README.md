# Experimentation Process for Traffic Sign Recognition

In this project, I explored how neural networks can classify traffic signs using the GTSRB dataset. The journey was both challenging and rewarding, filled with moments of confusion, discovery, and finally, a sense of accomplishment when everything worked. Here’s a look into what I tried, what worked, and what didn’t.

### Getting Started
At first, the task of recognizing 43 different types of traffic signs felt a bit overwhelming. I started by focusing on loading the data correctly. Using OpenCV, I resized all images to 30x30 pixels so they’d have a uniform shape for the neural network. After successfully loading 26,640 images, I set up a basic sequential model using TensorFlow.

For the initial model, I went with two convolutional layers followed by max pooling and a dense output layer. I didn’t know what to expect at first, so I kept things simple, ran the model for 10 epochs, and crossed my fingers. It worked—but not great. Training accuracy was okay, but test accuracy was much lower, hinting at overfitting.

---

### What I Tried and Learned
1. **Tuning the Architecture**:
   - I experimented with the number of convolutional and pooling layers to find the right balance between complexity and efficiency. Adding more layers helped the model recognize patterns better but increased training time significantly.
   - I also tried different filter sizes for the convolutional layers. Larger filters seemed to help capture broader patterns, but beyond a certain point, the improvement was minimal.

2. **Pooling Layer Adjustments**:
   - I played around with the pool sizes in the max pooling layers. Smaller pool sizes preserved more details in the feature maps, while larger ones sped up computation at the cost of some accuracy. A size of `(2, 2)` worked best.

3. **Hidden Layers**:
   - For the fully connected (dense) layers, I tested different numbers and sizes of neurons. Adding too many neurons resulted in overfitting, while too few limited the model’s ability to learn. A dense layer with 128 neurons provided a good balance.

4. **Dropout**:
   - Adding dropout layers helped prevent overfitting, especially in the dense layers. By randomly "dropping out" neurons during training, the model became more robust and generalized better to unseen data.

---

### Final Results
After all the tweaks, my final model had:
- 3 convolutional layers (with 32, 64, and 128 filters) and max pooling after each.
- A dense hidden layer with 128 neurons.
- A softmax output layer for classifying the 43 traffic sign categories.

With this setup, the model achieved **97.66% accuracy** on the training set and **96.15% accuracy** on the test set. Seeing the model get so many predictions right was definitely a highlight of this project!

---

### Reflections
Working on this project helped me better understand how neural networks learn and improve. I also learned that overfitting can sneak up on you and that small tweaks (like dropout or more layers) can make a big difference. There’s still room to explore—like better handling of similar-looking signs—but I’m happy with how far I’ve come.