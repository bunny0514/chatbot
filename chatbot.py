import tensorflow as tf
import numpy as np

# Define dummy training data (questions and answers)
questions = ['What is your name?', 'How are you?', 'Who are you?', 'Exit']
answers = ['My name is Chatbot.', 'I am fine, thank you.', 'I am a chatbot.', 'Goodbye!']

# Tokenizing the data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(questions + answers)

# Convert text to sequences
questions_sequence = tokenizer.texts_to_sequences(questions)
answers_sequence = tokenizer.texts_to_sequences(answers)

# Padding sequences for fixed length
max_len = max(len(sequence) for sequence in questions_sequence + answers_sequence)
questions_padded = tf.keras.preprocessing.sequence.pad_sequences(questions_sequence, maxlen=max_len, padding='post')
answers_padded = tf.keras.preprocessing.sequence.pad_sequences(answers_sequence, maxlen=max_len, padding='post')

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 128, input_length=max_len),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(tokenizer.word_index)+1, activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(questions_padded, np.expand_dims(answers_padded, axis=-1), epochs=100)

# Function to generate responses
def generate_response(input_text):
    input_sequence = tokenizer.texts_to_sequences([input_text])
    input_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_len, padding='post')
    predicted = model.predict(input_sequence_padded)[0]
    predicted_word_index = np.argmax(predicted, axis=-1)
    return list(tokenizer.word_index.keys())[predicted_word_index[0]]

# Chat loop
print("Chatbot: Hi! How can I help you? (You can type 'Exit' to end the conversation)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break
    response = generate_response(user_input)
    print(f"Chatbot: {response}")
