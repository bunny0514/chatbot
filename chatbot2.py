from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import spacy
nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()
doc = nlp("This is a sentence.")
print([(w.text, w.pos_) for w in doc])

# Create a new instance of a ChatBot
chatbot = ChatBot('MyBot')

# Create a new trainer for the chatbot
trainer = ListTrainer(chatbot)

# Provide some training data
training_data = [
    'Hello!',
    'Hi there!',
    'How are you?',
    'I am doing well, thank you.',
    # Add more pairs of input and responses as needed
]

# Train the chatbot with the custom data
trainer.train(training_data)

# Get a response from the chatbot
response = chatbot.get_response('Hi')
print(response)
