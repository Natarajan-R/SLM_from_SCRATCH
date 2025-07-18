import torch
from train_slm import SmallLanguageModel, encode, decode, vocab_size  # Import from your training script

# --- Configuration ---
CHECKPOINT_PATH = 'checkpoints_v2/checkpoint_epoch_43.pth'  # IMPORTANT: Use the path to your BEST checkpoint
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_NEW_TOKENS = 300  # How many tokens to generate

# --- Model Loading ---
# Note: Ensure the model parameters in your_script.py match the ones used for training the checkpoint!
# (NUM_EMBED, NUM_HEAD, NUM_LAYER, etc.)
model = SmallLanguageModel()
model.to(DEVICE)

print(f"Loading model from {CHECKPOINT_PATH}...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully.")

# --- Generation ---
# The "prompt" is the starting text to kick off the generation
#prompt = "Arjuna said: " 
#prompt = "Q: What is the main lesson of the Bhagavad Gita? A:"
#prompt = "You have a right to perform your prescribed duties, but"


#prompt = "Therefore, O Prince, the wise man acts without thought of reward, seeing"

#prompt = "The Self, which is unborn and eternal,"

#prompt = "He who is free from the grip of desire"

#prompt = "Dharma"

prompt = "karma yog"

# Encode the prompt and prepare it for the model
context = torch.tensor(encode(prompt), dtype=torch.long, device=DEVICE).unsqueeze(0)

print(f"\n--- Starting generation from prompt: '{prompt}' ---")
generated_indices = model.generate(context, max_new_tokens=MAX_NEW_TOKENS)[0].tolist()
generated_text = decode(generated_indices)

print("\n--- Generated Text ---")
print(generated_text)