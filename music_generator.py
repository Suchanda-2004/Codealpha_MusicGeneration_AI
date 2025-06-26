import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import pretty_midi # Added pretty_midi import

# --- Configuration ---
# Make sure your MIDI files are in a subfolder named 'midi_songs'
# within your project directory (e.g., C:\...\Musicals\midi_songs)
DATASET_PATH = 'midi_songs'
SEQUENCE_LENGTH = 100 # How many notes to consider for predicting the next note
EPOCHS = 50          # Number of training epochs (can be increased for better results)
BATCH_SIZE = 64      # Batch size for training

# --- Data Loading and Preprocessing ---
def get_notes():
    """Extracts notes and chords from MIDI files."""
    notes = []
    # Create the directory if it doesn't exist
    if not os.path.exists(DATASET_PATH):
        print(f"Error: The '{DATASET_PATH}' directory does not exist.")
        print(f"Please create it inside your project folder: {os.getcwd()}\\{DATASET_PATH}")
        print("And place your MIDI files there.")
        return [], 0

    midi_files = glob.glob(os.path.join(DATASET_PATH, "*.mid"))
    if not midi_files:
        print(f"No MIDI files found in '{DATASET_PATH}'. Please place some .mid files there.")
        return [], 0

    for file in midi_files:
        try:
            print(f"Parsing {file}")
            midi = converter.parse(file)
            notes_to_parse = None

            try: # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                if s2.parts: # Check if there are parts
                    notes_to_parse = s2.parts[0].recurse()
                else: # No parts, try flat notes
                    notes_to_parse = midi.flat.notes
            except Exception: # Fallback to flat notes if partitioning fails
                notes_to_parse = midi.flat.notes

            if notes_to_parse:
                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        # Store pitch (e.g., 'C4', 'D#5')
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        # Store normal form of chord pitches (e.g., '0.4.7' for C major)
                        notes.append('.'.join(str(n) for n in element.normalOrder))
            else:
                print(f"No parseable notes found in {file}")

        except Exception as e:
            print(f"Could not parse {file}: {e}")
            continue

    if not notes:
        print("No notes extracted from any MIDI files. Please ensure your MIDI files are valid and contain playable notes.")
        return [], 0

    # Save the extracted notes for faster loading next time
    # Ensure 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)
    print(f"Extracted {len(notes)} notes/chords.")
    return notes, len(set(notes))

def prepare_sequences(notes, n_vocab):
    """
    Prepare the input and output sequences for the neural network.
    """
    # Get all unique note names/chord names
    pitchnames = sorted(list(set(notes)))

    # Create a dictionary to map notes to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - SEQUENCE_LENGTH, 1):
        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    # Input shape: (number_of_sequences, sequence_length, number_of_features_per_step)
    # Here, features_per_step is 1 (the integer ID of the note)
    network_input = np.reshape(network_input, (n_patterns, SEQUENCE_LENGTH, 1))

    # Normalize input to be between 0 and 1
    # This is crucial for neural networks, especially when using Activation functions like tanh/sigmoid implicitly
    network_input = network_input / float(n_vocab)

    # Convert output to one-hot encoding (for categorical crossentropy loss)
    network_output = to_categorical(network_output, num_classes=n_vocab)

    print(f"Prepared {n_patterns} sequences for training.")
    print(f"Input shape: {network_input.shape}")
    print(f"Output shape: {network_output.shape}")

    return (network_input, network_output, pitchnames, note_to_int)

# --- Model Definition ---
def create_model(input_shape, n_vocab):
    """Create the Keras model for music generation."""
    model = Sequential()
    # First LSTM layer: returns sequences to feed to the next LSTM layer
    model.add(LSTM(
        512, # Number of LSTM units (neurons)
        input_shape=(input_shape[1], input_shape[2]), # (sequence_length, features_per_step)
        return_sequences=True, # Important for stacked LSTMs
        recurrent_dropout=0.3 # Dropout for the recurrent connections
    ))
    model.add(Dropout(0.3)) # Dropout for the output of the LSTM layer

    # Second LSTM layer
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(Dropout(0.3))

    # Third LSTM layer: does NOT return sequences, as it's the last LSTM before Dense layers
    model.add(LSTM(512))
    model.add(Dense(256)) # A dense layer before the output layer
    model.add(Dropout(0.3))

    # Output layer: n_vocab neurons, with softmax activation for probability distribution
    # over all possible next notes/chords
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print("Model created and compiled successfully.")
    model.summary() # Print model summary to console
    return model

# --- Training the Model ---
def train_model():
    """Train the neural network to generate music."""
    notes, n_vocab = get_notes()

    if not notes:
        print("Training cannot proceed without extracted notes. Please check MIDI files.")
        return

    network_input, network_output, _, _ = prepare_sequences(notes, n_vocab)

    model = create_model(network_input.shape, n_vocab)

    # Setup ModelCheckpoint to save the best weights during training
    # Saves only when the validation loss improves (or just training loss if no validation split)
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras" # Saved in .keras format (TensorFlow 2.x)
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss', # Monitor training loss
        verbose=1,      # Log checkpoint messages
        save_best_only=True, # Save only the best model found so far
        mode='min'      # 'min' means we want to minimize the loss
    )
    callbacks_list = [checkpoint]

    print(f"Starting training for {EPOCHS} epochs with batch size {BATCH_SIZE}...")
    model.fit(network_input, network_output,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=callbacks_list,
              verbose=1) # Display progress bar during training

    print("Model training complete.")

# --- Music Generation ---
def generate_music(model_weights_path):
    """Generate music using the trained neural network."""
    # Load notes and vocabulary from the saved pickle file
    try:
        with open('data/notes', 'rb') as filepath:
            notes = pickle.load(filepath)
    except FileNotFoundError:
        print("Error: 'data/notes' file not found. Please ensure the training step has completed successfully.")
        return

    # Get all unique note names/chord names
    pitchnames = sorted(list(set(notes)))
    n_vocab = len(pitchnames)

    # Create mappings between integers and notes/chords
    int_to_note = dict((number, note_val) for number, note_val in enumerate(pitchnames))
    note_to_int = dict((note_val, number) for number, note_val in enumerate(pitchnames))

    # Prepare input sequences (only need to get the shape and a starting pattern)
    network_input, _, _, _ = prepare_sequences(notes, n_vocab)

    # Create the model architecture (it needs to be the same as during training)
    model = create_model(network_input.shape, n_vocab)

    # Load the trained weights
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found at '{model_weights_path}'.")
        print("Please ensure you've trained the model and the path is correct.")
        return
    model.load_weights(model_weights_path)
    print(f"Model weights loaded from: {model_weights_path}")

    # Pick a random sequence from the input as a starting point for the prediction
    # This gives the generation process a context
    start = np.random.randint(0, len(network_input) - 1)
    # The 'pattern' should contain the integer IDs, not normalized floats, for array indexing
    # FIX for DeprecationWarning: Ensure x is treated as a scalar for int() conversion
    pattern = [int(item[0] * n_vocab) for item in network_input[start]]
    prediction_output = []

    # Generate 500 notes/chords
    NUM_NOTES_TO_GENERATE = 500
    print(f"Generating {NUM_NOTES_TO_GENERATE} notes/chords...")
    for note_index in range(NUM_NOTES_TO_GENERATE):
        # Prepare the input for prediction: reshape and normalize
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        # Get prediction probabilities
        prediction = model.predict(prediction_input, verbose=0)[0] # Get the first (and only) prediction

        # Sample the next note/chord based on probabilities
        # Using argmax for a more deterministic output (most probable)
        index = np.argmax(prediction)
        # For more creative/less deterministic output, you could use np.random.choice
        # index = np.random.choice(len(prediction), p=prediction)

        # Convert the predicted integer index back to a note/chord string
        result = int_to_note[index]
        prediction_output.append(result)

        # Add the predicted note's integer ID to the pattern and remove the first element
        # This creates a sliding window for prediction
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print("\nGenerated music sequence (first 20 elements):")
    print(prediction_output[:20])

    # Convert the generated notes back to a MIDI file
    offset = 0.0 # Time offset for placing notes
    output_notes = []

    # Create music21 stream elements from the generated sequence
    for pattern_element in prediction_output:
        # If it's a chord (contains '.', or is just digits from normalOrder)
        if ('.' in pattern_element) or pattern_element.isdigit():
            notes_in_chord = pattern_element.split('.')
            notes_obj_list = []
            for current_note_str in notes_in_chord:
                try:
                    # Convert to integer, then to a music21 Note
                    new_note = note.Note(int(current_note_str))
                    new_note.storedInstrument = instrument.Piano()
                    notes_obj_list.append(new_note)
                except Exception as e:
                    print(f"Warning: Could not parse note in chord '{current_note_str}': {e}")
                    continue
            if notes_obj_list:
                new_chord = chord.Chord(notes_obj_list)
                new_chord.offset = offset
                output_notes.append(new_chord)
        # If it's a single note
        else:
            try:
                new_note = note.Note(pattern_element)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            except Exception as e:
                print(f"Warning: Could not parse single note '{pattern_element}': {e}")

        # Advance the offset for the next note/chord.
        # This is a fixed duration. For more complex rhythms, you might
        # need to predict durations as well, or infer them.
        offset += 0.5 # Each note/chord is roughly a half-note apart

    # Convert the music21 stream elements to a pretty_midi object for robust MIDI writing
    midi_data = pretty_midi.PrettyMIDI()
    # Choose a common instrument program for the output
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano_instrument = pretty_midi.Instrument(program=piano_program)
    midi_data.instruments.append(piano_instrument)

    for element in output_notes:
        start_time = float(element.offset)
        # Default duration for generated notes/chords
        end_time = start_time + 0.5 # Each note/chord lasts 0.5 units of time

        if isinstance(element, note.Note):
            # Use .pitch.midi for robustness.
            pitch = element.pitch.midi
            # velocity (how loud) typically 0-127
            new_pm_note = pretty_midi.Note(velocity=64, pitch=pitch, start=start_time, end=end_time)
            piano_instrument.notes.append(new_pm_note)
        elif isinstance(element, chord.Chord):
            for n_in_chord in element.notes:
                # Use .pitch.midi for robustness.
                pitch = n_in_chord.pitch.midi
                new_pm_note = pretty_midi.Note(velocity=64, pitch=pitch, start=start_time, end=end_time)
                piano_instrument.notes.append(new_pm_note)
        else:
            print(f"Warning: Unknown element type encountered during pretty_midi conversion: {type(element)}")
            continue # Skip elements that are not notes or chords

    output_midi_file = 'generated_music.mid'
    midi_data.write(output_midi_file)
    print(f"Generated MIDI file: {output_midi_file}")

# --- Main execution block ---
if __name__ == "__main__":
    # Ensure 'data' directory exists for storing notes and other temporary files
    if not os.path.exists('data'):
        os.makedirs('data')

    # --- Step 1: Train the model ---
    # Uncomment the line below to train the model.
    # This will take a significant amount of time depending on your dataset and EPOCHS.
    # It will save model weights as 'weights-improvement-XX-YY.ZZZZ-bigger.keras' files.
    # train_model() # This line should be uncommented to run training

    # --- Step 2: Generate music using the trained model ---
    # AFTER training is complete, you need to provide the path to the best saved weights.
    # The script tries to find the most recently modified weights file.
    all_weights_files = glob.glob("weights-improvement-*-bigger.keras")

    if not all_weights_files:
        print("\nNo model weights found. Please ensure 'train_model()' ran successfully.")
        print("After training, re-run the script. It will then automatically find the best weights.")
        print("If you haven't trained yet, uncomment 'train_model()' line above.")
    else:
        # Sort files by modification time (most recent first) and pick the first one
        best_weights_file = max(all_weights_files, key=os.path.getmtime)
        print(f"\nAttempting to generate music using weights from: {best_weights_file}")
        generate_music(best_weights_file)
