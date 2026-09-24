import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# =====================================================================
# STEP 1: DESIGNING THE DIGITAL BRAIN ARCHITECTURE (THE BLUEPRINT)
# =====================================================================
class SimpleFitnessCNN(nn.Module):
    def __init__(self):
        # We must call 'super' to inherit PyTorch's hidden tracking powers.
        # It sets up internal wiring like memory grids and tracking logs automatically.
        super(SimpleFitnessCNN, self).__init__()

        # 'nn.Sequential' acts like a conveyor belt assembly line.
        # It automatically pushes data from one machine directly into the next.
        self.assembly_line = nn.Sequential(
            # --- STATION 1: THE FIRST CONVOLUTIONAL LAYER ---
            # Expects 1 input channel (Grayscale camera feed).
            # Hires 8 different "magnifying glasses" (out_channels=8) to scan the image.
            # Each magnifier is a 3x3 pixel square (kernel_size=3).
            # 'padding=1' adds a border of fake empty pixels around the frame so our
            # magnifiers can inspect the outer edges without shrinking the image grid!
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            # The Reality Filter (ReLU): Wipes out all negative match scores to 0.
            # It only allows positive, useful structural patterns to pass forward.
            nn.ReLU(),
            # The Compactor (MaxPool): Zooms out by looking at 2x2 blocks of pixels,
            # keeping only the strongest feature score and discarding the rest.
            # This cuts the height and width of the data perfectly in half!
            nn.MaxPool2d(kernel_size=2),
            # --- STATION 2: THE SECOND CONVOLUTIONAL LAYER ---
            # It takes the 8 pattern maps made by Station 1 and uses 16 more advanced
            # magnifiers to stitch those simple lines into whole shapes (like bent knees or arms).
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # --- STATION 3: THE PACKING Department ---
            # We don't need a 2D image grid anymore now that shapes are detected.
            # This flattens the remaining data blocks into a single flat strip of numbers.
            nn.Flatten(),
        )

        # --- STATION 4: THE JUDGING PANEL ---
        # Takes the final flat list of shapes and turns them into a final prediction decision.
        # Since our 28x28 image grid was cut in half twice by MaxPool layers, it shrank to 7x7.
        # 16 final feature maps * 7x7 pixels = 784 structural input shapes.
        # It maps these 784 shapes to 2 final choice buckets: [Squat, Pushup].
        self.judging_panel = nn.Linear(in_features=16 * 7 * 7, out_features=2)

    def forward(self, camera_frame):
        # Push the raw camera image down the assembly line to get the pattern maps
        detected_shapes = self.assembly_line(camera_frame)

        # Pass the shapes to the judges to get final prediction scores
        final_scores = self.judging_panel(detected_shapes)
        return final_scores


# =====================================================================
# STEP 2: SETTING UP THE CLASSROOM (PREPARING DATA & THE TEACHER)
# =====================================================================
if __name__ == "__main__":
    print("Initializing Digital Fitness Academy...")

    # --- 1. PICKING OUR COMPUTE ENVIRONMENT ---
    # Check if a powerful graphics hardware card (CUDA) is connected.
    # If not, the model will fall back to using the standard computer central brain (CPU).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computer core: {device}")

    # --- 2. GENERATING MOCK HOMEWORK & EXAMS ---
    # Create 100 random training images (1 channel, 28x28 size pixels)
    mock_train_images = torch.randn(100, 1, 28, 28)
    # Correct Answer Key: 0 means Squat, 1 means Pushup
    mock_train_labels = torch.randint(0, 2, (100,))

    # Create 20 brand-new exam images the model will never study during training
    mock_eval_images = torch.randn(20, 1, 28, 28)
    mock_eval_labels = torch.randint(0, 2, (20,))

    # Wrap datasets into 'DataLoaders' to hand out data cards in manageable batches of 16.
    # This stops the computer's memory from overloading. Shuffling training cards keeps it fair!
    train_loader = DataLoader(
        TensorDataset(mock_train_images, mock_train_labels), batch_size=16, shuffle=True
    )
    eval_loader = DataLoader(
        TensorDataset(mock_eval_images, mock_eval_labels), batch_size=16, shuffle=False
    )

    # --- 3. CREATING THE SYSTEM PIECES ---
    coach = SimpleFitnessCNN().to(
        device
    )  # Ship our digital brain structure to the hardware
    scoring_rubric = nn.CrossEntropyLoss()  # Measures how horribly wrong a guess was

    # The Optimizer (tweak_mechanism): Scientific micro-adjuster tool.
    # It tweaks the 9 microscopic numbers inside our 3x3 magnifiers to correct mistakes.
    # 'lr=0.001' (Learning Rate) restricts it to follow only 0.1% of the error advice
    # per step. This ensures gradual, stable learning without overcorrecting!
    tweak_mechanism = optim.Adam(coach.parameters(), lr=0.001)

    # =====================================================================
    # STEP 3: THE TRAINING PHASE (STUDY SESSION)
    # =====================================================================
    print("\n--- STARTING THE STUDY SESSION (TRAINING) ---")
    coach.train()  # Flip the switch to "Learning Mode" so magnifier settings can be changed

    for epoch in range(
        3
    ):  # Read through the entire stack of flashcards 3 times (epochs)
        total_batch_error = 0.0

        for batch_images, batch_labels in train_loader:
            # Send our batch data over to the matching compute hardware (CPU or GPU)
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # 1. Clear out the blackboard memory of the last batch's mistakes
            tweak_mechanism.zero_grad()

            # 2. Forward Pass: Digital coach reads the 16 images and writes down raw guesses
            guesses = coach(batch_images)

            # 3. Calculate Loss: Check the grading rubric to see how wrong the guesses were
            loss = scoring_rubric(guesses, batch_labels)

            # 4. Backward Pass: Trace the error path backwards down the conveyor belt to
            # figure out exactly which magnifier settings caused the bad judgment call
            loss.backward()

            # 5. Tweak Settings: Turn the fine-tuning knobs on the filters to correct the weights
            tweak_mechanism.step()

            total_batch_error += loss.item()

        print(
            f"Epoch {epoch + 1} Average Penalty Score (Loss): {total_batch_error / len(train_loader):.4f}"
        )

    # =====================================================================
    # STEP 4: THE EVALUATION PHASE (THE FINAL EXAM)
    # =====================================================================
    print("\n--- STARTING THE FINAL EXAM (EVALUATION) ---")
    coach.eval()  # Flip switch to "Exam Mode" (Locks the magnifier layers completely!)

    correct_answers = 0
    total_questions = 0

    # 'torch.no_grad()' orders the computer to stop drawing hidden computation maps
    # and tracking mistakes since we are no longer tweaking settings. Saves massive compute speed!
    with torch.no_grad():
        for batch_images, batch_labels in eval_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            # Model takes the test
            guesses = coach(batch_images)

            # The model outputs raw point scores, like [Squat: 5.4, Pushup: -1.2]
            # 'torch.max' looks at the row and extracts the index of the highest score (Bucket 0 or 1)
            _, final_choice = torch.max(guesses, dim=1)

            # Tally up the correct answers
            correct_answers += (final_choice == batch_labels).sum().item()
            total_questions += batch_labels.size(0)

    accuracy = (correct_answers / total_questions) * 100
    print(f"Coach's Evaluation Score: {accuracy:.1f}% Accuracy on New Images!")
