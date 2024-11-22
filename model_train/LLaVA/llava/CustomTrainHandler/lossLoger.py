from transformers import TrainerCallback
import os

class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_loss.txt")
    
    def on_log(self, args, state, control, **kwargs):
        with open(self.log_file, "a") as f:
            f.write(f"Step: {state.global_step}, Loss: {state.log_history[-1]}\n")