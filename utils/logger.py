import os
from datetime import datetime


class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, "log.txt")
        self.f = open(self.log_path, "a")

    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {msg}"
        print(msg)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def init_wandb(args, log_dir="logs/"):
    import wandb
    os.makedirs(log_dir, exist_ok=True)
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
        dir=log_dir,
    )


def log_wandb(metrics, step):
    import wandb
    metrics["epoch"] = step
    wandb.log(metrics)
