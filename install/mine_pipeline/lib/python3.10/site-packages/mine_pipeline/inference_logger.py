import csv
from datetime import datetime

class InferenceLogger:
    def __init__(self, path='/tmp/infer_log.csv'):
        self.log_file = open(path, mode='w', newline='')
        self.logger = csv.writer(self.log_file)
        self.logger.writerow(['timestamp', 'score', 'alarm'])

    def log(self, score: float):
        now = datetime.utcnow().isoformat()
        alarm = score > 0.5
        self.logger.writerow([now, score, alarm])
        self.log_file.flush()