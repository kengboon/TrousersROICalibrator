class EarlyStopping:
    def __init__(self, patience, min_delta, mode="min"):
        self.patience = patience
        self.min_delta = abs(min_delta)
        self.mode = mode
        self.best_score = None
        self.stag_cnt = 0
        self.is_stagnant = False

    def check(self, last_score):
        if self.best_score is None:
            self.best_score = last_score
            self.is_stagnant = False
        elif (self.mode == "min" and self.best_score - last_score > self.min_delta) or \
            (self.mode == "max" and last_score - self.best_score > self.min_delta):
            self.best_score = last_score
            self.stag_cnt = 0
            self.is_stagnant = False
        else:
            self.stag_cnt += 1
            self.is_stagnant = True
            if self.stag_cnt >= self.patience:
                return True

    def reset(self):
        self.best_score = None
        self.stag_cnt = 0
        self.is_stagnant = False

    def state_dict(self):
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_score": self.best_score,
            "stag_cnt": self.stag_cnt,
            "is_stagnant": self.is_stagnant,
        }

    def load_state_dict(self, state_dict: dict):
        [setattr(self, k, v) for k, v in state_dict]