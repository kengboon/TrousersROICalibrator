from .fashions import DeepFashion2Dataset

# Collate function is REQUIRED for detection datasets
def collate_fn(batch):
    return tuple(zip(*batch))