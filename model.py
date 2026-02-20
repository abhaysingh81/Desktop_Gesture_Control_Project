import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors

# ---------- Configuration ----------
EMBEDDING_DIM = 128
MARGIN = 1.0
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
MODEL_PATH = "models/embedding_net.pth"
GESTURE_DB_PATH = "gesture_db.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------------

class EmbeddingNet(nn.Module):
    """MLP that maps 63 landmarks → normalized embedding."""
    def __init__(self, input_dim=63, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)   # L2 normalize
        return x

def triplet_loss(anchor, positive, negative, margin=MARGIN):
    pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
    neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
    loss = torch.relu(pos_dist - neg_dist + margin)
    return loss.mean()

class LandmarkDataset(Dataset):
    """Simple dataset from lists of landmarks and labels."""
    def __init__(self, landmarks_list, labels_list):
        self.landmarks = torch.tensor(landmarks_list, dtype=torch.float32)
        self.labels = labels_list

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.labels[idx]

def create_triplets(dataset, num_triplets=5000):
    """Generate (anchor, positive, negative) indices from dataset."""
    labels = np.array(dataset.labels)
    unique_labels = np.unique(labels)
    label_to_indices = {lab: np.where(labels == lab)[0] for lab in unique_labels}
    triplets = []
    for _ in range(num_triplets):
        pos_class = np.random.choice(unique_labels)
        a_idx = np.random.choice(label_to_indices[pos_class])
        p_idx = np.random.choice(label_to_indices[pos_class])
        neg_class = np.random.choice([l for l in unique_labels if l != pos_class])
        n_idx = np.random.choice(label_to_indices[neg_class])
        triplets.append((a_idx, p_idx, n_idx))
    return triplets

class TripletDataset(Dataset):
    def __init__(self, base_dataset, triplets):
        self.base = base_dataset
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        return self.base[a][0], self.base[p][0], self.base[n][0]

def train_embedding_model(landmarks_list, labels_list):
    """
    Train the embedding network using triplet loss.
    landmarks_list: list of [63] floats
    labels_list: list of integer labels (must be 0..N-1)
    Returns trained model.
    """
    if len(set(labels_list)) < 2:
        print("Need at least 2 classes for training.")
        return None

    dataset = LandmarkDataset(landmarks_list, labels_list)
    triplets = create_triplets(dataset)
    triplet_dataset = TripletDataset(dataset, triplets)
    loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EmbeddingNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for a, p, n in loader:
            a, p, n = a.to(DEVICE), p.to(DEVICE), n.to(DEVICE)
            optimizer.zero_grad()
            emb_a = model(a)
            emb_p = model(p)
            emb_n = model(n)
            loss = triplet_loss(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    return model

class GestureDB:
    """
    Manages reference embeddings, labels, and actions.
    Uses k‑NN for inference.
    """
    def __init__(self, model=None, db_path=GESTURE_DB_PATH):
        self.model = model if model else EmbeddingNet().to(DEVICE)
        self.model.eval()
        self.db_path = db_path
        self.embeddings = []       # list of numpy arrays
        self.labels = []            # list of gesture names (strings)
        self.class_to_action = {}   # gesture name → action string
        self.knn = None
        self.load()

    def _compute_embedding(self, landmarks):
        """landmarks: list of 63 floats → embedding vector."""
        with torch.no_grad():
            x = torch.tensor(landmarks, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            emb = self.model(x).cpu().numpy().flatten()
        return emb

    def add_gesture_samples(self, gesture_name, action, list_of_landmarks):
        """
        Add new samples for an existing or new gesture.
        - gesture_name: string
        - action: string (will overwrite previous action for this gesture)
        - list_of_landmarks: list of [63] lists
        """
        # Update action mapping
        self.class_to_action[gesture_name] = action
        # Compute and store embeddings
        for lm in list_of_landmarks:
            emb = self._compute_embedding(lm)
            self.embeddings.append(emb)
            self.labels.append(gesture_name)
        self._build_index()
        self.save()

    def remove_gesture(self, gesture_name):
        """Remove all samples and the action mapping for a gesture."""
        # Filter out embeddings and labels
        new_emb = []
        new_lab = []
        for emb, lab in zip(self.embeddings, self.labels):
            if lab != gesture_name:
                new_emb.append(emb)
                new_lab.append(lab)
        self.embeddings = new_emb
        self.labels = new_lab
        if gesture_name in self.class_to_action:
            del self.class_to_action[gesture_name]
        if self.embeddings:
            self._build_index()
        else:
            self.knn = None
        self.save()

    def update_action(self, gesture_name, new_action):
        """Change the action associated with a gesture."""
        if gesture_name in self.class_to_action:
            self.class_to_action[gesture_name] = new_action
            self.save()

    def _build_index(self):
        if len(self.embeddings) == 0:
            self.knn = None
            return
        X = np.array(self.embeddings)
        self.knn = NearestNeighbors(n_neighbors=min(5, len(X)), metric='euclidean')
        self.knn.fit(X)

    def predict(self, landmarks, k=1):
        """
        Return (gesture_name, action) for the given landmarks.
        If k>1, majority vote among k nearest neighbors.
        """
        if self.knn is None:
            return None, None
        query = self._compute_embedding(landmarks).reshape(1, -1)
        distances, indices = self.knn.kneighbors(query, n_neighbors=k)
        indices = indices[0]
        neighbor_labels = [self.labels[i] for i in indices]
        if k == 1:
            pred = neighbor_labels[0]
        else:
            from collections import Counter
            pred = Counter(neighbor_labels).most_common(1)[0][0]
        action = self.class_to_action.get(pred, "")
        return pred, action

    def save(self):
        data = {
            'embeddings': self.embeddings,
            'labels': self.labels,
            'class_to_action': self.class_to_action
        }
        with open(self.db_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        if not os.path.exists(self.db_path):
            return
        with open(self.db_path, 'rb') as f:
            data = pickle.load(f)
        self.embeddings = data['embeddings']
        self.labels = data['labels']
        self.class_to_action = data['class_to_action']
        self._build_index()

    def rebuild_from_samples(self, samples_list):
        """
        samples_list: list of (gesture_name, landmarks) tuples
        Use current model to recompute all embeddings and rebuild index.
        """
        self.embeddings = []
        self.labels = []
        # Keep existing class_to_action (maybe we want to preserve actions)
        for gname, lm in samples_list:
            emb = self._compute_embedding(lm)
            self.embeddings.append(emb)
            self.labels.append(gname)
        self._build_index()
        self.save()