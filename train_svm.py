import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

label_mapping = {
    'lang2': 0,
    'other': 1,
    'lang1': 2,
    'ne': 3,
    'unk': 4,
    'mixed': 5,
}

def load_embeddings(data_dir, batch_size=100):  
    embeddings = []
    labels = []
    total_embeddings = 0
    total_labels = 0

    files = list(data_dir.glob("*.npz"))
    
    for idx, file in enumerate(files):
        data = np.load(file)
        num_embeddings = data['embeddings'].shape[0]
        num_labels = len(data['labels'])

        valid_embeddings = []
        valid_labels = []

        for i in range(num_embeddings):
            label = data['labels'][i]
            if label == 'PAD':
                continue

            if label in label_mapping:
                valid_embeddings.append(data['embeddings'][i])
                valid_labels.append(label_mapping[label])

        embeddings.extend(valid_embeddings)
        labels.extend(valid_labels)

        total_embeddings += len(valid_embeddings)
        total_labels += len(valid_labels)

        if (idx + 1) % batch_size == 0 or (idx + 1) == len(files):
            yield np.vstack(embeddings), np.array(labels, dtype=np.int32)
            embeddings, labels = [], []

train_dir = Path('embeddings/train')
dev_dir = Path('embeddings/dev')

train_embeddings, train_labels = next(load_embeddings(train_dir, batch_size=200))
dev_embeddings, dev_labels = next(load_embeddings(dev_dir, batch_size=100))

train_embeddings = np.array([embedding.flatten() for embedding in train_embeddings])
dev_embeddings = np.array([embedding.flatten() for embedding in dev_embeddings])

X_train, X_val, y_train, y_val = train_test_split(train_embeddings, train_labels, test_size=0.2, random_state=42)

classifier = SVC(kernel="linear", C=1.0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_val)
print("Classification Report (SVM):")
print(classification_report(y_val, y_pred))
