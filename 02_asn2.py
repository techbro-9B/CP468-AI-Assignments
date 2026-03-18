"""
CP468 - A2

"The overall structure and code commenting of this assignment were assisted by Claude Sonnet 4.6 (Anthropic, 2025), a large language 
model. All algorithms, analysis, and written responses were reviewed and verified by the authors."

Group Members:
Ali Khairreddin, #169074484
Adam Narciso, #169071458
Daniel Gonzalez, #169060144 
Lloyd Nsambu, #169113748
Maor Ethan Chernitzky, #169107770
Roneet Topiwala, #169048411
Suleman Ali, #169044667
Zoya Adnan, #169063952 
"""

import csv
import math
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
#  Output directory — same folder as this script
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def out(filename):
    return os.path.join(SCRIPT_DIR, filename)

def data_path(filename):
    return os.path.join(SCRIPT_DIR, filename)

# ─────────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────────

def read_csv(path, numeric_cols):
    rows = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(row[k]) for k in numeric_cols})
    return rows


def euclidean(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


# ===================================================================
#  PROBLEM 1 — kNN
# ===================================================================

def load_q1():
    rows = read_csv(data_path('CustomerDataset_Q1.csv'), ['x1', 'x2', 'c'])
    return [[r['x1'], r['x2'], int(r['c'])] for r in rows]


def q1a_scatter(dataset):
    blues = [(d[0], d[1]) for d in dataset if d[2] == 0]
    reds  = [(d[0], d[1]) for d in dataset if d[2] == 1]

    _, ax = plt.subplots(figsize=(7, 5))
    if blues:
        bx, by = zip(*blues)
        ax.scatter(bx, by, c='blue', label='y=0 (Not Interested)', s=80, zorder=3)
    if reds:
        rx, ry = zip(*reds)
        ax.scatter(rx, ry, c='red', label='y=1 (Interested)', s=80, zorder=3)

    ax.set_xlabel('x1: Avg Amount Spent per Purchase ($)')
    ax.set_ylabel('x2: Frequency of Purchases per Month')
    ax.set_title('Q1a: Customer Data Scatter Plot')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out('q1a_scatter.png'), dpi=150)
    plt.close()
    print("Q1a: scatter plot saved -> q1a_scatter.png")


def fnKNN(dataset, new_point, k):
    distances = []
    for row in dataset:
        d = euclidean([row[0], row[1]], new_point)
        distances.append((d, int(row[2])))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    votes = [label for _, label in k_nearest]
    return 1 if votes.count(1) >= votes.count(0) else 0


def evaluate_knn(dataset, train_frac, k, seed=42):
    random.seed(seed)
    shuffled = dataset[:]
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    train = shuffled[:n_train]
    test  = shuffled[n_train:]

    correct = sum(
        1 for row in test
        if fnKNN(train, [row[0], row[1]], k) == int(row[2])
    )
    return correct / len(test) if test else 0.0, len(train), len(test)


def q1_accuracy_table(dataset):
    splits = [0.80, 0.60, 0.50]
    ks     = [1, 2, 3, 4]

    print("\nQ1c/d — Accuracy Table")
    print(f"{'Split':<8}", end='')
    for k in ks:
        print(f"{'k='+str(k):>10}", end='')
    print()
    print('-' * 48)

    results = {}
    for frac in splits:
        label = f"{int(frac*100)}/{100 - int(frac*100)}"
        print(f"{label:<8}", end='')
        for k in ks:
            acc, n_tr, n_te = evaluate_knn(dataset, frac, k)
            results[(frac, k)] = acc
            print(f"{acc:>10.2%}", end='')
        print()

    return results


# ===================================================================
#  PROBLEM 2 — K-Means Clustering
# ===================================================================

def load_q2():
    rows = read_csv(data_path('CustomerProfiles_Q2.csv'), ['f1', 'f2'])
    return [[r['f1'], r['f2']] for r in rows]


def q2a_scatter(data_pts):
    x = [p[0] for p in data_pts]
    y = [p[1] for p in data_pts]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, c='steelblue', s=80, edgecolors='k', linewidths=0.5)
    ax.set_xlabel('x1: Avg Monthly Spending')
    ax.set_ylabel('x2: Total Number of Purchases')
    ax.set_title('Q2A: Customer Profiles Before Clustering')
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out('q2a_scatter.png'), dpi=150)
    plt.close()
    print("Q2A: pre-cluster scatter saved -> q2a_scatter.png")


def kmeans(data_pts, k):
    """
    K-Means clustering from scratch.
    Step 1 : Use first k data points as initial centroids.
    Steps 2-4: Assign, label, recompute centroids.
    Step 5 : Repeat until convergence.
    Returns : (labels, centroids)
    """
    centroids = [p[:] for p in data_pts[:k]]
    labels = [0] * len(data_pts)

    for iteration in range(1000):
        new_labels = []
        for p in data_pts:
            dists = [euclidean(p, c) for c in centroids]
            new_labels.append(dists.index(min(dists)))

        new_centroids = []
        for cid in range(k):
            members = [data_pts[i] for i, l in enumerate(new_labels) if l == cid]
            if members:
                centroid = [sum(col) / len(members) for col in zip(*members)]
            else:
                centroid = centroids[cid]
            new_centroids.append(centroid)

        if new_centroids == centroids and new_labels == labels:
            labels = new_labels
            centroids = new_centroids
            print(f"  K-Means converged in {iteration + 1} iteration(s).")
            break

        labels = new_labels
        centroids = new_centroids

    return labels, centroids


def q2c_plot(data_pts, labels, centroids, k=2):
    colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:red']

    _, ax = plt.subplots(figsize=(7, 5))
    for cid in range(k):
        pts = [data_pts[i] for i, l in enumerate(labels) if l == cid]
        if pts:
            cx, cy = zip(*pts)
            ax.scatter(cx, cy, c=colors[cid],
                       label=f'Cluster {cid + 1}', s=80,
                       edgecolors='k', linewidths=0.5)

    for i, c in enumerate(centroids):
        ax.scatter(c[0], c[1], marker='X', s=200,
                   c=colors[i], edgecolors='black', linewidths=1.5,
                   zorder=5, label=f'Centroid {i + 1}')

    ax.set_xlabel('x1: Avg Monthly Spending')
    ax.set_ylabel('x2: Total Number of Purchases')
    ax.set_title(f'Q2C: K-Means Clustering (k={k})')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out('q2c_clusters.png'), dpi=150)
    plt.close()
    print("Q2C: cluster plot saved -> q2c_clusters.png")


def q2_report(data_pts, labels, centroids, k=2):
    print(f"\nQ2D — Cluster sizes (k={k}):")
    for cid in range(k):
        count = labels.count(cid)
        print(f"  Cluster {cid + 1}: {count} data points")

    print(f"\nQ2E — Final centroids (k={k}):")
    for i, c in enumerate(centroids):
        print(f"  Centroid {i + 1}: x1={c[0]:.4f}, x2={c[1]:.4f}")


# ===================================================================
#  PROBLEM 3 — Perceptron (Rain Prediction)
# ===================================================================

def load_q3():
    rows = read_csv(data_path('WeatherData_Q3.csv'), ['temp', 'humid', 'rain'])
    return [[r['temp'], r['humid'], int(r['rain'])] for r in rows]


def q3a_scatter(dataset):
    no_rain = [(d[0], d[1]) for d in dataset if d[2] == 0]
    rain    = [(d[0], d[1]) for d in dataset if d[2] == 1]

    _, ax = plt.subplots(figsize=(7, 5))
    if no_rain:
        nx, ny = zip(*no_rain)
        ax.scatter(nx, ny, c='blue', marker='s', s=90, label='y=0 (No Rain)', zorder=3)
    if rain:
        rx, ry = zip(*rain)
        ax.scatter(rx, ry, c='red', marker='o', s=90, label='y=1 (Rain)', zorder=3)

    ax.set_xlabel('x1: Temperature (scaled 0-1)')
    ax.set_ylabel('x2: Humidity (scaled 0-1)')
    ax.set_title('Q3a: Weather Data with Rain vs No Rain')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out('q3a_scatter.png'), dpi=150)
    plt.close()
    print("Q3a: scatter saved -> q3a_scatter.png")


class Perceptron:
    """
    Single-layer Perceptron with bias and step activation.
    Weights and bias initialised uniformly in [-0.5, 0.5].
    Update rule: w <- w + lr * (y - y_hat) * x
                 b <- b + lr * (y - y_hat)
    Stops early when 0 misclassifications on the training set.
    """
    def __init__(self, n_features, lr=0.1, seed=42):
        random.seed(seed)
        self.w  = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.b  = random.uniform(-0.5, 0.5)
        self.lr = lr

    def predict_one(self, x):
        net = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return 1 if net >= 0 else 0

    def train(self, X, y, max_iter=1000):
        for _ in range(max_iter):
            errors = 0
            for xi, yi in zip(X, y):
                y_hat = self.predict_one(xi)
                diff  = yi - y_hat
                if diff != 0:
                    errors += 1
                    self.w = [wi + self.lr * diff * xij
                              for wi, xij in zip(self.w, xi)]
                    self.b += self.lr * diff
            if errors == 0:
                break

    def accuracy(self, X, y):
        preds   = [self.predict_one(xi) for xi in X]
        correct = sum(1 for p, t in zip(preds, y) if p == t)
        return correct / len(y)


def run_perceptron(dataset, train_size, lr=0.1, seed=42, label=""):
    X = [[d[0], d[1]] for d in dataset]
    y = [d[2] for d in dataset]

    X_train, y_train = X[:train_size], y[:train_size]
    X_test,  y_test  = X[train_size:],  y[train_size:]

    p = Perceptron(n_features=2, lr=lr, seed=seed)
    p.train(X_train, y_train)

    train_acc = p.accuracy(X_train, y_train)
    test_acc  = p.accuracy(X_test,  y_test)
    print(f"  {label:<38} Train Acc={train_acc:.2%}, Test Acc={test_acc:.2%}")
    return p, train_acc, test_acc


def q3b_perceptron(dataset):
    print("\nQ3b — Primary split (first 15 train, last 5 test, lr=0.1):")
    p_main, _, _ = run_perceptron(
        dataset, train_size=15, lr=0.1, label="15/5 split, lr=0.1 |")
    return p_main


def q3c_evaluate(dataset, p_main):
    print("\nQ3c — Additional train/test splits and learning rates:")
    run_perceptron(dataset, train_size=12, lr=0.1,  label="12/8  split, lr=0.1  |")
    run_perceptron(dataset, train_size=10, lr=0.1,  label="10/10 split, lr=0.1  |")
    run_perceptron(dataset, train_size=15, lr=0.01, label="15/5  split, lr=0.01 |")
    run_perceptron(dataset, train_size=15, lr=0.5,  label="15/5  split, lr=0.5  |")

    # Decision boundary plot
    no_rain = [(d[0], d[1]) for d in dataset if d[2] == 0]
    rain    = [(d[0], d[1]) for d in dataset if d[2] == 1]

    _, ax = plt.subplots(figsize=(7, 5))
    if no_rain:
        nx, ny = zip(*no_rain)
        ax.scatter(nx, ny, c='blue', marker='s', s=90, label='y=0 (No Rain)', zorder=3)
    if rain:
        rx, ry = zip(*rain)
        ax.scatter(rx, ry, c='red', marker='o', s=90, label='y=1 (Rain)', zorder=3)

    w0, w1 = p_main.w
    b = p_main.b
    if abs(w1) > 1e-9:
        x1_vals = [0.0, 1.0]
        x2_vals = [-(w0 * x1 + b) / w1 for x1 in x1_vals]
        ax.plot(x1_vals, x2_vals, 'k--', linewidth=2, label='Decision Boundary')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x1: Temperature (scaled 0-1)')
    ax.set_ylabel('x2: Humidity (scaled 0-1)')
    ax.set_title('Q3c:Perceptron Decision Boundary')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out('q3c_boundary.png'), dpi=150)
    plt.close()
    print("Q3c: decision boundary saved -> q3c_boundary.png")


# ===================================================================
#  MAIN
# ===================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PROBLEM 1 — kNN")
    print("=" * 60)
    ds1 = load_q1()
    q1a_scatter(ds1)
    pred = fnKNN(ds1[:16], [2.0, 1.5], k=3)
    print(f"Q1b demo: fnKNN([2.0, 1.5], k=3) -> predicted class = {pred}")
    results = q1_accuracy_table(ds1)
    best = max(results, key=results.get)
    print(f"\nQ1e: Best combo -> split={int(best[0]*100)}/{100-int(best[0]*100)}, "
          f"k={best[1]}, accuracy={results[best]:.2%}")

    print("\n" + "=" * 60)
    print("PROBLEM 2 — K-Means")
    print("=" * 60)
    ds2 = load_q2()
    q2a_scatter(ds2)
    labels, centroids = kmeans(ds2, k=2)
    q2c_plot(ds2, labels, centroids)
    q2_report(ds2, labels, centroids)

    print("\n" + "=" * 60)
    print("PROBLEM 3 — Perceptron")
    print("=" * 60)
    ds3 = load_q3()
    q3a_scatter(ds3)
    p_main = q3b_perceptron(ds3)
    q3c_evaluate(ds3, p_main)

    print("\nPNG of graphs saved to:", SCRIPT_DIR)
