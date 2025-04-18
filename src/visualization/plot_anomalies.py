
import matplotlib.pyplot as plt

def plot_series_with_anomalies(timestamps, values, anomalies, title=""):
    plt.figure(figsize=(14, 5))
    plt.plot(timestamps, values, label="Valeurs")
    plt.scatter(timestamps[anomalies], values[anomalies], color="red", label="Anomalies")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.legend()
    plt.tight_layout()
    plt.show()
