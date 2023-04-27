import numpy as np
import matplotlib.pyplot as plt

def f(x, a, b, c, d):
    return a * np.sin(b * (x + c)) + d

def g(x, a, b, c, d):
    return a * np.sin(b * (x + c)) + d

def main():

    num_samples = 1000

    #Funktion als 1.Bild
    x = np.random.uniform(0, 4 * np.pi, num_samples)
    y_wahrheit = f(x, 1, 2, 0, 0)
    y_pred = g(x, 0.5, 2, 0, 0)

    expected_g = np.mean(y_pred)

    # bias und Varianz als 2. Bild
    bias = np.mean((expected_g - y_wahrheit) ** 2)
    variance = np.mean((y_pred - expected_g) ** 2)

    # Erstellt die Figure und die beiden Subplots
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    # Fügt das erste Bild zum ersten Subplot hinzu
    ax1.plot(x, y_wahrheit, label='Target Function')
    ax1.plot(x, y_pred, label='Model Prediction')
    ax1.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi, 5 * np.pi / 2, 3 * np.pi, 7 * np.pi / 2, 4 * np.pi])
    ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π', '5π/2', '3π', '7π/2', '4π'])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Model and Truth')
    ax1.legend()

    # Fügt das zweite Bild zum zweiten Subplot hinzu
    ax2.plot(x, bias, label='bias')
    ax2.plot(x,variance,label='variance')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Bias and Variance')
    ax2.legend()

    # Zeigt die plt-Figur
    plt.show()

if __name__ == '__main__':
    main()