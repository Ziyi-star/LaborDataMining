import numpy as np
import matplotlib.pyplot as plt

def f(x, a, b, c, d):
    return a * np.sin(b * (x + c)) + d


def g(x, a, b, c, d):
    return a * np.sin(b * (x + c)) + d


def main():
    # Funktion als 1.Bild
    x = np.linspace(0, 4 * np.pi, 10000)
    y_wahrheit = f(x, 1, 2, 0, 0)
    # low Varance and low Bias
    y_pred_lvlb = g(x, 0.5, 2, 0, 0)
    # low Varance and high Bias
    y_pred_lvhb = g(x, 1, 2, 0, 1)
    # High Varance and low Bias
    y_pred_hvlb = g(x, 1, 10, 0, 0)
    # High Varance and high Bias
    y_pred_hvhb = g(x, 0.5, 5, np.pi / 2, 2)

    # erwartungswert
    expected_g_lvlb = np.mean(y_pred_lvlb)
    expected_g_lvhb = np.mean(y_pred_lvhb)
    expected_g_hvlb = np.mean(y_pred_hvlb)
    expected_g_hvhb = np.mean(y_pred_hvhb)

    # bias und Varianz als 2. Bild
    bias_lvlb = np.mean((expected_g_lvlb - y_wahrheit) ** 2)
    variance_lvlb = np.mean((y_pred_lvlb - expected_g_lvlb) ** 2)
    bias_lvhb = np.mean((expected_g_lvhb - y_wahrheit) ** 2)
    variance_lvhb = np.mean((y_pred_lvhb - expected_g_lvhb) ** 2)
    bias_hvlb = np.mean((expected_g_hvlb - y_wahrheit) ** 2)
    variance_hvlb = np.mean((y_pred_hvlb - expected_g_hvlb) ** 2)
    bias_hvhb = np.mean((expected_g_hvhb - y_wahrheit) ** 2)
    variance_hvhb = np.mean((y_pred_hvhb - expected_g_hvhb) ** 2)

    # plotten
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

    # 1. Bild
    ax1.plot(x, y_wahrheit, label='Target Function')
    ax1.plot(x, y_pred_lvlb, label='Model Prediction with low varance and low bias')
    ax1.plot(x, y_pred_lvhb, label='Model Prediction with low varance and high bias')
    ax1.plot(x, y_pred_hvlb, label='Model Prediction with high varance and low bias')
    ax1.plot(x, y_pred_hvhb, label='Model Prediction with high varance and high bias')
    ax1.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi, 5 * np.pi / 2, 3 * np.pi, 7 * np.pi / 2, 4 * np.pi])
    ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π', '5π/2', '3π', '7π/2', '4π'])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()

    # 2. Bild
    ax2.bar(['Low Variance and Low Bias', 'Low Variance and High Bias', 'High Variance and Low Bias',
             'High Variance and High Bias'], [variance_lvlb, variance_lvhb, variance_hvlb, variance_hvhb],
            label='Variance')
    ax2.bar(['Low Variance and Low Bias', 'Low Variance and High Bias', 'High Variance and Low Bias',
             'High Variance and High Bias'], [bias_lvlb, bias_lvhb, bias_hvlb, bias_hvhb],
            bottom=[variance_lvlb, variance_lvhb, variance_hvlb, variance_hvhb], label='Bias')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Error')
    ax2.set_title('Bias-Variance Decomposition')
    ax2.legend()

    # Zeigt die plt-Figur
    plt.show()


if __name__ == '__main__':
    main()
