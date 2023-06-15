import numpy as np
import matplotlib.pyplot as plt

font = {
    "size": 28
}

index = np.arange(100)
data = np.sin(index)

if __name__ == "__main__":
    plt.figure(figsize=(8, 6))
    plt.plot(data)
    plt.ylabel("data", font)
    plt.xlabel("index", font)
    # plt.legend(["正弦曲线"], fontsize=28)
    save_path = "./font_size_test.jpg"
    plt.savefig(save_path, dpi=1000, bbox_inches="tight")