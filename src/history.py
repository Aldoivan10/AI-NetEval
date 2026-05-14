from typing import Any
from config.settings import config
from keras import models, preprocessing

import matplotlib.pyplot as plt
import numpy as np
import json
import sys

plt.style.use("pacoty.mplstyle")

args = sys.argv
model = config.MODEL_PATH if len(args) == 1 else config.MODEL_ROOT_PATH / args[1]

# Abrir el historial
with open(model / "history.json", "r") as file:
    history_loaded = json.load(file)

# Extraer la pérdida y precisión
loss = history_loaded["loss"]
val_loss = history_loaded["val_loss"]
accuracy = history_loaded["accuracy"]
val_accuracy = history_loaded["val_accuracy"]

# Encontrar la época con la menor pérdida de validación
min_val_loss_epoch = np.argmin(val_loss) + 1  # +1 porque las épocas empiezan en 1
min_val_loss_value = np.min(val_loss)

# Valores finales (última época)
final_train_loss = loss[-1]
final_val_loss = val_loss[-1]
final_train_acc = accuracy[-1]
final_val_acc = val_accuracy[-1]

# Extraer datos finales del entrenamiento
cnn: Any = models.load_model(config.MODEL)
test_ds: Any = preprocessing.image_dataset_from_directory(
    config.TEST_PATH, color_mode="grayscale", shuffle=False
)
test_loss, test_acc = cnn.evaluate(test_ds, verbose=0)
print("\n" + "=" * 60)
print("           PÉRDIDA Y PRECISIÓN FINALES")
print("=" * 60)
print(f"{'Conjunto':<20} {'Pérdida':<12} {'Precisión':<12}")
print("-" * 60)
print(f"{'Entrenamiento':<20} {final_train_loss:<12.4f} {final_train_acc:<12.4f}")
print(f"{'Validación':<20} {final_val_loss:<12.4f} {final_val_acc:<12.4f}")
print(f"{'Prueba':<20} {test_loss:<12.4f} {test_acc:<12.4f}")
print("=" * 60)

# Crear figura
plt.figure(figsize=(14, 5))

# ==================  GRÁFICO DE PÉRDIDA  ==================
plt.subplot(1, 2, 1)
plt.plot(loss, label="Entrenamiento", linewidth=2)
plt.plot(val_loss, label="Validación", linewidth=2)

# Línea punteada en el mínimo de val_loss
plt.axhline(
    y=min_val_loss_value,
    color="orange",
    linestyle="--",
    linewidth=1,
    label=f"Mín. val_loss = {min_val_loss_value:.4f} (época {min_val_loss_epoch})",
)

# Anotación del valor final de val_loss
plt.annotate(
    f"{final_val_loss:.4f}",
    xy=(len(val_loss) - 1, final_val_loss),
    xytext=(len(val_loss) - 3.5, final_val_loss + 2),
    arrowprops=dict(arrowstyle="->", color="orange", lw=2),
    fontsize=10,
    color="orange",
    fontweight="bold",
)

plt.title("Evolución de la pérdida")
plt.xlabel("Épocas")
plt.ylabel("Pérdida (entropía cruzada categórica)")
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)

# ==================  GRÁFICO DE PRECISIÓN  ==================
plt.subplot(1, 2, 2)
plt.plot(accuracy, label="Entrenamiento", linewidth=2)
plt.plot(val_accuracy, label="Validación", linewidth=2)

# Línea punteada en el valor máximo de precisión de validación
max_val_acc = np.max(val_accuracy)
max_val_acc_epoch = np.argmax(val_accuracy) + 1
plt.axhline(
    y=max_val_acc,
    color="green",
    linestyle="--",
    linewidth=1,
    label=f"Máx. val_acc = {max_val_acc:.4f} (época {max_val_acc_epoch})",
)

# Anotación de los valores finales
plt.annotate(
    text=f"{final_val_acc:.4f}",
    xy=(len(val_accuracy) - 1, final_val_acc),
    xytext=(len(val_accuracy) - 6, final_val_acc - 0.15),
    arrowprops=dict(arrowstyle="->", color="green", lw=2),
    fontsize=10,
    color="green",
    fontweight="bold",
)

plt.title("Evolución de la precisión")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend(loc="lower right")
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(config.MODEL_PATH / "history.png", dpi=300, bbox_inches="tight")
plt.show()
