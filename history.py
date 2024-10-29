import matplotlib.pyplot as plt
import json
import sys

plt.style.use("./pacoty.mplstyle")

args = sys.argv
model = 'mish_augmented' if len(args) == 1 else args[1]
path = f"./model/{model}"

# Mostrar la estructura del modelo
with open(f'{path}/summary.txt') as f:
    print(f.read())

# Abrir el historial
with open(f"{path}/history.json", 'r') as file:
    history_loaded = json.load(file)

# Extraer la pérdida y precisión
loss = history_loaded['loss']
val_loss = history_loaded['val_loss']
accuracy = history_loaded['accuracy']
val_accuracy = history_loaded['val_accuracy']

# Graficar la pérdida
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss, label='Pérdida de Entrenamiento')
plt.plot(val_loss, label='Pérdida de Validación')
plt.title('Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Graficar la precisión
plt.subplot(1, 2, 2)
plt.plot(accuracy, label='Precisión de Entrenamiento')
plt.plot(val_accuracy, label='Precisión de Validación')
plt.title('Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()