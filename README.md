# Clasificación de imágenes de indumentaria — Fashion-MNIST

Proyecto grupal de la materia Laboratorio de Datos, FCEyN – Universidad de Buenos Aires (UBA).

## Descripción

Trabajo sobre el dataset Fashion-MNIST (10 clases de prendas de ropa). Incluye análisis exploratorio de las clases, un clasificador binario (clases 0 y 8) con **KNN**, y un clasificador multiclase con **Árboles de Decisión**, con selección de hiperparámetros por validación cruzada (K-folding).

## Resultados

- **KNN (binario):** accuracy de **0.9446** con 10 píxeles seleccionados y k=6.
- **Árbol de Decisión (multiclase):** accuracy de **0.8133** sobre held-out, con criterio entropía y profundidad máxima 10.

## Herramientas

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · DuckDB

## Informe

El análisis completo, con las conclusiones detalladas, está en en [Informe TP-02 Laboratorio de datos.pdf](./Informe%20TP-02%20Laboratorio%20de%20datos.pdf).

## Autoría

Proyecto grupal realizado junto a Juana Maria Leiton y Guadalupe Cataneo. Las consignas y herramientas a utilizar fueron definidas por la cátedra; el desarrollo, análisis y código fueron elaborados por el grupo.
