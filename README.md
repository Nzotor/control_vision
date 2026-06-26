# 🌊 Système de Vision Embarqué pour AUV (BlueROV) - Reconnaissance de Gestes

Ce dépôt contient l'ensemble du code et des modèles pour un système de reconnaissance de gestes sous-marins basé sur YOLOv8. L'objectif est de permettre à un plongeur de contrôler un BlueROV via des signes.

## 📋 Architecture du Projet

Le projet explore et compare deux approches d'inférence, entraînées sur des jeux de données de plongée :
1. **Architecture "Modèle Unique" (Multiclasses) :** Un seul réseau YOLOv8n qui détecte la main ET classifie le geste simultanément. *(Solution retenue pour l'embarqué)*
2. **Architecture "Cascade" (2 Modèles) :** Un détecteur de main (YOLOv8n) suivi d'un algorithme de "Smart Crop" (redimensionnement dynamique), puis d'un classifieur (YOLOv8-cls).

## 📂 Contenu du Dépôt

* `/notebooks/` : Les scripts Google Colab utilisés pour l'entraînement des modèles sur le dataset.
* `/test/test_minimal.py` : Script de "Sanity Check" pour valider rapidement l'environnement et le chargement des poids sur une image fixe.
* `/scripts/benchmark_rov.py` : Outil de test comparant les FPS et la latence des deux architectures sous différents formats d'export (PyTorch, ONNX, NCNN).
* `/scripts/integration_rov.py` : La boucle principale de contrôle. Intègre une Machine à États (Standby/Listening) déclenchée par des gestes clés (`start_comm`, `end_comm`) et un filtre pour éliminer les faux positifs.

## 🚀 Installation & Prérequis

Le code embarqué est conçu pour tourner sur une machine cible (ex: Raspberry Pi) connectée au flux caméra du ROV.

```bash
# Cloner le dépôt
git clone https://github.com/Nzotor/control_vision
cd control_vision

# Installer les dépendances
pip install ultralytics opencv-python matplotlib
```

## ⚙️ Utilisation

Etape 1 : Test Rapide (Sanity Check)

```bash
python scripts/test_minimal.py
```

Etape 2 : lancer le système de controle

```bash
python scripts/integration_rov.py
```

**Contrôles clavier (Debug) pendant l'exécution :**

* Touche s : Forcer le passage en mode "Listening" (Écoute).
* Touche e : Forcer le retour en mode "Standby" et afficher le résumé.
* Touche q : Quitter le programme proprement.
