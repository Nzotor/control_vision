from ultralytics import YOLO
import cv2

print("=== DÉMARRAGE DU TEST MINIMAL ===")

# Chargement du modèle
chemin_modele = "best_modele_unique.onnx"
print(f"Chargement du modèle : {chemin_modele}")
model = YOLO(chemin_modele)

# Définition de l'image source
chemin_image = "test_image.jpg"
print(f"Analyse de l'image : {chemin_image}")

# L'inférence pure (Le fameux model.predict)
resultats = model.predict(source=chemin_image, save=True, conf=0.5)

# Extraction et affichage du résultat brut dans la console
prediction = resultats[0]
print("\n=== RÉSULTATS DE L'IA ===")

if len(prediction.boxes) > 0:
    for box in prediction.boxes:
        class_id = int(box.cls[0].item())
        confiance = box.conf[0].item()
        nom_geste = prediction.names[class_id]
        
        print(f"-> Geste détecté : {nom_geste.upper()} (Confiance : {confiance:.2f})")
else:
    print("-> Aucun geste détecté sur cette image.")

print("\nUne image avec le résultat visuel a été sauvegardée dans le dossier 'runs/detect/predict/'")